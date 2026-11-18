"""Download every closed Brill field event with each table's auction.

Run on Windows with the Elo venv (it has the Brill sign-in through postmortem-brill):

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/fetch_brill_field.py OUT.jsonl [--full]

When OUT.jsonl exists, only events that are not in it yet are downloaded and
added (a closed event does not change); --full downloads everything again.

One line per event and board:
  {tournament_id, event, scoring, date, board, dealer, vul, deal,
   tables: [{player_id, name, type, auction, contract, declarer, tricks, ns_score}]}

`deal` is 'N:north east south west'; `auction` is the list of calls from the
dealer on, as P / X / XX / 1C..7N; `vul` is None / NS / EW / All.
"""
import json
import pathlib
import re
import sys
from collections import Counter
from datetime import datetime

from tqdm import tqdm

SRC = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC / 'elo'))
sys.path.insert(0, str(SRC / 'postmortem-brill'))

import brill_client  # noqa: E402
import brill_pbn  # noqa: E402
import elo_brill  # noqa: E402

RESULT_COLUMNS = (
    'board_number,contract,declarer,tricks,ns_score,dealer,vulnerability,deal_pbn,auction_pbn,'
    'participant_id,participant:challenge_participants!participant_id('
    'id,display_name,participant_type,robot_name,user_id,join_order)'
)
SEATS = 'NESW'


def calls_of(auction):
    """The stored auction as a list of calls: P, X, XX, 1C..7N."""
    text, _ = brill_pbn._format_auction(auction)
    calls = []
    for token in text.split():
        if re.fullmatch(r'=\d+=', token):
            continue
        token = token.upper()
        calls.append({'PASS': 'P'}.get(token, token.replace('NT', 'N')))
    return calls


def rotate_to_north(deal, dealer):
    """'E:h1 h2 h3 h4' (first hand = the named seat) -> 'N:north east south west'."""
    first, hands = deal[0], deal[2:].split()
    start = SEATS.index(first)
    by_seat = {SEATS[(start + i) % 4]: hand for i, hand in enumerate(hands)}
    return 'N:' + ' '.join(by_seat[s] for s in SEATS)


def most_common(values):
    counted = Counter(v for v in values if v not in (None, ''))
    return counted.most_common(1)[0][0] if counted else None


def main():
    out = pathlib.Path(sys.argv[1])
    full = '--full' in sys.argv[2:]
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    # events already in OUT are kept as they are: a closed event does not change
    known = {}
    if out.exists() and not full:
        for line in out.open(encoding='utf-8'):
            known.setdefault(json.loads(line)['tournament_id'], []).append(line)
    tournaments = elo_brill.list_closed_tournaments()
    ids = [t['id'] for t in tournaments if t['id'] not in known]
    print(f'{len(tournaments)} closed events, {len(known)} already in {out.name}, {len(ids)} to fetch')
    challenge_of = {}
    for i in range(0, len(ids), 40):
        chunk = f'in.({",".join(ids[i:i + 40])})'
        for row in brill_client._select('tournament_rounds', {
            'select': 'tournament_id,round_number,challenge_id',
            'tournament_id': chunk, 'challenge_id': 'not.is.null',
            'order': 'tournament_id,round_number',
        }):
            challenge_of.setdefault(row['tournament_id'], row['challenge_id'])

    boards = new_events = 0
    listed = {t['id'] for t in tournaments}
    temp = out.with_name(out.name + '.new')
    with temp.open('w', encoding='utf-8') as f:
        for tournament_id, lines in known.items():      # kept events the site no longer lists
            if tournament_id not in listed:
                f.writelines(lines)
        for t in tqdm(tournaments, disable=not ids):
            if t['id'] in known:
                f.writelines(known[t['id']])
                continue
            challenge_id = challenge_of.get(t['id'])
            if not challenge_id:
                continue
            results = brill_client._select('challenge_board_results', {
                'select': RESULT_COLUMNS, 'challenge_id': f'eq.{challenge_id}',
                'order': 'board_number.asc'})
            event_time = t.get('starts_at') or t.get('created_at') or ''
            new_events += 1
            for number in sorted({r['board_number'] for r in results}):
                rows = [r for r in results if r['board_number'] == number]
                with_deal = [r for r in rows if r.get('deal_pbn')]
                deal = most_common(r['deal_pbn'] for r in with_deal)
                if not deal:
                    continue
                same = [r for r in with_deal if r['deal_pbn'] == deal]
                dealer = most_common(r.get('dealer') for r in same)
                if not re.match(r'^[NESW]:', deal):
                    deal = f'{dealer}:{deal}'
                vul = most_common(brill_pbn.vulnerable_to_pbn(r.get('vulnerability')) for r in same)
                tables = []
                for r in rows:
                    if r.get('ns_score') is None:
                        continue
                    player_id, name, kind = elo_brill.player_identity(r.get('participant') or {})
                    tables.append({
                        'player_id': player_id, 'name': name, 'type': kind,
                        'auction': calls_of(r.get('auction_pbn')),
                        'contract': brill_pbn.contract_to_pbn(r.get('contract')).upper().replace(' ', ''),
                        'declarer': r.get('declarer') if r.get('declarer') in tuple(SEATS) else None,
                        'tricks': r.get('tricks'), 'ns_score': int(r['ns_score']),
                    })
                f.write(json.dumps({
                    'tournament_id': t['id'], 'event': t.get('name') or '',
                    'scoring': 'IMP' if t.get('scoring_method') == 'imp' else 'MP',
                    'date': event_time[:10], 'board': number, 'dealer': dealer, 'vul': vul,
                    'deal': rotate_to_north(deal, dealer), 'tables': tables,
                }, ensure_ascii=False) + '\n')
                boards += 1
    temp.replace(out)
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  events {len(tournaments)}  '
          f'new events {new_events}  new boards {boards}  -> {out}')


if __name__ == '__main__':
    main()
