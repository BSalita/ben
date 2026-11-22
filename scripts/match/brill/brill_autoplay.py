"""Have the Brill service play the cards of a replayed contract (its /autoplay with a fixed auction).

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_autoplay.py FIELD.jsonl REPLAY.jsonl OUT.jsonl [--no-requests]

--no-requests only copies other models' results of the same auctions (below) and
sends nothing: for a rebuild while another run holds the service.

For each board of REPLAY (brill_replay.py) the Brill robots play all four hands
in the auction our model reached. Boards that need no request:
  brill-auction   the Brill entrant had the same auction in the event: its result is Brill's play of it
  brill-contract  the Brill entrant played the same contract by the same declarer
  passout         nobody bid
  copied          another model's autoplay (a file next to OUT) has the same auction:
                  its line is copied, with copied_from
Every other board is sent to /autoplay, one request at a time (the service plays
live events too). The events our entrant played itself are played too: the
replay pages show the model on every event. OUT.jsonl gets one line per played board and the run can be
started again: boards already in OUT are skipped. Three failures in a row stop it (six when they
are gateway time-outs: those mark a board the service plays for more than 180 s; a time-out on a
board that failed before is not counted).
A board that failed twice is left out and keeps its double-dummy result.
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_brill_replay import contract_of, score_ns, table_contract  # noqa: E402

BASE = 'https://brillservice.aalborgdata.dk'
BRILL = 'R:Brill'
TIMEOUT = 300
GIVE_UP = 2       # attempts at a board before it is left out (the gateway cuts a request off at 180 s)
COOL_DOWN = 90    # seconds to wait after a failed request: the service may still be playing it


def is_daily(board):
    """A board of the daily series. field.jsonl also holds the other closed events (one so far:
    'EBU Sunday', 2026-08-15), which Brill's daily leaderboard does not count."""
    return board['event'].startswith('Daily')


ENTRANT = 'R:MortyAI'   # our model as a real entrant of the dailies (first event 2026-11-19)


def entered_events(field):
    """The events our model played itself: it has a table at every board of the event.
    For these the event's own table is our result on the main page; the replay pages
    show the replay there too."""
    boards, ours = {}, {}
    for b in field:
        boards[b['tournament_id']] = boards.get(b['tournament_id'], 0) + 1
        if any(t['player_id'] == ENTRANT for t in b['tables']):
            ours[b['tournament_id']] = ours.get(b['tournament_id'], 0) + 1
    return {t for t, n in ours.items() if n == boards[t]}


def brill_source(board, walk):
    """Where Brill's play of this auction's contract comes from, and the Brill entrant's table if it is used."""
    ours = contract_of(walk, board['dealer'])
    if ours is None:
        return 'passout', None
    table = next((t for t in board['tables'] if t['player_id'] == BRILL), None)
    if table is not None and table['auction'] == walk:
        return 'brill-auction', table
    if table is not None and table_contract(table) == ours:
        return 'brill-contract', table
    return 'autoplay', None


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'brill-replay'})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def autoplay(board, walk):
    query = {
        'board': board['board'], 'nosave': 'true', 'deal': board['deal'], 'dealer': board['dealer'],
        'vul': board['vul'], 'scoring': board['scoring'], 'nnOpeningLead': 'true', 'playInference': 'true',
        'external': 'S', 'auction': ','.join(walk),
    }
    data = get(f'{BASE}/autoplay?{urllib.parse.urlencode(query)}')
    if data.get('error'):
        raise RuntimeError(data['error'])
    ours = contract_of(walk, board['dealer'])
    contract = data['contract'].replace('NT', 'N')
    if contract.endswith('XXX'):
        contract = contract[:-1]    # the service writes a redoubled contract as X + XX: 2DXXX
    played = (contract, data['declarer'])
    expected = (f'{ours[0]}{ours[1]}{ours[2]}', ours[3])
    if played != expected:
        raise RuntimeError(f'the service played {played}, the auction gives {expected}')
    tricks = int(data['tricks'])
    return {'contract': expected[0], 'declarer': expected[1], 'tricks': tricks,
            'ns_score': score_ns(*ours, tricks, board['vul']), 'service_ns_score': data.get('nsScore'),
            'play': data.get('play'), 'timing': data.get('timing')}


def main():
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}', flush=True)
    field_path, replay_path, out_path = (pathlib.Path(p) for p in sys.argv[1:4])
    field = {(b['tournament_id'], b['board']): b for b in map(json.loads, field_path.open(encoding='utf-8'))}
    replay = [json.loads(line) for line in replay_path.open(encoding='utf-8')]
    done, failed = set(), {}
    if out_path.exists():
        for line in out_path.open(encoding='utf-8'):
            r = json.loads(line)
            key = (r['tournament_id'], r['board'])
            if 'error' not in r:
                done.add(key + (tuple(r['auction']),))   # a board replayed again with another auction is played again
            elif "XXX'" in r['error']:
                continue    # a redoubled contract refused by the contract check before 2026-11-19: not a failed attempt
            else:
                failed[key] = failed.get(key, 0) + 1
    given_up = {key for key, n in failed.items() if n >= GIVE_UP and key not in {d[:2] for d in done}}
    # the same auction already played for another model (another file of OUT's folder) is copied, not sent again
    elsewhere = {}
    for other in sorted(out_path.parent.glob('*.jsonl')):
        if other.resolve() == out_path.resolve():
            continue
        for line in other.open(encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r and 'tricks' in r:
                elsewhere.setdefault((r['tournament_id'], r['board'], tuple(r['auction'])), dict(r, copied_from=other.stem))
    todo, copied = [], []
    for r in replay:
        key = (r['tournament_id'], r['board'])
        board = field[key]
        if ('error' in r or key + (tuple(r['walk']),) in done or key in given_up or not is_daily(board)
                or brill_source(board, r['walk'])[0] != 'autoplay'):
            continue
        same = elsewhere.get(key + (tuple(r['walk']),))
        if same is not None:
            copied.append(same)
            done.add(key + (tuple(r['walk']),))
            continue
        at_a_table = any(table_contract(t) == contract_of(r['walk'], board['dealer']) for t in board['tables'])
        todo.append((key in failed, at_a_table, board['date'], r))
    # boards never tried before those that failed once; contracts nobody played first, then the most recent events
    todo.sort(key=lambda item: (item[0], item[1], -int(item[2].replace('-', '')[:8])))
    if copied:
        with out_path.open('a', encoding='utf-8') as out:
            for r in copied:
                out.write(json.dumps(r) + '\n')
    print(f'{len(replay)} boards replayed, {len(done) - len(copied)} already played, {len(copied)} copied from another '
          f"model's autoplay of the same auction, {len(todo)} to send to {BASE}, "
          f'{len(given_up)} left out after {GIVE_UP} failed attempts', flush=True)
    if not todo or '--no-requests' in sys.argv[4:]:
        return

    status = get(f'{BASE}/ready')
    if status.get('status') != 'ready':
        raise SystemExit(f'Brill service not ready: {status}')
    failures = slow_run = 0
    with out_path.open('a', encoding='utf-8') as out:
        for retry, _, _, r in tqdm(todo, mininterval=60, file=sys.stdout):
            board = field[(r['tournament_id'], r['board'])]
            row = {'tournament_id': r['tournament_id'], 'board': r['board'], 'auction': r['walk']}
            t0 = time.time()
            try:
                row.update(autoplay(board, r['walk']))
                failures = slow_run = 0
            except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, ValueError) as error:
                detail = error.read().decode('utf-8', 'replace')[:300] if isinstance(error, urllib.error.HTTPError) else ''
                row['error'] = f'{type(error).__name__}: {error} {detail}'.strip()
                # a 504 is a board the service plays for more than 180 s, not an outage. On a board that
                # already failed once it is expected and says nothing about the service: the retries
                # come last, so a run of them would otherwise stop every run that has some.
                if isinstance(error, urllib.error.HTTPError) and error.code == 504:
                    slow_run += 0 if retry else 1
                else:
                    failures += 1
            row['seconds'] = round(time.time() - t0, 1)
            out.write(json.dumps(row) + '\n')
            out.flush()
            if failures >= 3 or slow_run >= 6:
                raise SystemExit(f'{max(failures, slow_run)} failures in a row, last: {row["error"]}')
            if 'error' in row:
                time.sleep(COOL_DOWN)
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}', flush=True)


if __name__ == '__main__':
    main()
