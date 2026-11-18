"""Board-by-board table of a replayed model in the Brill daily events.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_board_table.py FIELD.jsonl REPLAY.jsonl AUTOPLAY.jsonl SD.json OUT_DIR

Our model bids South (brill_replay.py). The result of its contract is Brill's
own card play: the Brill entrant's table when it had our auction or played our
contract, else the Brill service's /autoplay of our auction (brill_autoplay.py).
A board whose autoplay is not in AUTOPLAY.jsonl yet gets the double dummy result
and result_source 'dd-pending'.

DD_EV and Par_EV are the single-dummy expected values of mlBridgeAugmentLib
(EV_Score and EV_Max), from the trick probabilities of brill_sd_probs.py:
  dd_ev    the expected North-South score of our contract, over random deals of
           the two hands that are not declarer's side (the DD column's counterpart)
  par_ev   the dd_ev of the par contract; with several par contracts, the highest
           of their expected scores (North-South's view)

Standings: every entrant's boards are scored again with our table added to the
field. standings.csv has every event's ranking of the entrants who played all its
boards (our model, named Morty AI, is player_id OURS); lifetime.csv has each entrant's average
board over its events of one scoring, ranked among those with 20 or more events.
"""
import json
import pathlib
import re
import sys
from datetime import datetime

import numpy as np
import polars as pl
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from brill_autoplay import brill_source, is_daily  # noqa: E402
from score_brill_replay import contract_of, imps, score_ns, table_contract  # noqa: E402

SEATS = 'NESW'
STRAINS = 'CDHSN'
US = 'OURS'
OUR_NAME = 'Morty AI'   # the name of our entrant in the standings and on the page
LIFETIME_MIN_EVENTS = 20   # events of one scoring an entrant must have completed to be ranked in lifetime.csv


def dd_and_par(field, cache_path):
    """{deal: {'tricks': {strain+seat: n}, 'par_score': NS points, 'par': 'contracts'}} for every board."""
    from endplay.dds import calc_all_tables, par
    from endplay.types import Deal, Denom, Player, Vul
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    denoms = {'C': Denom.clubs, 'D': Denom.diamonds, 'H': Denom.hearts, 'S': Denom.spades, 'N': Denom.nt}
    players = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}
    vuls = {'None': Vul.none, 'NS': Vul.ns, 'EW': Vul.ew, 'All': Vul.both}
    todo = [b for b in field if f"{b['deal']}|{b['dealer']}|{b['vul']}" not in cache]
    for start in tqdm(range(0, len(todo), 32), desc='double dummy', file=sys.stdout):
        chunk = todo[start:start + 32]
        tables = list(calc_all_tables([Deal(b['deal']) for b in chunk]))
        for b, table in zip(chunk, tables):
            result = par(table, vuls[b['vul']], players[b['dealer']])
            contracts = []
            for c in result:
                text = str(c)
                if text not in contracts:
                    contracts.append(text)
            cache[f"{b['deal']}|{b['dealer']}|{b['vul']}"] = {
                'tricks': {s + p: int(table[denoms[s], players[p]]) for s in STRAINS for p in SEATS},
                'par_score': int(result.score), 'par': ' / '.join(contracts) if contracts else 'Pass'}
    if todo:
        cache_path.write_text(json.dumps(cache), encoding='utf-8')
    return cache


def expected_score(probs, level, strain, dbl, declarer, vul):
    return sum(p * score_ns(level, strain, dbl, declarer, tricks, vul) for tricks, p in enumerate(probs) if p)


PAR_CONTRACT = re.compile(r'^(\d)(NT|[♠♥♦♣])([NESW])(x{0,2})(=|[+-]\d+)$')
PAR_STRAIN = {'NT': 'N', '♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}


def par_ev(par, sd, vul):
    """(expected North-South score, contract) of the par contract. With several par
    contracts, the one with the highest expected score. The solver writes a par contract
    at the lowest level that earns the par score (1S+2, not 3S=); that contract is used."""
    if par == 'Pass':
        return 0.0, 'Pass'
    best = None
    for part in par.split(' / '):
        m = PAR_CONTRACT.match(part)
        if not m:
            raise ValueError(f'par contract not understood: {part}')
        level, strain, declarer, dbl = int(m[1]), PAR_STRAIN[m[2]], m[3], m[4].upper()
        ev = expected_score(sd[declarer + strain], level, strain, dbl, declarer, vul)
        if best is None or ev > best[0]:
            best = (ev, f'{level}{strain}{dbl} {declarer}')
    return best


def result_text(level, tricks):
    over = tricks - level - 6
    return '=' if over == 0 else f'{over:+d}'


def cross(mine, others, scoring):
    """Matchpoint percentage and average IMPs of one score against the other tables."""
    others = np.asarray(others)
    pct = (np.sum(others < mine) + 0.5 * np.sum(others == mine)) / len(others) * 100
    return float(pct), float(imps(mine - others).mean())


def main():
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    field_path, replay_path, autoplay_path, sd_path, out_dir = (pathlib.Path(p) for p in sys.argv[1:6])
    out_dir.mkdir(parents=True, exist_ok=True)
    every = [json.loads(line) for line in field_path.open(encoding='utf-8')]
    field = [b for b in every if is_daily(b)]
    left_out = sorted({b['event'] for b in every if not is_daily(b)})
    print(f'{len(field)} boards of daily events; left out as not a daily: {len(every) - len(field)} boards of {left_out}')
    replay ={(r['tournament_id'], r['board']): r for r in map(json.loads, replay_path.open(encoding='utf-8'))}
    played = {}
    if autoplay_path.exists():
        for line in autoplay_path.open(encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                played[(r['tournament_id'], r['board'])] = r
    dd = dd_and_par(field, out_dir / 'dd.json')
    sd_all = json.loads(sd_path.read_text(encoding='utf-8'))['deals']

    rows, standings = [], []
    for b in field:
        key = (b['tournament_id'], b['board'])
        r = replay.get(key)
        if r is None or 'error' in r:
            raise SystemExit(f'board {key} has no replay')
        info = dd[f"{b['deal']}|{b['dealer']}|{b['vul']}"]
        walk = r['walk']
        contract = contract_of(walk, b['dealer'])
        source, table = brill_source(b, walk)
        if b['deal'] not in sd_all:
            raise SystemExit(f'board {key}: no single-dummy probabilities in {sd_path}')
        sd = sd_all[b['deal']]
        par_ev_score, par_ev_contract = par_ev(info['par'], sd, b['vul'])
        dd_tricks = dd_score = None
        if contract is None:
            tricks, score, dd_score, dd_ev = None, 0, 0, 0.0
        else:
            dd_ev = expected_score(sd[contract[3] + contract[1]], *contract, b['vul'])
            dd_tricks = info['tricks'][contract[1] + contract[3]]
            dd_score = score_ns(*contract, dd_tricks, b['vul'])
            if source == 'autoplay':
                if key in played:
                    if played[key]['auction'] != walk:
                        raise SystemExit(f'board {key}: the autoplay is of another auction')
                    tricks, score = played[key]['tricks'], played[key]['ns_score']
                else:
                    source, tricks, score = 'dd-pending', dd_tricks, dd_score
            else:
                tricks, score = int(table['tricks']), table['ns_score']
        scores = [t['ns_score'] for t in b['tables']]
        pct, imp = cross(score, scores, b['scoring'])
        brill = next((t for t in b['tables'] if t['player_id'] == 'R:Brill'), None)
        contracts = [t['contract'] + ' ' + (t['declarer'] or '') if t['declarer'] else 'Pass' for t in b['tables']]
        common = max(sorted(set(contracts)), key=contracts.count)  # a tie goes to the first in order
        rows.append({
            'date': b['date'][:10], 'event': b['event'], 'tournament_id': b['tournament_id'], 'scoring': b['scoring'],
            'board': b['board'], 'dealer': b['dealer'], 'vul': b['vul'], 'deal': b['deal'],
            'auction': ' '.join(walk),
            'contract': 'Pass' if contract is None else f'{contract[0]}{contract[1]}{contract[2]}',
            'declarer': None if contract is None else contract[3],
            'tricks': tricks, 'result': None if contract is None else result_text(contract[0], tricks),
            'score': int(score), 'pct': round(pct, 1), 'imp': round(imp, 2),
            'event_measure': round(pct, 1) if b['scoring'] == 'MP' else round(imp, 2),
            'dd_tricks': dd_tricks, 'dd_result': None if contract is None else result_text(contract[0], dd_tricks),
            'dd_score': int(dd_score), 'par_score': info['par_score'], 'par': info['par'],
            'dd_ev': round(dd_ev), 'par_ev': round(par_ev_score), 'par_ev_contract': par_ev_contract,
            'imp_dd_vs_par': int(imps(dd_score - info['par_score'])),
            'imp_ev_vs_par_ev': int(imps(round(dd_ev) - round(par_ev_score))),
            'imp_play_vs_dd': int(imps(score - dd_score)),
            'result_source': source, 'robots': 'recorded' if r['status'] == 'record' else 'simulated',
            'tables': len(scores), 'field_median': int(np.median(scores)), 'field_common': common,
            'field_common_share': round(contracts.count(common) / len(contracts) * 100),
            'brill_contract': None if brill is None else (brill['contract'] + ' ' + brill['declarer'] if brill['declarer'] else 'Pass'),
            'brill_score': None if brill is None else brill['ns_score'],
        })
        # every entrant's result on this board with our table added to the field
        entries = [(t['player_id'], t['name'], t['ns_score']) for t in b['tables']] + [(US, OUR_NAME, score)]
        for i, (player, name, mine) in enumerate(entries):
            others = [s for j, (_, _, s) in enumerate(entries) if j != i]
            p, m = cross(mine, others, b['scoring'])
            standings.append({'tournament_id': b['tournament_id'], 'board': b['board'], 'player_id': player,
                              'name': name, 'measure': p if b['scoring'] == 'MP' else m})

    boards = pl.DataFrame(rows, infer_schema_length=None)
    per_event_boards = boards.group_by('tournament_id').agg(pl.len().alias('boards'))
    totals = (pl.DataFrame(standings).group_by('tournament_id', 'player_id')
              .agg(pl.col('name').first(), pl.col('measure').mean(), pl.len().alias('played'))
              .join(per_event_boards, on='tournament_id')
              .filter(pl.col('played') == pl.col('boards'))
              .with_columns(pl.col('measure').rank('min', descending=True).over('tournament_id').cast(pl.Int32).alias('rank'),
                            pl.len().over('tournament_id').alias('entrants')))
    # every event's standings, and the lifetime standings of each scoring
    event_info = boards.group_by('tournament_id').agg(pl.col('date').first(), pl.col('event').first(), pl.col('scoring').first())
    full = totals.join(event_info, on='tournament_id').sort('date', 'event', 'rank', 'name')
    (full.select('date', 'event', 'tournament_id', 'scoring', 'rank', 'entrants', 'player_id', 'name',
                 pl.col('measure').round(2), 'boards').write_csv(out_dir / 'standings.csv'))
    qualifies = pl.col('events') >= LIFETIME_MIN_EVENTS
    lifetime = (full.group_by('scoring', 'player_id')
                .agg(pl.col('name').last(), pl.len().alias('events'), pl.col('boards').sum(),
                     (pl.col('measure') * pl.col('boards')).sum().alias('total'),
                     (pl.col('rank') == 1).sum().alias('wins'), (pl.col('rank') <= 3).sum().alias('top3'),
                     pl.col('rank').mean().round(1).alias('mean_rank'))
                .with_columns((pl.col('total') / pl.col('boards')).alias('measure')).drop('total')
                .with_columns(pl.when(qualifies).then(pl.col('measure')).rank('min', descending=True).over('scoring')
                              .cast(pl.Int32).alias('rank'), qualifies.sum().over('scoring').alias('ranked'))
                .with_columns(pl.col('measure').round(2))
                .sort('scoring', 'rank', 'measure', descending=[False, False, True], nulls_last=True))
    lifetime.select('scoring', 'rank', 'ranked', 'player_id', 'name', 'measure', 'events', 'boards', 'wins', 'top3',
                    'mean_rank').write_csv(out_dir / 'lifetime.csv')
    winners = totals.filter(pl.col('rank') == 1).group_by('tournament_id').agg(
        pl.col('name').first().alias('winner'), pl.col('measure').first().alias('winner_measure'))
    events = (boards.group_by('tournament_id', maintain_order=True)
              .agg(pl.col('date').first(), pl.col('event').first(), pl.col('scoring').first(), pl.len().alias('boards'),
                   pl.col('pct').mean().round(1), pl.col('imp').mean().round(2),
                   pl.col('imp_dd_vs_par').mean().round(2), pl.col('imp_ev_vs_par_ev').mean().round(2),
                   (pl.col('result_source') == 'dd-pending').sum().alias('pending'))
              .join(totals.filter(pl.col('player_id') == US).select('tournament_id', 'rank', 'entrants', 'measure'), on='tournament_id')
              .join(winners, on='tournament_id')
              .with_columns(pl.col('measure').round(2), pl.col('winner_measure').round(2))
              .sort('date', 'event'))
    boards = boards.sort('date', 'event', 'board')
    boards.write_parquet(out_dir / 'boards.parquet')
    boards.drop('deal').write_csv(out_dir / 'boards.csv')
    events.write_csv(out_dir / 'events.csv')
    (out_dir / 'table.json').write_text(json.dumps({
        'made': f'{datetime.now():%Y-%m-%d %H:%M}',
        'events': events.to_dicts(), 'boards': boards.to_dicts()}), encoding='utf-8')
    counts = boards.group_by('result_source').len().sort('len', descending=True)
    print(counts)
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {boards.height} boards, {events.height} events -> {out_dir}')


if __name__ == '__main__':
    main()
