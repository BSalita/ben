"""Board-by-board table of our model in the Brill daily events.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_board_table.py FIELD.jsonl REPLAY.jsonl AUTOPLAY.jsonl SD.json OUT_DIR
        [--view as-played|real|replay] [--model KEY] [--model-name NAME] [--prod-models CSV] [--dd DD.json]

In a replay our model bids South (brill_replay.py). The result of its contract is Brill's
own card play: the Brill entrant's table when it had our auction or played our
contract, else the Brill service's /autoplay of our auction (brill_autoplay.py).
A board whose autoplay is not in AUTOPLAY.jsonl yet gets the double dummy result
and result_source 'dd-pending'.

Since 2026-11-19 the model is an entrant of the dailies itself (R:MortyAI). Views:
  as-played  (default) in an event it played, its own table is the row (result_source
             'played', robots 'live') and the replay is not used; every other event is the replay
  real       only the events it played, its own tables
  replay     the replay on every event, also those it played; an event with a board
             the replay lacks is left out
Its real table is never also counted as one of the field. Each row names the model
behind it: for a real table the prod setting of the time it was played (PROD_MODELS,
data/brill-replay/prod-models.csv: since_utc,key,label), for a replay --model and
--model-name. OUT_DIR/meta.json describes the view for the page.

DD_EV and Par_EV are the single-dummy expected values of mlBridgeAugmentLib
(EV_Score and EV_Max), from the trick probabilities of brill_sd_probs.py:
  dd_ev    the expected North-South score of our contract, over random deals of
           the two hands that are not declarer's side (the DD column's counterpart)
  par_ev   the dd_ev of the par contract; with several par contracts, the highest
           of their expected scores (North-South's view)

Standings: every entrant's boards are scored again with our table (the replay, or
the event's own table once the model had entered) in the field. standings.csv has every event's ranking of the entrants who played all its
boards (our model, named Morty AI, is player_id OURS); lifetime.csv has each entrant's average
board over its events of one scoring, ranked among those with 20 or more events.
"""
import argparse
import bisect
import csv
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
from brill_autoplay import ENTRANT, brill_source, entered_events, is_daily  # noqa: E402
from score_brill_replay import contract_of, imps, score_ns, table_contract  # noqa: E402

SEATS = 'NESW'
STRAINS = 'CDHSN'
US = 'OURS'
OUR_NAME = 'Morty AI'   # the name of our entrant in the standings and on the page
LIFETIME_MIN_EVENTS = 20   # events of one scoring an entrant must have completed to be ranked in lifetime.csv
VIEWS = ('as-played', 'real', 'replay')


class ProdModels:
    """Which of our models prod served at a time: rows (since_utc, key, label) of prod-models.csv."""

    def __init__(self, path):
        self.rows = []
        if path and pathlib.Path(path).exists():
            with open(path, encoding='utf-8', newline='') as f:
                self.rows = sorted((r['since_utc'], r['key'], r['label']) for r in csv.DictReader(f))
        self.since = [r[0] for r in self.rows]

    def at(self, played_at):
        """(key, label) of the setting live at played_at ('2026-11-21T17:19:51.9+00:00'), or 'unknown'."""
        if not played_at:
            return 'unknown', 'not known (no play time)'
        stamp = played_at[:19] + 'Z'    # the CSV writes 2026-11-21T16:44:00Z; Brill +00:00
        k = bisect.bisect_right(self.since, stamp) - 1
        return (self.rows[k][1], self.rows[k][2]) if k >= 0 else ('unknown', 'before the first known prod setting')


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
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('field', type=pathlib.Path)
    parser.add_argument('replay', help="the model's replay (brill_replay.py); '-' for none (view real)")
    parser.add_argument('autoplay', help="Brill's autoplay of the replay's contracts; '-' for none")
    parser.add_argument('sd', type=pathlib.Path)
    parser.add_argument('out_dir', type=pathlib.Path)
    parser.add_argument('--view', choices=VIEWS, default='as-played')
    parser.add_argument('--model', default='', help='key of the replayed model (runs/KEY.jsonl)')
    parser.add_argument('--model-name', default='', help='its name on the page')
    parser.add_argument('--prod-models', type=pathlib.Path, default=None,
                        help='prod-models.csv (default: next to FIELD)')
    parser.add_argument('--dd', type=pathlib.Path, default=None, help='double-dummy cache (default OUT_DIR/dd.json)')
    a = parser.parse_args()
    build(a.field, None if a.replay == '-' else pathlib.Path(a.replay), None if a.autoplay == '-' else pathlib.Path(a.autoplay),
          a.sd, a.out_dir, a.view, a.model, a.model_name,
          a.prod_models or a.field.with_name('prod-models.csv'), a.dd)


def build(field_path, replay_path, autoplay_path, sd_path, out_dir, view='as-played', model_key='', model_name='',
          prod_models_path=None, dd_path=None):
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}  view {view}  model {model_key or "-"}')
    out_dir.mkdir(parents=True, exist_ok=True)
    every = [json.loads(line) for line in field_path.open(encoding='utf-8')]
    field = [b for b in every if is_daily(b)]
    left_out = sorted({b['event'] for b in every if not is_daily(b)})
    print(f'{len(field)} boards of daily events; left out as not a daily: {len(every) - len(field)} boards of {left_out}')
    replay = {}
    if replay_path is not None:
        replay = {(r['tournament_id'], r['board']): r for r in map(json.loads, replay_path.open(encoding='utf-8'))}
    played = {}
    if autoplay_path is not None and autoplay_path.exists():
        for line in autoplay_path.open(encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                played[(r['tournament_id'], r['board'])] = r
    prod = ProdModels(prod_models_path)

    # Since 2026-11-19 our model is an entrant itself (ENTRANT). In an event it played, its own
    # table is our result in the views as-played and real; in every other event the replay is.
    # Its table never also counts as one of the field, so each event and the lifetime table hold one row for it.
    entered = entered_events(field)
    print(f'{len(entered)} events played by {ENTRANT} itself')
    no_replay = {b['tournament_id'] for b in field
                 if 'error' in replay.get((b['tournament_id'], b['board']), {'error': 'missing'})}
    dropped = set()
    if view == 'real':
        dropped = {b['tournament_id'] for b in field} - entered
    elif view == 'replay':
        dropped = no_replay
        if dropped:
            print(f'{len(dropped)} events left out: the replay lacks some of their boards')
    field = [b for b in field if b['tournament_id'] not in dropped]
    if not field:
        raise SystemExit(f'view {view}: no event to show')
    dd = dd_and_par(field, dd_path or out_dir / 'dd.json')
    sd_all = json.loads(sd_path.read_text(encoding='utf-8'))['deals']

    rows, standings = [], []
    for b in field:
        key = (b['tournament_id'], b['board'])
        own = None
        if view != 'replay' and b['tournament_id'] in entered:
            own = next(t for t in b['tables'] if t['player_id'] == ENTRANT)
        b = dict(b, tables=[t for t in b['tables'] if t['player_id'] != ENTRANT])
        r = replay.get(key)
        if own is None and (r is None or 'error' in r):
            raise SystemExit(f'board {key} has no replay')
        if own is None:
            row_model, played_at = (model_key, model_name), None
        else:
            played_at = own.get('played_at')
            row_model = prod.at(played_at)
        info = dd[f"{b['deal']}|{b['dealer']}|{b['vul']}"]
        walk = r['walk'] if own is None else own['auction']
        contract = contract_of(walk, b['dealer'])
        if own is None:
            source, table = brill_source(b, walk)
        else:
            if table_contract(own) != contract:
                raise SystemExit(f'board {key}: the contract of our table does not follow from its auction')
            source, table = ('passout' if contract is None else 'played'), own
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
                # an autoplay of another auction is of an earlier replay of the board (the
                # setting changed since): the board waits for the autoplay of the new one
                if key in played and played[key]['auction'] == walk:
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
            'result_source': source,
            'robots': 'live' if own is not None else 'recorded' if r['status'] == 'record' else 'simulated',
            'tables': len(scores), 'field_median': int(np.median(scores)), 'field_common': common,
            'field_common_share': round(contracts.count(common) / len(contracts) * 100),
            'brill_contract': None if brill is None else (brill['contract'] + ' ' + brill['declarer'] if brill['declarer'] else 'Pass'),
            'brill_score': None if brill is None else brill['ns_score'],
            'model_key': row_model[0], 'model': row_model[1], 'played_at': played_at,
        })
        # every entrant's result on this board with our table (the event's own, or the replay) in the field
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
    # a view of few events (real) ranks those with half its events of the scoring, at most 20
    per_scoring = event_info.group_by('scoring').len()
    lifetime_min = max(2, min(LIFETIME_MIN_EVENTS, int(per_scoring['len'].min()) // 2))
    qualifies = pl.col('events') >= lifetime_min
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
    models = (boards.group_by('model_key', 'model').agg(pl.len().alias('boards'), pl.col('tournament_id').n_unique().alias('events'),
                                                         pl.col('date').min().alias('first'), pl.col('date').max().alias('last'),
                                                         pl.col('played_at').min().alias('first_played'))
              .sort('first', 'first_played', 'model_key', nulls_last=False).drop('first_played'))
    (out_dir / 'meta.json').write_text(json.dumps({
        'view': view, 'model_key': model_key, 'model_name': model_name, 'lifetime_min': lifetime_min,
        'events_left_out': len(dropped) if view == 'replay' else 0,
        'entered_events': len(entered & {b['tournament_id'] for b in field}),
        'models': models.to_dicts()}, indent=1), encoding='utf-8')
    counts = boards.group_by('result_source').len().sort('len', descending=True)
    print(counts)
    with pl.Config(tbl_width_chars=200, fmt_str_lengths=90):
        print(models)
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {boards.height} boards, {events.height} events -> {out_dir}')


if __name__ == '__main__':
    main()
