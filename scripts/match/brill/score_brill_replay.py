"""Score Brill replays (brill_replay.py) against the real field, the way the Brill Elo app does.

Run on Windows with the Elo venv:

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/score_brill_replay.py FIELD.jsonl OUT_DIR name=replay.jsonl [name=replay.jsonl ...]

Each replayed config becomes one more entrant of every event. Its result on a
board comes from the real tables, in this order:
  auction   a table with exactly the same auction (BBOBen's, else BEN's, else the median of the others)
  contract  a table in the same contract by the same declarer (same preference)
  dd        no table played that contract: the double dummy result

Writes OUT_DIR/boards.parquet (one row per config and board) and prints the tables.
"""
import json
import pathlib
import statistics
import sys
from datetime import datetime

import numpy as np
import polars as pl

SRC = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC / 'elo'))
import elo_brill  # noqa: E402

SEATS = 'NESW'
BEN_FAMILY = ('R:BBOBen', 'R:BEN')
REFERENCE = ('R:BBOBen', 'R:BEN', 'R:Brill')
imps = elo_brill.imps  # the WBF IMP table, as Brill scores


def contract_of(calls, dealer):
    """(level, strain, 'X'|'XX'|'', declarer) of an auction, or None when passed out."""
    dealer_i = SEATS.index(dealer)
    last = None
    for i, call in enumerate(calls):
        if call[0].isdigit():
            last = i
    if last is None:
        return None
    level, strain = int(calls[last][0]), calls[last][1]
    dbl = ''
    for call in calls[last + 1:]:
        if call in ('X', 'XX'):
            dbl = call
    side = (dealer_i + last) % 2
    for i, call in enumerate(calls):
        if call[0].isdigit() and call[1] == strain and (dealer_i + i) % 2 == side:
            return level, strain, dbl, SEATS[(dealer_i + i) % 4]
    raise AssertionError(calls)


def score_ns(level, strain, dbl, declarer, tricks, vul):
    """North-South score of a played contract. vul is None / NS / EW / All."""
    vulnerable = vul == 'All' or (vul == 'NS') == (declarer in 'NS') and vul != 'None'
    over = tricks - level - 6
    if over < 0:
        down = -over
        if not dbl:
            points = -down * (100 if vulnerable else 50)
        else:
            if vulnerable:
                points = -(200 + 300 * (down - 1))
            else:
                points = -(100 + 200 * min(down - 1, 2) + 300 * max(down - 3, 0))
            if dbl == 'XX':
                points *= 2
    else:
        per = 20 if strain in 'CD' else 30
        below = per * level + (10 if strain == 'N' else 0)
        below *= {'': 1, 'X': 2, 'XX': 4}[dbl]
        points = below + ((500 if vulnerable else 300) if below >= 100 else 50)
        if level == 6:
            points += 750 if vulnerable else 500
        if level == 7:
            points += 1500 if vulnerable else 1000
        if dbl:
            points += 50 if dbl == 'X' else 100
            points += over * (100 if dbl == 'X' else 200) * (2 if vulnerable else 1)
        else:
            points += over * per
    return points if declarer in 'NS' else -points


def table_contract(table):
    text = (table['contract'] or '').replace('NT', 'N')
    if not table['declarer'] or text in ('PASS', ''):
        return None
    dbl = 'XX' if text.endswith('XX') else 'X' if text.endswith('X') else ''
    return int(text[0]), text[1], dbl, table['declarer']


def pick(tables):
    """The score that stands for ours: BBOBen's, else BEN's (the same card play
    engine), else the median of these tables."""
    for player in BEN_FAMILY:
        for t in tables:
            if t['player_id'] == player:
                return t['ns_score']
    return statistics.median_low(t['ns_score'] for t in tables)


def dd_tricks(deal, strain, declarer):
    from endplay.dds import calc_dd_table
    from endplay.types import Deal, Denom, Player
    table = calc_dd_table(Deal(deal))
    denom = {'S': Denom.spades, 'H': Denom.hearts, 'D': Denom.diamonds, 'C': Denom.clubs, 'N': Denom.nt}[strain]
    player = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}[declarer]
    return table[denom, player]


def main():
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    field_path, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    replays = dict(arg.split('=', 1) for arg in sys.argv[3:])
    field = [json.loads(line) for line in field_path.open(encoding='utf-8')]
    by_key = {(b['tournament_id'], b['board']): b for b in field}

    # self-test of the scoring rules against every real table
    wrong = 0
    for b in field:
        for t in b['tables']:
            c = table_contract(t)
            expect = 0 if c is None else score_ns(*c, int(t['tricks'] or 0), b['vul'])
            wrong += expect != t['ns_score']
            derived = contract_of(t['auction'], b['dealer'])
            if (derived or None) != c:
                wrong += 1
    tables_n = sum(len(b['tables']) for b in field)
    print(f'scoring self-test: {wrong} of {tables_n} real tables disagree with the site')

    rows = []
    for name, path in replays.items():
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'error' in r:
                print(f"{name}: board {r['board']} of {r['tournament_id']} failed and is left out: "
                      f"{r['error'].strip().splitlines()[-1]}")
                continue
            b = by_key[(r['tournament_id'], r['board'])]
            walk = r['walk']
            contract = contract_of(walk, b['dealer'])
            scores_field = np.array([t['ns_score'] for t in b['tables']])
            same_auction = [t for t in b['tables'] if t['auction'] == walk]
            same_contract = [t for t in b['tables'] if table_contract(t) == contract]
            if r['status'] == 'record' and not same_auction:
                raise AssertionError(f'{name}: an on-record auction that no table had: {r}')
            if same_auction:
                source, mine = 'auction', pick(same_auction)
            elif same_contract:
                source, mine = 'contract', pick(same_contract)
            elif contract is None:
                source, mine = 'dd', 0
            else:
                tricks = dd_tricks(b['deal'], contract[1], contract[3])
                source, mine = 'dd', score_ns(*contract, tricks, b['vul'])
            row = {
                'config': name, 'tournament_id': r['tournament_id'], 'board': r['board'],
                'event': b['event'], 'scoring': b['scoring'], 'date': b['date'],
                'status': r['status'], 'source': source,
                'auction': '-'.join(walk),
                'contract': 'Pass' if contract is None else f'{contract[0]}{contract[1]}{contract[2]} {contract[3]}',
                'score': int(mine),
            }
            # against every real table, and against the real tables without one reference entrant
            for label, drop in (('all', None), *((p, p) for p in REFERENCE)):
                others = np.array([t['ns_score'] for t in b['tables'] if t['player_id'] != drop])
                pct = (np.sum(others < mine) + 0.5 * np.sum(others == mine)) / len(others) * 100
                imp = imps(mine - others).mean()
                row[f'pct_{label}'], row[f'imp_{label}'] = float(pct), float(imp)
            # the reference entrants' own auctions and what this config calls on their way
            calls = r['south_calls']
            dealer_i = SEATS.index(b['dealer'])
            for player in REFERENCE:
                table = next((t for t in b['tables'] if t['player_id'] == player), None)
                tag = player[2:]
                if table is None:
                    row[f'{tag}_turns'] = row[f'{tag}_same'] = row[f'{tag}_auction_same'] = None
                    continue
                a = table['auction']
                turns = [i for i in range(len(a)) if (dealer_i + i) % 4 == 2]
                row[f'{tag}_turns'] = len(turns)
                row[f'{tag}_same'] = sum(calls.get('-'.join(a[:i])) == a[i] for i in turns)
                row[f'{tag}_auction_same'] = a == walk
            rows.append(row)
    boards = pl.DataFrame(rows, infer_schema_length=None)
    boards.write_parquet(out_dir / 'boards.parquet')
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {boards.height} rows -> {out_dir / "boards.parquet"}')


if __name__ == '__main__':
    main()
