#!/usr/bin/env python3
"""Real-play trick odds for BEN's bidding search, fitted on the Brill dailies.

For every field table: the contract, its double-dummy tricks on the real deal (dd.json) and the
tricks really taken. A cell is (side, kind, margin):
  side    'ours' when North-South declare (the entrant's side: the entrant is South, partner
          Brill's robot), 'theirs' when East-West declare (Brill's robots against the entrant)
  kind    'part', 'game' or 'slam' (grand slams counted as slams)
  margin  double-dummy tricks minus the tricks the contract needs, clipped to -3..+3
and holds P(real - double dummy = d) for d = -3..+3 (clipped). A cell with fewer than MIN
tables takes its (side, margin) row over all kinds.

  elo/.venv/Scripts/python.exe fit_trick_odds.py FIELD.jsonl DD.json OUT.json

BEN reads OUT.json with [sampling] real_play_tricks = OUT.json (botbidder.real_play_odds).
"""
import json
import re
import sys
from collections import defaultdict

GAME = {'C': 5, 'D': 5, 'H': 4, 'S': 4, 'N': 3}
D = list(range(-3, 4))
MIN = 150


def kind_of(level, strain):
    return 'slam' if level >= 6 else 'game' if level >= GAME[strain] else 'part'


def main(field_path, dd_path, out_path):
    dd = json.load(open(dd_path, encoding='utf-8'))
    counts = defaultdict(lambda: [0] * len(D))
    tables = 0
    for line in open(field_path, encoding='utf-8'):
        b = json.loads(line)
        if not b['event'].startswith('Daily'):
            continue
        t0 = dd.get(f"{b['deal']}|{b['dealer']}|{b['vul']}")
        if not t0:
            continue
        for t in b['tables']:
            m = re.fullmatch(r'([1-7])(NT|N|S|H|D|C)(X{0,2})', str(t['contract']))
            if not m:
                continue
            level, strain = int(m.group(1)), 'N' if m.group(2).startswith('N') else m.group(2)
            ddt = t0['tricks'][strain + t['declarer']]
            side = 'ours' if t['declarer'] in 'NS' else 'theirs'
            margin = max(-3, min(3, ddt - level - 6))
            d = max(-3, min(3, t['tricks'] - ddt))
            for key in (f'{side}|{kind_of(level, strain)}|{margin}', f'{side}|all|{margin}'):
                counts[key][D.index(d)] += 1
            tables += 1
    table = {}
    for side in ('ours', 'theirs'):
        for kind in ('part', 'game', 'slam'):
            for margin in range(-3, 4):
                c = counts.get(f'{side}|{kind}|{margin}')
                if c is None or sum(c) < MIN:
                    c = counts[f'{side}|all|{margin}']
                n = sum(c)
                table[f'{side}|{kind}|{margin}'] = [round(x / n, 4) for x in c] if n else [0, 0, 0, 1, 0, 0, 0]
    out = {'about': __doc__.split('\n')[0], 'd': D, 'tables': tables, 'min_cell': MIN,
           'n': {k: sum(v) for k, v in counts.items()}, 'odds': table}
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
    print(f'{tables} tables -> {out_path}')
    for k in ('ours|game|-1', 'ours|game|0', 'ours|part|0', 'theirs|game|0', 'theirs|game|-1'):
        print(k, table[k], out['n'].get(k))


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
