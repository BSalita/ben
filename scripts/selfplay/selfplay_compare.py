#!/usr/bin/env python3
"""Compare two runs of selfplay_gen.py on the same deals: is learner A better than learner B?

    python selfplay_compare.py A_DIR B_DIR [--out compare.txt] [--name-a new --name-b incumbent]

Both runs must be net or search runs of the brill table (the learner South) on the same
deals. Each deal is a board of a match between the two: North-South score of A against
North-South score of B, in IMPs, and as won, tied or lost (the matchpoint view). Reports the
mean with its 95% interval over all deals, and by scoring and vulnerability.

The verdict line is what the generation driver reads:
  better              the interval of the IMPs a deal lies above 0
  worse               it lies below 0
  no difference shown otherwise
Exit code 0 in every case.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sp_common import imps  # noqa: E402


def load(path):
    path = os.path.join(path, 'games.jsonl') if os.path.isdir(path) else path
    games = {}
    for line in open(path, encoding='utf-8'):
        g = json.loads(line)
        if 'error' not in g:
            games[g['tournament_id'], g['board']] = g
    return games


def mean_ci(values):
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0), float('nan')
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, 1.96 * sd / math.sqrt(n)


def line(label, rows):
    if not rows:
        return f'{label:<22} no deals'
    mean, half = mean_ci([r[0] for r in rows])
    won = sum(1 for r in rows if r[0] > 0)
    lost = sum(1 for r in rows if r[0] < 0)
    mp, mp_half = mean_ci([50 + 50 * ((r[0] > 0) - (r[0] < 0)) for r in rows])
    differ = sum(1 for r in rows if r[1])
    return (f'{label:<22} {len(rows):>7} deals   {mean:+.4f} +-{half:.4f} IMPs a deal   '
            f'{mp:.2f}% +-{mp_half:.2f}   better {won}, worse {lost}   auction differs {100 * differ / len(rows):.1f}%')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--name-a', default='A')
    parser.add_argument('--name-b', default='B')
    parser.add_argument('--out')
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit('The two runs share no deal.')
    for k in keys:
        if a[k]['hands'] != b[k]['hands']:
            raise SystemExit(f'Deal {k} is not the same in the two runs.')
    rows = {k: (imps(a[k]['score'] - b[k]['score']), a[k]['calls'] != b[k]['calls']) for k in keys}
    text = [f'{args.name_a} against {args.name_b}, South against the same three robots on the same deals, '
            f'double dummy result of the contract reached',
            f'  A = {args.a}', f'  B = {args.b}',
            f'  {len(keys)} deals in both ({len(a)} in A, {len(b)} in B)', '',
            line('all deals', list(rows.values()))]
    for field, title in (('scoring', 'scoring'), ('vul', 'vulnerable')):
        for value in sorted({a[k][field] for k in keys}):
            text.append(line(f'{title} {value}', [rows[k] for k in keys if a[k][field] == value]))
    mean, half = mean_ci([r[0] for r in rows.values()])
    verdict = 'better' if mean - half > 0 else 'worse' if mean + half < 0 else 'no difference shown'
    text += ['', f'verdict: {verdict}']
    text = '\n'.join(text) + '\n'
    print(text, end='')
    if args.out:
        with open(args.out, 'w', newline='\n') as f:
            f.write(text)
        with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
            json.dump({'a': args.a, 'b': args.b, 'deals': len(keys), 'imps_a_deal': mean, 'half_interval': half,
                       'verdict': verdict}, f, indent=1)


if __name__ == '__main__':
    main()
