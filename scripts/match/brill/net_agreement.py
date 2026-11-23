#!/usr/bin/env python3
"""How often a bidding network's first choice is the call that was really made on the Brill boards.

For every recorded turn of the Brill robots (North, East, West) and of the named
entrants (South), the network of --conf is asked for its call with no search:
the legal call with the highest output. Robot turns show how good the network
is as a model of partner and opponents in rollouts; South turns show how close
it is to an entrant.

    python net_agreement.py --conf A.conf --conf B.conf --field field.jsonl --out DIR [--workers 6]

Writes DIR/<conf name>.json and prints a table.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import ab_match  # noqa: E402

SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
ENTRANTS = ('R:BBOBen', 'R:DownTwo', 'R:Brill', 'R:BEN', 'R:PowerShark', 'R:WBridge5', 'R:Lia')
W = {}


def worker_init(conf, tf_threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(tf_threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.chdir(ab_match.SRC)
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 1)
    from nn.models_tf2 import Models
    W['conf'] = ab_match.load_conf(conf, None)
    W['models'] = Models.from_conf(W['conf'], ab_match.BEN_ROOT)
    ab_match.platform_fixups(W['models'])


def run_board(board):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample

    hands = board['deal'][2:].split()
    dealer_i = SEATS.index(board['dealer'])
    vuln = VULN[board['vul']]
    sampler = Sample.from_conf(W['conf'])
    nodes = defaultdict(Counter)       # prefix -> real calls
    who = defaultdict(dict)            # prefix -> entrant -> call
    for t in board['tables']:
        a = t['auction']
        for i, c in enumerate(a):
            nodes[tuple(a[:i])][c] += 1
            if t['player_id'] in ENTRANTS and (dealer_i + i) % 4 == 2:
                who[tuple(a[:i])][t['player_id']] = c
    out = Counter()
    for prefix, calls in nodes.items():
        seat_i = (dealer_i + len(prefix)) % 4
        auction = ['PAD_START'] * dealer_i + ['PASS' if c == 'P' else c for c in prefix]
        bot = BotBid(vuln, hands[seat_i], W['models'], sampler, seat_i, dealer_i, None, False, False)
        soft, _ = bot.next_bid_np(auction)
        soft = np.array(soft, dtype=np.float64)
        for k in np.argsort(-soft):
            call = bidding.ID2BID[k]
            if k >= 2 and bidding.can_bid(call, auction):
                break
        call = 'P' if call == 'PASS' else call
        if seat_i != 2:
            real = calls.most_common(1)[0][0]
            group = 'North' if seat_i == 0 else 'East/West'
            for tag in (group, group + (' pass' if real == 'P' else ' not pass'), 'robots'):
                out[tag, 'n'] += 1
                out[tag, 'same'] += call == real
            out['robots', 'p_real'] += float(soft[bidding.BID2ID['PASS' if real == 'P' else real]])
        else:
            for player, real in who[prefix].items():
                for tag in (player, player + (' pass' if real == 'P' else ' not pass')):
                    out[tag, 'n'] += 1
                    out[tag, 'same'] += call == real
    return dict(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--conf', action='append', required=True)
    parser.add_argument('--field', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--tf-threads', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    boards = [json.loads(line) for line in open(args.field, encoding='utf-8')]
    if args.limit:
        boards = boards[-args.limit:]
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for conf in args.conf:
        name = os.path.splitext(os.path.basename(conf))[0]
        path = os.path.join(args.out, name + '.json')
        if os.path.exists(path):
            results[name] = json.load(open(path))
            continue
        total = Counter()
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, initializer=worker_init, initargs=(ab_match.resolve(conf), args.tf_threads)) as pool:
            for r in pool.imap_unordered(run_board, boards, chunksize=8):
                for k, v in r.items():
                    total[k] += v
        res = {}
        for (tag, what), v in total.items():
            res.setdefault(tag, {})[what] = v
        results[name] = res
        json.dump(res, open(path, 'w'), indent=1)
        print(f'{name} done', file=sys.stderr, flush=True)
    tags = ['robots', 'North', 'North not pass', 'East/West', 'East/West not pass']
    tags += [t for e in ENTRANTS for t in (e, e + ' not pass')]
    print(f"{'same call %':28}" + ''.join(f'{n[:16]:>18}' for n in results))
    for tag in tags:
        row = f'{tag:28}'
        for res in results.values():
            r = res.get(tag)
            row += f"{100 * r['same'] / r['n']:18.2f}" if r else f"{'':18}"
        n = next((res[tag]['n'] for res in results.values() if tag in res), 0)
        print(row + f'   n={n}')
    print(f"{'mean output for real call':28}" + ''.join(
        f"{res['robots']['p_real'] / res['robots']['n']:18.3f}" for res in results.values()))


if __name__ == '__main__':
    main()
