#!/usr/bin/env python3
"""Seat a BEN configuration South on the boards of past Brill events.

In a Brill event every entrant sits South with three Brill robots, and all
entrants play the same boards. The field file (fetch_brill_field.py) holds the
auction of every table, so for any auction that happened at some table the
Brill robots' real next call is on record.

For each board the config bids South's hand. North, East and West make the
call the real robots made after the same calls so far. That works until South
makes a call no entrant made at that point; from there the three robots are
played by --robots (a BEN config trained on Brill's bidding) and the board is
marked as simulated.

    python brill_replay.py --conf config/GIB-BBO.conf --field /var/lib/ben/brill/field.jsonl --out /var/lib/ben/brill/gibbbo

Output: OUT/replay.jsonl, one line per board:
  walk         the auction reached (P, X, XX, 1C..7N), from the dealer
  status       'record' (every robot call is a real one) or 'sim'
  off_at       number of calls made when the record was left (sim only)
  south_calls  with --all-nodes: the config's call at every South turn that
               occurred at any table, keyed by the calls so far joined with '-'
  robot_check  with --robot-check: how often --robots makes the real robots' call
Scoring against the field is done by score_brill_replay.py.
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, MATCH)
import ab_match  # noqa: E402  (puts src first on sys.path)

SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
W = {}


def to_ben(call):
    return 'PASS' if call == 'P' else call


def from_ben(call):
    return 'P' if call == 'PASS' else call


def worker_init(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.chdir(ab_match.SRC)
    log = open(os.path.join(cfg['out'], 'logs', f'worker-{os.getpid()}.log'), 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    from nn.models_tf2 import Models
    from ddsolver.ddsolver import DDSolver

    W['cfg'] = cfg
    W['conf'] = ab_match.load_conf(cfg['conf'], None)
    W['models'] = Models.from_conf(W['conf'], ab_match.BEN_ROOT)
    ab_match.platform_fixups(W['models'])
    W['robot_conf'] = W['robot_models'] = None
    if cfg['robots']:
        W['robot_conf'] = ab_match.load_conf(cfg['robots'], None)
        W['robot_models'] = Models.from_conf(W['robot_conf'], ab_match.BEN_ROOT)
        ab_match.platform_fixups(W['robot_models'])
    W['dds'] = DDSolver(max_threads=cfg['dds_threads'])
    print(f"worker {os.getpid()} ready: {cfg['conf']} robots={cfg['robots']}")


def run_board(board):
    t0 = time.time()
    key = {'tournament_id': board['tournament_id'], 'board': board['board']}
    try:
        return {**key, **replay(board), 'seconds': round(time.time() - t0, 1)}
    except Exception:
        return {**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 1)}


def replay(board):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample

    cfg = W['cfg']
    hands = board['deal'][2:].split()
    dealer_i = SEATS.index(board['dealer'])
    vuln = VULN[board['vul']]
    matchpoint = board['scoring'] == 'MP' if cfg['matchpoint'] == 'event' else cfg['matchpoint'] == 'true'
    W['models'].matchpoint = matchpoint
    if W['robot_models'] is not None:
        W['robot_models'].matchpoint = matchpoint

    def call_of(models, conf, seat_i, calls):
        np.random.seed(42)
        bot = BotBid(vuln, hands[seat_i], models, Sample.from_conf(conf), seat_i, dealer_i, W['dds'], False, False)
        auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in calls]
        call = bot.bid(auction).bid
        if not bidding.can_bid(call, auction):
            raise ValueError(f'illegal call {call} after {calls}')
        return from_ben(call)

    # what was called after each sequence of calls, over all tables
    record = defaultdict(Counter)
    for table in board['tables']:
        calls = table['auction']
        for i, call in enumerate(calls):
            record[tuple(calls[:i])][call] += 1

    def seat_after(calls):
        return (dealer_i + len(calls)) % 4

    def over(calls):
        return bidding.auction_over(['PAD_START'] * dealer_i + [to_ben(c) for c in calls])

    south_calls = {}

    def south(calls):
        k = '-'.join(calls)
        if k not in south_calls:
            hook = W.get('south_call')       # another South bidder (book_south_replay.py)
            south_calls[k] = hook(board, hands[2], calls) if hook else call_of(W['models'], W['conf'], 2, calls)
        return south_calls[k]

    walk, status, off_at = [], 'record', None
    while not over(walk):
        seat_i = seat_after(walk)
        if seat_i == 2:
            call = south(walk)
            if status == 'record' and call not in record[tuple(walk)]:
                if W['robot_models'] is None:
                    return {'walk': walk + [call], 'status': 'off', 'off_at': len(walk), 'south_calls': south_calls}
                status, off_at = 'sim', len(walk)
        elif status == 'record':
            call = record[tuple(walk)].most_common(1)[0][0]
        else:
            call = call_of(W['robot_models'], W['robot_conf'], seat_i, walk)
        walk.append(call)

    out = {'walk': walk, 'status': status, 'off_at': off_at}
    if cfg['all_nodes']:
        for calls in list(record):
            if seat_after(calls) == 2:
                south(list(calls))
    out['south_calls'] = south_calls

    if W['robot_models'] is not None and board['index'] < cfg['robot_check']:
        agree, total, misses = 0, 0, []
        for calls, counted in record.items():
            seat_i = seat_after(calls)
            if seat_i == 2:
                continue
            real = counted.most_common(1)[0][0]
            got = call_of(W['robot_models'], W['robot_conf'], seat_i, list(calls))
            total += 1
            agree += got == real
            if got != real:
                misses.append({'calls': '-'.join(calls), 'seat': SEATS[seat_i], 'real': real, 'got': got})
        out['robot_check'] = {'agree': agree, 'total': total, 'misses': misses}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--conf', required=True, help='config that sits South (relative to src, or absolute)')
    parser.add_argument('--field', required=True, help='field.jsonl from fetch_brill_field.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--robots', default='config/Brill-21GF.conf',
                        help="config that bids North, East and West once the record is left; 'none' stops there")
    parser.add_argument('--matchpoint', default='event', choices=['event', 'true', 'false'],
                        help="the config's matchpoint setting: by each event's scoring (default), or fixed")
    parser.add_argument('--all-nodes', action='store_true',
                        help="also ask for South's call at every South turn on record, not only those on the way")
    parser.add_argument('--robot-check', type=int, default=0,
                        help='on the first N boards, compare --robots with every real robot call')
    parser.add_argument('--limit', type=int, default=0, help='only the last N boards of the field file')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dds-threads', type=int, default=4)
    parser.add_argument('--tf-threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    started = datetime.datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    boards = [json.loads(line) for line in open(args.field, encoding='utf-8')]
    if args.limit:
        boards = boards[-args.limit:]
    for i, b in enumerate(boards):
        b['index'] = i
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    path = os.path.join(args.out, 'replay.jsonl')
    done = set()
    if args.resume and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['tournament_id'], r['board']))
    elif os.path.exists(path):
        raise SystemExit(f'{path} exists; use --resume or another --out')
    tasks = [b for b in boards if (b['tournament_id'], b['board']) not in done]

    cfg = {'conf': ab_match.resolve(args.conf),
           'robots': None if args.robots == 'none' else ab_match.resolve(args.robots),
           'out': args.out, 'matchpoint': args.matchpoint, 'all_nodes': args.all_nodes,
           'robot_check': args.robot_check, 'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads}
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump({'argv': sys.argv, 'cfg': cfg, 'boards': len(boards), 'git': ab_match.git_head(),
                   'conf_sha1': ab_match.sha1_file(cfg['conf'])}, f, indent=1)
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{len(tasks)} of {len(boards)} boards to bid, {workers} workers, conf {cfg["conf"]}')
    sys.stdout.flush()

    counts = Counter()
    t0 = time.time()
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, r in enumerate(pool.imap_unordered(run_board, tasks, chunksize=1), start=1):
                out.write(json.dumps(r) + '\n')
                out.flush()
                counts['error' if 'error' in r else r['status']] += 1
                if k % 50 == 0 or k == len(tasks):
                    print(f'[{k}/{len(tasks)}] {time.time() - t0:6.0f} s  {dict(counts)}')
                    sys.stdout.flush()
    ended = datetime.datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {dict(counts)}')
    return 1 if counts['error'] else 0


if __name__ == '__main__':
    sys.exit(main())
