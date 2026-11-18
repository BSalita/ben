#!/usr/bin/env python3
"""Seat a BEN configuration South with Brill's own robots in the other three seats.

brill_replay.py takes the robots' calls from the record of past events and has
to simulate them once South leaves every recorded auction. Here North, East and
West are asked of Brill's public service (its /bid, the engine the events run
on today), so every auction is finished by the real robots and any deal can be
bid, not only past boards. The engine changes: on boards of earlier months it
no longer makes every call the robots made then.

    python brill_live.py --conf config/GIB-BBO.conf --field /var/lib/ben/brill/field.jsonl --out /var/lib/ben/brill/live/gibbbo --cache /var/lib/ben/brill/live/robot_calls.jsonl

--field is field.jsonl, or any file with one board a line: tournament_id,
board, deal (PBN), dealer, vul (None, NS, EW, All), scoring (MP, IMP).

Requests go one at a time over all workers and over every run that shares the
--cache file (a file lock beside it). Every answer is appended to --cache (the
call, and the meaning Brill gives it), so a second config, at the same time or
later, only asks about auctions the first did not reach.

Output: OUT/replay.jsonl in brill_replay.py's format (walk, status 'service',
south_calls for the South turns on the way), scored by score_brill_replay.py.
"""
import argparse
import datetime
import fcntl
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brill_replay  # noqa: E402  (puts scripts/match and src on sys.path)
from brill_replay import SEATS, VULN, W, ab_match, from_ben, to_ben  # noqa: E402

BASE = 'https://brillservice.aalborgdata.dk'
TO_CTX = {'P': '--', 'X': 'Db', 'XX': 'Rd'}
FROM_SERVICE = {'PASS': 'P', 'Pass': 'P', '--': 'P', 'Db': 'X', 'DBL': 'X', 'Rd': 'XX', 'RDBL': 'XX'}


class FileLock:
    """One holder at a time over every process that opens the same path."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, 'w')
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def worker_init(cfg):
    brill_replay.worker_init(cfg)
    W['lock'] = FileLock(cfg['cache'] + '.lock')
    W['cache'] = {}
    W['cache_size'] = 0
    read_cache()


def read_cache():
    """Take in what any worker (or an earlier run) appended to the cache file since the last look."""
    path = W['cfg']['cache']
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        f.seek(W['cache_size'])
        data = f.read()
    whole = data[:data.rfind(b'\n') + 1]  # a line another worker is still writing waits for the next look
    for line in whole.splitlines():
        r = json.loads(line)
        W['cache'][r['key']] = r['bid']
    W['cache_size'] += len(whole)


def robot_call(board, seat_i, hand, calls):
    """The Brill robot's call in this seat after these calls: from the cache, else from the service."""
    cfg = W['cfg']
    ctx = ''.join(TO_CTX.get(c, c) for c in calls)
    key = f"{SEATS[seat_i]}|{hand}|{board['dealer']}|{board['vul']}|{board['scoring']}|{ctx}"
    if key in W['cache']:
        return W['cache'][key], False
    with W['lock']:
        read_cache()
        if key in W['cache']:
            return W['cache'][key], False
        query = {'hand': hand, 'seat': SEATS[seat_i], 'dealer': board['dealer'], 'vul': board['vul'],
                 'ctx': ctx, 'board': board['board'], 'scoring': board['scoring']}
        url = f"{cfg['url']}/bid?{urllib.parse.urlencode(query)}"
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'brill-replay'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                if attempt == 3:
                    raise RuntimeError(f'Brill service: {error} for {key}')
                time.sleep(5 * (attempt + 1))
        if not data.get('bid'):
            raise RuntimeError(f'Brill service gave no call for {key}: {str(data)[:300]}')
        call = FROM_SERVICE.get(data['bid'], data['bid']).replace('NT', 'N')
        row = {'key': key, 'bid': call, 'means': data.get('means'), 'requires': data.get('requires'),
               'ruleSet': data.get('ruleSet')}
        with open(cfg['cache'], 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        W['cache'][key] = call
        if cfg['pause']:
            time.sleep(cfg['pause'])
    return call, True


def run_board(board):
    t0 = time.time()
    key = {'tournament_id': board['tournament_id'], 'board': board['board']}
    try:
        return {**key, **live(board), 'seconds': round(time.time() - t0, 1)}
    except Exception:
        return {**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 1)}


def live(board):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample

    cfg = W['cfg']
    hands = board['deal'][2:].split()
    first = SEATS.index(board['deal'][0])
    hands = [hands[(i - first) % 4] for i in range(4)]
    dealer_i = SEATS.index(board['dealer'])
    vuln = VULN[board['vul']]
    W['models'].matchpoint = board['scoring'] == 'MP' if cfg['matchpoint'] == 'event' else cfg['matchpoint'] == 'true'

    def ben_auction(calls):
        return ['PAD_START'] * dealer_i + [to_ben(c) for c in calls]

    walk, south_calls, asked = [], {}, 0
    while not bidding.auction_over(ben_auction(walk)):
        seat_i = (dealer_i + len(walk)) % 4
        if seat_i == 2:
            np.random.seed(42)
            bot = BotBid(vuln, hands[2], W['models'], Sample.from_conf(W['conf']), 2, dealer_i, W['dds'], False, False)
            call = from_ben(bot.bid(ben_auction(walk)).bid)
            south_calls['-'.join(walk)] = call
        else:
            call, new = robot_call(board, seat_i, hands[seat_i], walk)
            asked += new
        if not bidding.can_bid(to_ben(call), ben_auction(walk)):
            raise ValueError(f'illegal call {call} by {SEATS[seat_i]} after {walk}')
        walk.append(call)
    return {'walk': walk, 'status': 'service', 'off_at': None, 'south_calls': south_calls, 'requests': asked}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--conf', required=True, help='config that sits South (relative to src, or absolute)')
    parser.add_argument('--field', required=True, help='field.jsonl, or one board a line')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache', required=True, help="the robots' answers, shared by every config's run")
    parser.add_argument('--url', default=BASE)
    parser.add_argument('--pause', type=float, default=0.0, help='seconds to wait after each request')
    parser.add_argument('--matchpoint', default='event', choices=['event', 'true', 'false'])
    parser.add_argument('--limit', type=int, default=0, help='only the last N boards of the file')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dds-threads', type=int, default=4)
    parser.add_argument('--tf-threads', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    started = datetime.datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    boards = [json.loads(line) for line in open(args.field, encoding='utf-8')]
    if args.limit:
        boards = boards[-args.limit:]
    boards = [{k: b[k] for k in ('tournament_id', 'board', 'deal', 'dealer', 'vul', 'scoring')} for b in boards]
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
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

    cfg = {'conf': ab_match.resolve(args.conf), 'robots': None, 'out': args.out, 'matchpoint': args.matchpoint,
           'cache': os.path.abspath(args.cache), 'url': args.url.rstrip('/'), 'pause': args.pause,
           'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads}
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump({'argv': sys.argv, 'cfg': cfg, 'boards': len(boards), 'git': ab_match.git_head(),
                   'conf_sha1': ab_match.sha1_file(cfg['conf'])}, f, indent=1)
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{len(tasks)} of {len(boards)} boards to bid, {workers} workers, conf {cfg["conf"]}, robots {cfg["url"]}')
    sys.stdout.flush()

    counts = Counter()
    t0 = time.time()
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, r in enumerate(pool.imap_unordered(run_board, tasks, chunksize=1), start=1):
                out.write(json.dumps(r) + '\n')
                out.flush()
                counts['error' if 'error' in r else 'board'] += 1
                counts['requests'] += r.get('requests', 0)
                if k % 50 == 0 or k == len(tasks):
                    print(f'[{k}/{len(tasks)}] {time.time() - t0:6.0f} s  {dict(counts)}')
                    sys.stdout.flush()
    ended = datetime.datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {dict(counts)}')
    return 1 if counts['error'] else 0


if __name__ == '__main__':
    sys.exit(main())
