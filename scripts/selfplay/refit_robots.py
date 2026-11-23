#!/usr/bin/env python3
"""Refit the network copy of Brill's robots on the robots' own answers.

The self-play loop plays North, East and West with a network (config/Brill-21GF.conf) that makes
the real robots' call on about 96% of turns. Every answer of Brill's /bid service that the farm
has seen is in brill_live.py's cache file. This turns them into training rows, so that
selfplay_train.py can fine-tune the copy towards the real robots.

    python refit_robots.py cache --conf config/Brill-21GF.conf --out refit CACHE.jsonl [...]
    python refit_robots.py distill --conf config/Brill-21GF.conf --out distill.npz --limit 100000 GAMES_DIR
    python refit_robots.py agree --conf CONF --heldout refit.heldout.jsonl

cache    one row an answer: the robot's hand and the auction before its call, the call as
         target (weight 1), encoded as the engine encodes them (selfplay_build.encode). The
         answers of every 10th hand (by a hash of hand, dealer, vulnerability and scoring) go to
         OUT.heldout.jsonl instead, for `agree`. Writes OUT.npz (selfplay_train.py --sp form).
distill  the anchor: the copy's own calls at the North, East and West turns of explore games
         (where the copy played the robots), weight 1 at each turn, in the --base form with a
         weight array (selfplay_train.py takes it). So the copy keeps its calls where no answer
         says otherwise.
agree    how often the network of CONF makes the robot's call on the held-out answers (its
         legal first choice, as the self-play table plays it).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp_common  # noqa: E402
from sp_common import N_ROUNDS, SEATS, to_ben  # noqa: E402

FROM_CTX = {'--': 'PASS', 'Db': 'X', 'Rd': 'XX'}


def parse(row):
    """A cache row -> (seat index, hand, dealer, vul, scoring, calls before (engine form), target call)."""
    seat, hand, dealer, vul, scoring, ctx = row['key'].split('|')
    calls = [FROM_CTX.get(ctx[i:i + 2], ctx[i:i + 2]) for i in range(0, len(ctx), 2)]
    return SEATS.index(seat), hand, dealer, vul, scoring, calls, to_ben(row['bid'])


def held_out(hand, dealer, vul, scoring):
    return int(hashlib.md5(f'{hand}|{dealer}|{vul}|{scoring}'.encode()).hexdigest(), 16) % 10 == 0


def step_of(seat, dealer_i, i):
    return (i - (seat if seat >= dealer_i else seat + 4)) // 4


def build_cache(args):
    import binary  # noqa: F401  (selfplay_build.encode imports it)
    from bidding import bidding
    import selfplay_build
    spec = sp_common.NetSpec(sp_common.resolve_conf(args.conf))
    rows, seen, stats = [], set(), Counter()
    heldout = open(args.out + '.heldout.jsonl', 'w', encoding='utf-8')
    statics, bids, cls, wgt = [], [], [], []
    for path in args.cache:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r['key'] in seen:
                stats['repeated answers'] += 1
                continue
            seen.add(r['key'])
            seat, hand, dealer, vul, scoring, calls, target = parse(r)
            if target not in bidding.BID2ID:
                stats['answers not a call'] += 1
                continue
            if held_out(hand, dealer, vul, scoring):
                heldout.write(line if line.endswith('\n') else line + '\n')
                stats['held out'] += 1
                continue
            dealer_i = SEATS.index(dealer)
            auction = ['PAD_START'] * dealer_i + calls
            step = step_of(seat, dealer_i, len(auction))
            if step >= N_ROUNDS:
                stats['turns past round 8, left out'] += 1
                continue
            hands = ['', '', '', '']
            hands[seat] = hand
            static, bid_ids = selfplay_build.encode(binary, spec, auction, seat, {'hands': hands, 'vul': vul})
            c = np.zeros(N_ROUNDS, np.uint8)
            w = np.zeros(N_ROUNDS, np.uint8)
            c[step], w[step] = bidding.BID2ID[target], 1
            statics.append(static)
            bids.append(bid_ids)
            cls.append(c)
            wgt.append(w)
    heldout.close()
    stats['rows'] = len(statics)
    np.savez(args.out + '.npz', static=np.stack(statics), bids=np.stack(bids), cls=np.stack(cls), wgt=np.stack(wgt))
    meta = {'conf': spec.conf_path, 'cache': args.cache, **stats}
    with open(args.out + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(meta)


def build_distill(args):
    import binary  # noqa: F401
    from bidding import bidding
    import selfplay_build
    spec = sp_common.NetSpec(sp_common.resolve_conf(args.conf))
    statics, bids, cls, wgt = [], [], [], []
    n = 0
    for path in selfplay_build.games_files(args.games):
        for line in open(path, encoding='utf-8'):
            g = json.loads(line)
            if 'error' in g:
                continue
            n += 1
            if args.limit and n > args.limit:
                break
            dealer_i = SEATS.index(g['dealer'])
            auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in g['calls']]
            for seat in (0, 1, 3):
                c = np.zeros(N_ROUNDS, np.uint8)
                w = np.zeros(N_ROUNDS, np.uint8)
                for i in range(dealer_i, len(auction)):
                    if i % 4 == seat:
                        step = step_of(seat, dealer_i, i)
                        if step < N_ROUNDS:
                            c[step], w[step] = bidding.BID2ID[auction[i]], 1
                if not w.any():
                    continue
                static, bid_ids = selfplay_build.encode(binary, spec, auction, seat, g)
                statics.append(static)
                bids.append(bid_ids)
                cls.append(c)
                wgt.append(w)
    np.savez(args.out, static=np.stack(statics), bids=np.stack(bids), cls=np.stack(cls), wgt=np.stack(wgt))
    print(f'{min(n, args.limit or n)} games, {len(statics)} rows -> {args.out}')


def agree(args):
    from bidding import bidding
    net = sp_common.Net(sp_common.resolve_conf(args.conf))
    stats = Counter()
    by_kind = Counter()
    for line in open(args.heldout, encoding='utf-8'):
        seat, hand, dealer, vul, scoring, calls, target = parse(json.loads(line))
        if target not in bidding.BID2ID:
            continue
        auction = ['PAD_START'] * SEATS.index(dealer) + calls
        got = sp_common.ranked_legal(net.probs(net.hand(hand), sp_common.VULN[vul], auction), auction, 1)[0][0]
        kind = 'pass' if target == 'PASS' else 'double' if target in ('X', 'XX') else 'bid'
        stats['answers'] += 1
        stats['same'] += got == target
        by_kind[kind, 'n'] += 1
        by_kind[kind, 'same'] += got == target
    print(f"{args.conf}: same call on {stats['same']} of {stats['answers']} held-out answers "
          f"({100 * stats['same'] / max(stats['answers'], 1):.2f}%)")
    for kind in ('pass', 'double', 'bid'):
        if by_kind[kind, 'n']:
            print(f"  robot's call {kind:6}: {100 * by_kind[kind, 'same'] / by_kind[kind, 'n']:.2f}% of {by_kind[kind, 'n']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='what', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--conf', required=True)
    c.add_argument('--out', required=True, help='stem: OUT.npz, OUT.json, OUT.heldout.jsonl')
    c.add_argument('cache', nargs='+')
    d = sub.add_parser('distill')
    d.add_argument('--conf', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--limit', type=int, default=0)
    d.add_argument('games', nargs='+')
    a = sub.add_parser('agree')
    a.add_argument('--conf', required=True)
    a.add_argument('--heldout', required=True)
    args = parser.parse_args()
    os.chdir(sp_common.SRC)
    {'cache': build_cache, 'distill': build_distill, 'agree': agree}[args.what](args)


if __name__ == '__main__':
    main()
