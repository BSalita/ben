#!/usr/bin/env python3
"""Training rows for the self-play fine-tuning (selfplay_train.py).

    python selfplay_build.py games --conf CONF --out sp.npz [--weight mix] DIR_OR_FILE ...
    python selfplay_build.py decisions --conf CONF --out rl.npz DIR_OR_FILE ...
    python selfplay_build.py base BIN_DIR base.npz

games: from the games.jsonl of explore runs (selfplay_gen.py). One row (with --target pair
several) for each learner hand that had a turn where the candidates' results differ. The row holds the hand and the auction
as it went, encoded exactly as the engine encodes them when it asks the network
(binary.get_auction_binary). At such a turn the target is the candidate with the best double
dummy result on that deal (the call made, when it is among the best), weighted by what was at
stake: the IMPs between the best result and the call made, or, when the call made was best,
between it and the next best. Every other turn of the row has weight 0 and teaches nothing.

Why the weight: one deal says little about a call, and a target of "best on this deal" alone
would teach the call that wins most often. Weighted by the IMPs at stake, a network trained
on many deals ends up preferring, between two calls, the one with the better expected result
(--weight imp; mp weighs every difference alike, as at matchpoints; mix lies between).

decisions: the same turns without a target chosen: one row for each learner turn that had
candidates, with every tried call, the network's output for it and its result. This is what
the value network (selfplay_value.py) and the policy-gradient fine-tuning (selfplay_train.py
--pg) learn from. Arrays: static and bids as below, step (the round of the turn), n (number
of tried calls), cand (rows, K) the calls, the call made first, score (rows, K) their double
dummy scores for the learner's side, p (rows, K) the network's outputs, deal (number of the
deal, for splitting by deal), mp (1 on a matchpoint board), n_deals.

base: the binary training set of the bidding trainer (x.npy, y.npy of bidding_binary_keras.py)
in the same compact form, weight 1 at every turn. It anchors the fine-tuning to the system.

Arrays: static (rows, features before the calls) float16, bids (rows, 8, 4) uint8 = my last
call, left hand's, partner's, right hand's at each of the 8 rounds, cls (rows, 8) the target
call, wgt (rows, 8) its weight.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp_common  # noqa: E402
from sp_common import N_BIDS, N_ROUNDS, SEATS, VULN, imps, to_ben  # noqa: E402

WEIGHTS = {
    'imp': lambda i: min(i, 12),
    'mp': lambda i: 4,
    'mix': lambda i: min(2 + (i + 1) // 2, 8),
}


def targets_of(decision, side, rule):
    """Targets of one turn: [(call, points at stake, source of the call when it is not the one made), ...].
    side: +1 for a North-South seat, -1 for East-West; candidate scores are North-South.

    best  one target: the candidate with the best result (the call made, when it is among the
          best), at stake the difference to the call made, or from it to the next best.
    pair  one target for every other candidate: the better of it and the call made, at stake
          the difference between the two. With many candidates on one deal some call is
          always lucky; 'best' then leans to the others, 'pair' to the call made.
    """
    cands = decision['cands']
    values = [side * c[3] for c in cands]
    if rule == 'pair':
        return [(cands[k][0], values[k] - values[0], cands[k][4] if len(cands[k]) > 4 else 'net') if values[k] > values[0]
                else (cands[0][0], values[0] - values[k], None) for k in range(1, len(cands)) if values[k] != values[0]]
    best = max(values)
    if values[0] == best:
        points = best - max(values[1:])
        return [(cands[0][0], points, None)] if points else []
    k = values.index(best)          # candidates are in falling network output: the likeliest of the best
    return [(cands[k][0], best - values[0], cands[k][4] if len(cands[k]) > 4 else 'net')]


def kind_of(call):
    return 'pass' if call == 'P' else 'double' if call in ('X', 'XX') else f'bid at the {call[0]} level'


def games_files(paths):
    for p in paths:
        yield os.path.join(p, 'games.jsonl') if os.path.isdir(p) else p


def encode(binary, spec, auction, seat, g):
    """The hand and the auction as the engine encodes them for the network: (static, calls of the 8 rounds)."""
    x = binary.get_auction_binary(N_ROUNDS, auction, seat, spec.hand(g['hands'][seat]), VULN[g['vul']], spec)[0]
    blocks = x[:, spec.n_static:].reshape(N_ROUNDS, 4, N_BIDS)
    if not ((x[:, :spec.n_static] == x[0, :spec.n_static]).all() and (blocks.sum(axis=-1) == 1).all()
            and (blocks.max(axis=-1) == 1).all()):
        raise SystemExit('An encoded row is not of the expected form (one call in each of the four call fields).')
    return x[0, :spec.n_static].copy(), blocks.argmax(axis=-1).astype(np.uint8)


def build_decisions(args):
    import binary
    from bidding import bidding
    spec = sp_common.NetSpec(sp_common.resolve_conf(args.conf))
    statics, bids, steps, cands, scores, outputs, deals, mps = [], [], [], [], [], [], [], []
    stats = Counter()
    sources = Counter()
    deal_no = -1
    for path in games_files(args.games):
        for line in open(path, encoding='utf-8'):
            g = json.loads(line)
            if 'error' in g:
                stats['deals with an error'] += 1
                continue
            deal_no += 1
            dealer_i = SEATS.index(g['dealer'])
            auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in g['calls']]
            encoded = {}
            for d in g.get('decisions', ()):
                stats['turns with candidates'] += 1
                seat = SEATS.index(d['seat'])
                side = 1 if seat % 2 == 0 else -1
                i = dealer_i + d['n']
                step = (i - (seat if seat >= dealer_i else seat + 4)) // 4
                if step >= N_ROUNDS:
                    stats['turns past round 8, left out'] += 1
                    continue
                if auction[i] != to_ben(d['cands'][0][0]):
                    raise SystemExit(f'{path}: the first candidate of a turn is not the call made '
                                     f'({d["cands"][0][0]} against {auction[i]})')
                if seat not in encoded:
                    encoded[seat] = encode(binary, spec, auction, seat, g)
                values = [side * c[3] for c in d['cands']]
                stats['turns where the results differ'] += any(v != values[0] for v in values)
                stats['tried calls'] += len(values)
                for c in d['cands'][1:]:
                    sources[c[4] if len(c) > 4 else 'net'] += 1
                statics.append(encoded[seat][0])
                bids.append(encoded[seat][1])
                steps.append(step)
                cands.append([bidding.BID2ID[to_ben(c[0])] for c in d['cands']])
                scores.append(values)
                outputs.append([c[1] for c in d['cands']])
                deals.append(deal_no)
                mps.append(g['scoring'] == 'MP')
    if not statics:
        raise SystemExit('No turn in these games has candidates.')
    stats['deals'] = deal_no + 1
    stats['rows'] = len(statics)
    width = max(len(c) for c in cands)

    def padded(rows, dtype):
        out = np.zeros((len(rows), width), dtype)
        for k, row in enumerate(rows):
            out[k, :len(row)] = row
        return out

    meta = {'conf': spec.conf_path, 'games': list(games_files(args.games)), **stats, 'most tried calls at a turn': width,
            'tried calls other than the call made, by source': dict(sorted(sources.items()))}
    np.savez(args.out, static=np.stack(statics), bids=np.stack(bids), step=np.array(steps, np.uint8),
             n=np.array([len(c) for c in cands], np.uint8), cand=padded(cands, np.uint8), score=padded(scores, np.int32),
             p=padded(outputs, np.float32), deal=np.array(deals, np.int32), mp=np.array(mps, np.uint8),
             n_deals=np.int64(deal_no + 1), meta=json.dumps(meta))
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    for k, v in meta.items():
        print(f'{k}: {v}')


def build_games(args):
    import binary
    from bidding import bidding
    spec = sp_common.NetSpec(sp_common.resolve_conf(args.conf))
    weight = WEIGHTS[args.weight]
    statics, bids, cls, wgt = [], [], [], []
    stats = Counter()
    stake = Counter()
    by_source = {}
    by_kind = {}
    gain_imps = 0
    for path in games_files(args.games):
        for line in open(path, encoding='utf-8'):
            g = json.loads(line)
            if 'error' in g:
                stats['deals with an error'] += 1
                continue
            stats['deals'] += 1
            dealer_i = SEATS.index(g['dealer'])
            auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in g['calls']]
            rows = {}
            for d in g.get('decisions', ()):
                stats['turns with candidates'] += 1
                seat = SEATS.index(d['seat'])
                targets = targets_of(d, 1 if seat % 2 == 0 else -1, args.target)
                if not targets:
                    continue
                stats['turns where the results differ'] += 1
                i = dealer_i + d['n']
                step = (i - (seat if seat >= dealer_i else seat + 4)) // 4
                if step >= N_ROUNDS:
                    stats['turns past round 8, left out'] += 1
                    continue
                best_gain = 0
                for call, points, source in targets:
                    n_imps = imps(points)
                    stake[n_imps] += 1
                    if source is not None:
                        stats['targets that are not the call made'] += 1
                        best_gain = max(best_gain, n_imps)
                        for table, key in ((by_source, source), (by_kind, kind_of(call))):
                            entry = table.setdefault(key, [0, 0])
                            entry[0] += 1
                            entry[1] += n_imps
                    else:
                        stats['targets that are the call made'] += 1
                    rows.setdefault(seat, []).append((step, bidding.BID2ID[to_ben(call)], weight(n_imps)))
                gain_imps += best_gain
            for seat, targets in rows.items():
                static, bid_ids = encode(binary, spec, auction, seat, g)
                # 'best' has one target a turn, so one row holds them all; 'pair' can have several for one turn
                per_row = {}
                for step, call_id, value in targets:
                    k = 0
                    while (k, step) in per_row:
                        k += 1
                    per_row[k, step] = (call_id, value)
                for k in range(1 + max(k for k, _ in per_row)):
                    c = np.zeros(N_ROUNDS, np.uint8)
                    w = np.zeros(N_ROUNDS, np.uint8)
                    for (kk, step), (call_id, value) in per_row.items():
                        if kk == k:
                            c[step], w[step] = call_id, value
                    statics.append(static)
                    bids.append(bid_ids)
                    cls.append(c)
                    wgt.append(w)
    if not statics:
        raise SystemExit('No turn in these games has candidates with different results.')
    stats['rows'] = len(statics)
    table = lambda t: {k: {'targets': n, 'imps each': round(total / n, 2)} for k, (n, total) in sorted(t.items())}  # noqa: E731
    meta = {'conf': spec.conf_path, 'weight': args.weight, 'target': args.target, 'games': list(games_files(args.games)),
            **stats, 'imps a deal if the best target of every turn had been bid': round(gain_imps / max(stats['deals'], 1), 4),
            'targets not the call made, by source': table(by_source), 'targets not the call made, by kind': table(by_kind),
            'imps at stake': {str(k): v for k, v in sorted(stake.items())}}
    np.savez(args.out, static=np.stack(statics), bids=np.stack(bids), cls=np.stack(cls), wgt=np.stack(wgt), meta=json.dumps(meta))
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    for k, v in meta.items():
        print(f'{k}: {v}')


def build_base(args):
    sys.path.insert(0, os.path.join(sp_common.BEN_ROOT, 'scripts', 'training'))
    import gpu_epochs
    x = np.load(os.path.join(args.bin_dir, 'x.npy'), mmap_mode='r')
    y = np.load(os.path.join(args.bin_dir, 'y.npy'), mmap_mode='r')
    inputs = gpu_epochs.compact_inputs(x, y.shape[2])
    cls = gpu_epochs.compact_one_hot(y)
    if inputs is None or cls is None:
        raise SystemExit(f'{args.bin_dir}: x.npy or y.npy is not of the form the bidding trainer writes')
    static, bids = inputs
    np.savez(args.out, static=static, bids=bids, cls=cls)
    print(f'{len(static)} rows, {static.shape[1]} static features -> {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='what', required=True)
    g = sub.add_parser('games')
    g.add_argument('--conf', required=True, help="the learner's config (for the input encoding)")
    g.add_argument('--out', required=True)
    g.add_argument('--weight', choices=sorted(WEIGHTS), default='mix')
    g.add_argument('--target', choices=['best', 'pair'], default='best', help='see targets_of')
    g.add_argument('games', nargs='+', help='explore run folders or games.jsonl files')
    d = sub.add_parser('decisions')
    d.add_argument('--conf', required=True, help="the learner's config (for the input encoding)")
    d.add_argument('--out', required=True)
    d.add_argument('games', nargs='+', help='explore run folders or games.jsonl files')
    b = sub.add_parser('base')
    b.add_argument('bin_dir')
    b.add_argument('out')
    args = parser.parse_args()
    {'games': build_games, 'decisions': build_decisions, 'base': build_base}[args.what](args)


if __name__ == '__main__':
    main()
