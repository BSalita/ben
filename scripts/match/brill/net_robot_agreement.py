#!/usr/bin/env python3
"""How often the network the bidding search uses for partner and opponents makes the Brill robot's call,
over every answer in the shared cache of Brill's /bid service (missed-game leak, 2026-11-23).

The cache (brill_live.py's --cache) holds the robots' calls in North, East and West for every
auction our configs reached with Brill's service, plus the calls asked by sample_check.py. Each
entry is a hand, a seat and the calls before it. Here each is put to a network the way
BotBid.bidding_rollout does (binary.get_auction_binary_sampling, the model's pred_fun_seq, the legal
call with the highest output), so the agreement is that of one step of the search's continuations.

    python net_robot_agreement.py --conf CONF --cache robot_calls.jsonl [--tags TAGS.jsonl] --out OUT.json [--model opponent|partner|bidder]

--tags: lines {"key": cache key, "tag": name} for subsets to report apart (sample_check.py's
continuations: tags.py below builds them from roll-*.jsonl). Writes OUT.json and prints a table.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
FROM_CTX = {'--': 'PASS', 'Db': 'X', 'Rd': 'XX'}


def parse_key(key):
    seat, hand, dealer, vul, scoring, ctx = key.split('|')
    calls = [FROM_CTX.get(ctx[i:i + 2], ctx[i:i + 2]) for i in range(0, len(ctx), 2)]
    return SEATS.index(seat), hand, SEATS.index(dealer), vul, scoring, calls


def kind(call):
    return 'pass' if call in ('P', 'PASS') else 'double' if call in ('X', 'XX') else 'bid'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--conf', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--tags')
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='opponent', choices=['opponent', 'partner', 'bidder'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch', type=int, default=8192)
    args = p.parse_args()
    for k in ('cache', 'tags', 'out'):          # main() changes to BEN's src
        if getattr(args, k):
            setattr(args, k, os.path.abspath(getattr(args, k)))

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    sys.path.insert(0, '/ben/scripts/match')
    import ab_match
    import numpy as np
    os.chdir(ab_match.SRC)
    from nn.models_tf2 import Models
    from bidding import bidding
    import binary

    conf = ab_match.load_conf(ab_match.resolve(args.conf), None)
    models = Models.from_conf(conf, ab_match.BEN_ROOT)
    ab_match.platform_fixups(models)
    net = {'opponent': models.opponent_model, 'partner': getattr(models, 'partner_model', models.opponent_model),
           'bidder': models.bidder_model}[args.model]
    n_cards = models.n_cards_bidding
    parse_hand = binary.parse_hand_f(n_cards)

    tags = defaultdict(set)
    if args.tags:
        for line in open(args.tags):
            r = json.loads(line)
            tags[r['key']].add(r['tag'])

    entries = {}
    for line in open(args.cache, encoding='utf-8'):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        entries[r['key']] = r['bid']
    keys = list(entries)
    if args.limit:
        keys = keys[:args.limit]
    print(f'{len(keys)} cached robot calls, {sum(1 for k in keys if k in tags)} tagged', flush=True)

    t0 = time.time()
    by_steps = defaultdict(list)          # n_steps -> [(key, X row)]
    legal_cache = {}
    meta = {}
    for key in keys:
        seat, hand, dealer, vul, scoring, calls = parse_key(key)
        auction = ['PAD_START'] * dealer + calls
        i = seat
        while i < len(auction) and auction[i] == 'PAD_START':
            i += 4
        n_steps = 1 + (len(auction) - i) // 4
        h = np.array(parse_hand(hand), dtype=np.float32).reshape(1, n_cards)
        X = binary.get_auction_binary_sampling(n_steps, list(auction), seat, h, VULN[vul], models, n_cards)
        by_steps[n_steps].append((key, X[0]))
        ctx_key = (dealer, tuple(calls))
        if ctx_key not in legal_cache:
            legal_cache[ctx_key] = np.array([k == 2 or (k > 2 and bidding.can_bid(bidding.ID2BID[k], auction)) for k in range(40)])
        bids = [c for c in calls if c not in ('PASS', 'X', 'XX')]
        ns_bid = any(c not in ('PASS', 'X', 'XX') for j, c in enumerate(calls) if (dealer + j) % 2 == 0)
        ew_bid = any(c not in ('PASS', 'X', 'XX') for j, c in enumerate(calls) if (dealer + j) % 2 == 1)
        south_calls = [c for j, c in enumerate(calls) if (dealer + j) % 4 == 2]
        meta[key] = {'seat': 'North' if seat == 0 else 'East/West', 'steps': n_steps,
                     'contested': ns_bid and ew_bid, 'opened': bool(bids), 'ctx': ctx_key,
                     'south_acted': any(c != 'PASS' for c in south_calls)}
    print(f'inputs built in {time.time() - t0:.0f} s; groups by turn: {sorted((k, len(v)) for k, v in by_steps.items())}', flush=True)

    rows = []
    for n_steps, items in sorted(by_steps.items()):
        for b in range(0, len(items), args.batch):
            chunk = items[b:b + args.batch]
            X = np.stack([x for _, x in chunk])
            out, _ = net.pred_fun_seq(X)
            out = np.array(out)
            if out.ndim == 3:
                out = out[:, -1, :]
            for (key, _), soft in zip(chunk, out):
                legal = legal_cache[meta[key]['ctx']]
                masked = np.where(legal, soft, -1.0)
                call = bidding.ID2BID[int(np.argmax(masked))]
                real = entries[key]
                real_ben = 'PASS' if real == 'P' else real
                rows.append((key, call == real_ben, float(soft[bidding.BID2ID[real_ben]]) if real_ben in bidding.BID2ID else 0.0,
                             'P' if call == 'PASS' else call, real))
        print(f'turn {n_steps}: {len(items)} done, {time.time() - t0:.0f} s', flush=True)

    total = defaultdict(Counter)

    def add(tag, same, p):
        total[tag]['n'] += 1
        total[tag]['same'] += same
        total[tag]['p'] += p

    confusion = Counter()
    examples = []
    for key, same, p, net_call, real in rows:
        m = meta[key]
        groups = ['all', m['seat'], f"robot {kind(real)}", f"{m['seat']} robot {kind(real)}",
                  f"turn {min(m['steps'], 5)}{'+' if m['steps'] >= 5 else ''}",
                  'contested' if m['contested'] else 'not contested',
                  'South has bid or doubled' if m['south_acted'] else 'South only passed (or not yet called)']
        later = 'turn 3+' if m['steps'] >= 3 else f"turn {m['steps']}"
        cont = 'contested' if m['contested'] else 'not contested'
        groups += [f"{m['seat']} {cont} {later}", f"{m['seat']} {cont} {later} robot {kind(real)}"]
        groups += [f'tag {t}' for t in tags.get(key, ())]
        groups += [f'tag {t} {m["seat"]} robot {kind(real)}' for t in tags.get(key, ())]
        groups += [f'tag {t} {m["seat"]}' for t in tags.get(key, ())]
        for g in groups:
            add(g, same, p)
        if not same:
            confusion[(kind(real), kind(net_call))] += 1
            if tags.get(key) and m['seat'] == 'North':
                examples.append({'key': key, 'robot': real, 'network': net_call, 'tags': sorted(tags[key])})
    res = {g: {'n': c['n'], 'same': c['same'] / c['n'], 'p_real': c['p'] / c['n']} for g, c in total.items()}
    res['_confusion'] = {f'{a} -> {b}': v for (a, b), v in confusion.most_common()}
    res['_north_tagged_misses'] = examples
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f"\n{'group':<48} {'calls':>8} {'same call':>10} {'output for robot call':>22}")
    order = sorted((g for g in res if not g.startswith('_')), key=lambda g: (g.startswith('tag'), g))
    for g in order:
        r = res[g]
        print(f"{g:<48} {r['n']:>8} {100 * r['same']:>9.1f}% {r['p_real']:>22.3f}")
    print('\ndisagreements, robot call -> network call:', res['_confusion'])


if __name__ == '__main__':
    main()
