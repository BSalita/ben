#!/usr/bin/env python3
"""Are the search's samples deals Brill's robots would have bid this way? (missed-game leak, 2026-11-23)

On the leak positions of leak_diagnosis.py (South's deciding turn), bid the position again as the
replay did, with the table's call added to the candidates when the bidder did not offer it, and
keep every sample the search used and every candidate's score on each sample. Then ask Brill's
service, for every sample, what its robots in North, East and West would have called at each of
their earlier turns of the auction on that sample's hands. A sample is consistent when the robots
repeat every earlier call. The search's values are then recomputed from the consistent samples
alone (BEN's own calculate_imp_score / calculate_mp_score with weights), which shows whether the
samples that do not fit the auction are what keeps the bidder from the game.

Three steps:

  python sample_check.py dump --conf CONF --nodes NODES.jsonl --out OUT [--variant base|big] [--workers N]   (BEN container)
  python sample_check.py brill --dump OUT/dump-base.jsonl --out OUT/brill-base.jsonl --cache CACHE --max-requests N  (container with the shared cache)
  python sample_check.py score --dump OUT/dump-base.jsonl --brill OUT/brill-base.jsonl [--nodes NODES.jsonl]       (BEN code on the path)
  python sample_check.py roll --dump OUT/dump-base.jsonl --brill OUT/brill-base.jsonl --out OUT/roll-base.jsonl --cache CACHE --max-requests N
  python sample_check.py roll-score --brill OUT/roll-base.jsonl [--out OUT/roll-score-base.jsonl]

roll: on the consistent samples, the auction after our call and after the table's call finished by
Brill's robots in all four seats (South's later calls as a stand-in for Morty), scored double dummy
as the search scores, beside the search's own continuation (BEN's networks) on the same sample.

--select picks the leak positions (default: game leaks in contested later rounds).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
VARIANTS = {
    'base': {},
    'big': {'sample_hands_auction': 200, 'sample_boards_for_auction': 80000},
}
W = {}


def match_dir():
    for d in (os.path.join(HERE, '..'), '/ben/scripts/match'):
        if os.path.exists(os.path.join(d, 'ab_match.py')):
            return os.path.abspath(d)
    raise SystemExit('ab_match.py not found')


def select(nodes, how):
    if how == 'contested-game':
        return [n for n in nodes if n['leak'] == 'game' and n['seat'] == 'S' and n['stage'].startswith('6 ')]
    if how == 'game':
        return [n for n in nodes if n['leak'] == 'game' and n['seat'] == 'S']
    if how == 'all':
        return [n for n in nodes if n['seat'] == 'S']
    raise SystemExit(f'unknown --select {how}')


def read_nodes(path, how):
    lines = [json.loads(line) for line in open(path, encoding='utf-8')]
    return select([n for n in lines if not n.get('meta')], how)


def norm(hand):
    """A hand with each suit's cards in rank order (the field file has some out of order)."""
    return '.'.join(''.join(sorted(s, key='AKQJT98765432'.index)) for s in hand.split('.'))


def to_ben(call):
    return 'PASS' if call == 'P' else call


def from_ben(call):
    return 'P' if call == 'PASS' else call


# ---------------------------------------------------------------- dump (BEN container)

def worker_init(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    sys.path.insert(0, match_dir())
    import ab_match
    os.chdir(ab_match.SRC)
    log = open(os.path.join(cfg['out'], 'logs', f'dump-{os.getpid()}.log'), 'a')
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
    W['dds'] = DDSolver(max_threads=cfg['dds_threads'])
    print(f"worker {os.getpid()} ready: {cfg['conf']} variant {cfg['variant']}")


def dump_node(node):
    t0 = time.time()
    key = {'tournament_id': node['tournament_id'], 'board': node['board']}
    try:
        return {**key, **bid_and_record(node), 'seconds': round(time.time() - t0, 1)}
    except Exception:
        return {**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 1)}


def bid_and_record(node):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from objects import CandidateBid
    from sample import Sample

    models = W['models']
    hands = node['deal'][2:].split()
    first = SEATS.index(node['deal'][0])
    hands = [hands[(i - first) % 4] for i in range(4)]
    dealer_i = SEATS.index(node['dealer'])
    models.matchpoint = node['scoring'] == 'MP'
    sampler = Sample.from_conf(W['conf'])
    for k, v in VARIANTS[W['cfg']['variant']].items():
        setattr(sampler, '_' + k if k in ('sample_hands_auction', 'min_sample_hands_auction') else k, v)  # read-only properties
    np.random.seed(42)
    bot = BotBid(VULN[node['vul']], hands[2], models, sampler, 2, dealer_i, W['dds'], False, False)
    auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in node['prefix']]
    add_call = to_ben(node['their_call'])
    cap = {'ev': {}, 'contracts': {}, 'auctions': {}, 'cur': None, 'added': False}

    real_next = bot.next_bid_np

    def next_bid_np(a):
        r = real_next(a)
        if 'soft' not in cap and list(a) == auction:
            cap['soft'] = np.array(r[0], dtype=float).copy()
        return r

    real_candidates = bot.get_bid_candidates

    def get_bid_candidates(a):
        cands, passout = real_candidates(a)
        cap['offered'] = [(from_ben(c.bid), round(float(c.insta_score), 4)) for c in cands]
        if all(c.bid != add_call for c in cands):
            cands.append(CandidateBid(bid=add_call, insta_score=float(cap['soft'][bidding.BID2ID[add_call]]), alert=None))
            cap['added'] = True
        return cands, passout

    real_sha = bot.sample_hands_for_auction

    def sample_hands_for_auction(a, t):
        r = real_sha(a, t)
        if 'scores' not in cap:
            cap['scores'] = [round(float(x), 4) for x in r[1]]
            cap['quality'] = float(r[4])
        return r

    real_tr = bot.translate_hands

    def translate_hands(h, s, n):
        r = real_tr(h, s, n)
        if 'pbn' not in cap:
            cap['pbn'] = list(r)
        return r

    real_ro = bot.bidding_rollout

    def bidding_rollout(a, cb, h, p):
        r = real_ro(a, cb, h, p)
        if list(a) == auction:
            cap['cur'] = cb
            cap['auctions'][from_ben(cb)] = [bidding.get_auction_as_string(x) for x in r]
        return r

    real_es = bot.expected_score

    def expected_score(t, contracts, softmax):
        r = real_es(t, contracts, softmax)
        if cap['cur'] is not None:
            c = from_ben(cap['cur'])
            cap['ev'][c] = [float(x) for x in r]
            cap['contracts'][c] = [str(x) for x in contracts]
            cap['cur'] = None
        return r

    bot.next_bid_np, bot.get_bid_candidates = next_bid_np, get_bid_candidates
    bot.sample_hands_for_auction, bot.translate_hands = sample_hands_for_auction, translate_hands
    bot.bidding_rollout, bot.expected_score = bidding_rollout, expected_score
    resp = bot.bid(auction)

    # translate_hands puts the bidder's hand first and the others clockwise after it, labelled 'N:'
    samples = []
    for pbn in cap.get('pbn', []):
        h = pbn[2:].split()
        absolute = {SEATS[(2 + j) % 4]: h[j] for j in range(4)}
        assert norm(absolute['S']) == norm(hands[2]), (absolute['S'], hands[2])
        samples.append(absolute)

    def sim(c):
        v = c.expected_mp if models.matchpoint else c.expected_imp
        return c.expected_score if v is None else v

    return {
        'variant': W['cfg']['variant'], 'call': from_ben(resp.bid), 'who': resp.who,
        'quality': cap.get('quality'), 'n_samples': len(samples), 'added': cap['added'], 'offered': cap.get('offered'),
        'result': [{'call': from_ben(c.bid), 'p': None if c.insta_score is None else round(c.insta_score, 4),
                    'sim': None if sim(c) is None else round(sim(c), 3),
                    'adjust': None if c.adjust is None else round(c.adjust, 3)} for c in resp.candidates],
        'scores': cap.get('scores'), 'samples': samples, 'ev': cap['ev'], 'contracts': cap['contracts'],
        'auctions': cap['auctions'],
    }


def dump(args):
    nodes = read_nodes(args.nodes, args.select)
    if args.limit:
        nodes = nodes[:args.limit]
    args.out = os.path.abspath(args.out)    # the workers chdir to BEN's src
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    path = os.path.join(args.out, f'dump-{args.variant}.jsonl')
    done = set()
    if os.path.exists(path):
        done = {(r['tournament_id'], r['board']) for r in map(json.loads, open(path)) if 'error' not in r}
    tasks = [n for n in nodes if (n['tournament_id'], n['board']) not in done]
    sys.path.insert(0, match_dir())
    import ab_match
    cfg = {'conf': ab_match.resolve(args.conf), 'out': args.out, 'variant': args.variant,
           'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads}
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{len(tasks)} of {len(nodes)} positions, variant {args.variant}, {workers} workers', flush=True)
    t0, errors = time.time(), 0
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, r in enumerate(pool.imap_unordered(dump_node, tasks, chunksize=1), start=1):
                out.write(json.dumps(r) + '\n')
                out.flush()
                errors += 'error' in r
                print(f'[{k}/{len(tasks)}] {time.time() - t0:6.0f} s  errors {errors}', flush=True)
    return 1 if errors else 0


# ---------------------------------------------------------------- brill (container with the shared cache)

def brill(args):
    sys.path.insert(0, HERE)
    sys.path.insert(0, '/ben/scripts/match/brill')
    import brill_live as bl

    nodes = {(n['tournament_id'], n['board']): n for n in read_nodes(args.nodes, 'all')}
    bl.W['cfg'] = {'cache': os.path.abspath(args.cache), 'url': bl.BASE, 'pause': 0, 'daily_budget': args.daily_budget}
    bl.W['lock'] = bl.FileLock(args.cache + '.lock')
    bl.W['cache'] = {}
    bl.W['cache_size'] = 0
    bl.read_cache()
    done = set()
    if os.path.exists(args.out):
        done = {(r['tournament_id'], r['board']) for r in map(json.loads, open(args.out))}
    asked = 0
    dumps = [r for r in map(json.loads, open(args.dump)) if 'error' not in r]
    t0 = time.time()
    with open(args.out, 'a') as out:
        for d in dumps:
            key = (d['tournament_id'], d['board'])
            if key in done:
                continue
            node = nodes[key]
            board = {'board': node['board'], 'dealer': node['dealer'], 'vul': node['vul'], 'scoring': node['scoring']}
            dealer_i = SEATS.index(node['dealer'])
            prefix = node['prefix']
            hands = node['deal'][2:].split()
            first = SEATS.index(node['deal'][0])
            real = {SEATS[(first + j) % 4]: norm(hands[j]) for j in range(4)}

            def check(hset):
                """(index of the first earlier call the robots do not repeat, or None; requests made)"""
                nonlocal asked
                for i, call in enumerate(prefix):
                    seat_i = (dealer_i + i) % 4
                    if SEATS[seat_i] == 'S':
                        continue
                    got, new = bl.robot_call(board, seat_i, hset[SEATS[seat_i]], prefix[:i])
                    asked += new
                    if got != call:
                        return i, got
                return None, None

            if asked >= args.max_requests:
                print(f'stopped at --max-requests {args.max_requests}', flush=True)
                break
            try:
                real_miss = check(real)
                rows = []
                for s in d['samples']:
                    if asked >= args.max_requests:
                        raise bl.BudgetExhausted(f'--max-requests {args.max_requests}')
                    miss, got = check(s)
                    rows.append(None if miss is None else [miss, got])
            except bl.BudgetExhausted as e:
                print(f'stopped: {e}', flush=True)
                break
            out.write(json.dumps({'tournament_id': key[0], 'board': key[1], 'variant': d['variant'],
                                  'real_miss': None if real_miss[0] is None else list(real_miss), 'miss': rows}) + '\n')
            out.flush()
            ok = sum(r is None for r in rows)
            print(f'{key[0][:8]} {key[1]:>3}: {ok}/{len(rows)} consistent, real deal {"ok" if real_miss[0] is None else real_miss}, '
                  f'{asked} requests so far, {time.time() - t0:5.0f} s', flush=True)
    print(f'done: {asked} requests', flush=True)


# ---------------------------------------------------------------- roll (continuations by Brill's robots)

def roll(args):
    """On every consistent sample, finish the auction after our call and after the table's call with
    Brill's robots in all four seats (South's later calls too, as a stand-in for Morty), solve the
    contract double dummy as the search does, and keep the score beside the search's own continuation."""
    import numpy as np
    sys.path.insert(0, '/ben/src')
    sys.path.insert(0, '/ben/scripts/match/brill')
    import brill_live as bl
    from bidding import bidding
    import scoring
    from ddsolver.ddsolver import DDSolver

    dds = DDSolver(max_threads=4)
    nodes = {(n['tournament_id'], n['board']): n for n in read_nodes(args.nodes, 'all')}
    dumps = {(r['tournament_id'], r['board']): r for r in map(json.loads, open(args.dump)) if 'error' not in r}
    checks = [json.loads(line) for line in open(args.brill)]
    bl.W['cfg'] = {'cache': os.path.abspath(args.cache), 'url': bl.BASE, 'pause': 0, 'daily_budget': args.daily_budget}
    bl.W['lock'] = bl.FileLock(args.cache + '.lock')
    bl.W['cache'], bl.W['cache_size'] = {}, 0
    bl.read_cache()
    done = set()
    if os.path.exists(args.out):
        done = {(r['tournament_id'], r['board']) for r in map(json.loads, open(args.out))}
    asked, t0 = 0, time.time()

    def dd_score(sample, contract, vuln):
        if contract is None:
            return 0.0
        pbn = 'N:' + ' '.join(sample[SEATS[(2 + j) % 4]] for j in range(4))   # South first, as translate_hands
        strain = 'NSHDC'.index(contract[1])
        leader = ((SEATS.index(contract[-1]) + 1) % 4 + 4 - 2) % 4
        tricks = 13 - dds.solve(strain, leader, [], [pbn], 1, purpose='bid')['max'][0]
        by_trick = scoring.contract_scores_by_trick(contract, tuple(vuln))
        sign = -1 if (2 + SEATS.index(contract[-1])) % 2 == 1 else 1
        return float(sign * by_trick[tricks])

    with open(args.out, 'a') as out:
        for c in checks:
            key = (c['tournament_id'], c['board'])
            if key in done:
                continue
            n, d = nodes[key], dumps[key]
            board = {'board': n['board'], 'dealer': n['dealer'], 'vul': n['vul'], 'scoring': n['scoring']}
            dealer_i = SEATS.index(n['dealer'])
            pad = ['PAD_START'] * dealer_i
            rows = []
            try:
                for i, miss in enumerate(c['miss']):
                    if miss is not None:
                        continue
                    s = d['samples'][i]
                    row = {'i': i}
                    for cand in (n['our_call'], n['their_call']):
                        if asked >= args.max_requests:
                            raise bl.BudgetExhausted(f'--max-requests {args.max_requests}')
                        walk = list(n['prefix']) + [cand]
                        while not bidding.auction_over(pad + [to_ben(x) for x in walk]):
                            seat_i = (dealer_i + len(walk)) % 4
                            call, new = bl.robot_call(board, seat_i, s[SEATS[seat_i]], walk)
                            asked += new
                            if not bidding.can_bid(to_ben(call), pad + [to_ben(x) for x in walk]):
                                raise ValueError(f'illegal {call} after {walk}')
                            walk.append(call)
                        contract = bidding.get_contract(pad + [to_ben(x) for x in walk])
                        ben_walk = d['auctions'][cand][i]
                        ben_contract = bidding.get_contract(pad + [to_ben(x) for x in ben_walk.split('-')])
                        row[cand] = {'walk': '-'.join(walk), 'contract': contract, 'score': dd_score(s, contract, VULN[n['vul']]),
                                     'ben_score': d['ev'][cand][i], 'ben_walk': ben_walk,
                                     'ben_check': dd_score(s, ben_contract, VULN[n['vul']])}   # should equal ben_score
                    rows.append(row)
            except bl.BudgetExhausted as e:
                print(f'stopped: {e}', flush=True)
                break
            out.write(json.dumps({'tournament_id': key[0], 'board': key[1], 'rows': rows}) + '\n')
            out.flush()
            print(f'{key[0][:8]} {key[1]:>3}: {len(rows)} samples, {asked} requests so far, {time.time() - t0:5.0f} s', flush=True)
    print(f'done: {asked} requests', flush=True)


def roll_score(args):
    import numpy as np
    sys.path.insert(0, '/ben/src')
    import calculate
    nodes = {(n['tournament_id'], n['board']): n for n in read_nodes(args.nodes, 'all')}
    out = []
    print(f"{'board':<13} {'sc':<3} {'ours':>4} {'theirs':>6} {'n':>4}  {'same walk o/t':>13}  "
          f"{'BEN cont: ours-theirs':>21} {'Brill cont: ours-theirs':>23}  imp")
    for r in map(json.loads, open(args.brill)):
        n = nodes[(r['tournament_id'], r['board'])]
        o, t = n['our_call'], n['their_call']
        rows = r['rows']
        if len(rows) < args.min_consistent:
            continue
        f = calculate.calculate_mp_score if n['scoring'] == 'MP' else calculate.calculate_imp_score
        ben = f({o: [x[o]['ben_score'] for x in rows], t: [x[t]['ben_score'] for x in rows]})
        bri = f({o: [x[o]['score'] for x in rows], t: [x[t]['score'] for x in rows]})
        same_o = sum(x[o]['walk'] == x[o]['ben_walk'] for x in rows)
        same_t = sum(x[t]['walk'] == x[t]['ben_walk'] for x in rows)
        row = {'key': [r['tournament_id'], r['board']], 'scoring': n['scoring'], 'ours': o, 'theirs': t, 'n': len(rows),
               'ben': ben[o] - ben[t], 'brill': bri[o] - bri[t], 'same_o': same_o, 'same_t': same_t, 'imp': n['imp']}
        out.append(row)
        print(f"{r['tournament_id'][:8]} {r['board']:>3} {n['scoring']:<3} {o:>4} {t:>6} {len(rows):>4}  {same_o:>6}/{same_t:<6}  "
              f"{row['ben']:>+21.3f} {row['brill']:>+23.3f}  {n['imp']:+d}")
    for label, k in (('BEN continuations prefer theirs', 'ben'), ('Brill continuations prefer theirs', 'brill')):
        sel = [x for x in out if x[k] < 0]
        print(f'  {label}: {len(sel)} of {len(out)} (table result {sum(-x["imp"] for x in sel):+d} IMPs for those)')
    if args.out:
        with open(args.out, 'w') as f:
            for x in out:
                f.write(json.dumps(x) + '\n')


# ---------------------------------------------------------------- score

def hcp(hand):
    return sum({'A': 4, 'K': 3, 'Q': 2, 'J': 1}.get(c, 0) for c in hand)


def score(args):
    import numpy as np
    sys.path.insert(0, '/ben/src')
    import calculate

    nodes = {(n['tournament_id'], n['board']): n for n in read_nodes(args.nodes, 'all')}
    dumps = {(r['tournament_id'], r['board']): r for r in map(json.loads, open(args.dump)) if 'error' not in r}
    checks = {(r['tournament_id'], r['board']): r for r in map(json.loads, open(args.brill))}
    out = []
    print(f"{'board':<13} {'sc':<3} {'ours':>4} {'theirs':>6} {'n':>4} {'cons':>4} {'real':>4}  "
          f"{'all: ours-theirs':>16} {'cons: ours-theirs':>17}  {'N hcp all/cons':>14}  imp")
    for key, c in checks.items():
        d, n = dumps[key], nodes[key]
        ours, theirs = n['our_call'], n['their_call']
        ev = d['ev']
        if ours not in ev or theirs not in ev:
            print(f'{key[0][:8]} {key[1]:>3}: our or their call not searched ({list(ev)})')
            continue
        f = calculate.calculate_mp_score if n['scoring'] == 'MP' else calculate.calculate_imp_score
        cons = np.array([m is None for m in c['miss']], dtype=float)
        m = len(cons)
        data = {k: v[:m] for k, v in ev.items()}
        v_all = f(data)
        v_cons = f(data, cons) if cons.sum() >= args.min_consistent else None
        n_hcp = np.array([hcp(s['N']) for s in d['samples'][:m]])
        row = {'tournament_id': key[0], 'board': key[1], 'scoring': n['scoring'], 'ours': ours, 'theirs': theirs,
               'n': m, 'consistent': int(cons.sum()), 'real_ok': c['real_miss'] is None,
               'all': {k: v_all[k] for k in (ours, theirs)}, 'cons': None if v_cons is None else {k: v_cons[k] for k in (ours, theirs)},
               'all_best': max(v_all, key=v_all.get), 'cons_best': None if v_cons is None else max(v_cons, key=v_cons.get),
               'n_hcp_all': float(n_hcp.mean()), 'n_hcp_cons': float(n_hcp[cons > 0].mean()) if cons.sum() else None,
               'n_hcp_real': hcp(dict(zip([SEATS[(SEATS.index(n['deal'][0]) + j) % 4] for j in range(4)], n['deal'][2:].split()))['N']),
               'imp': n['imp'], 'quality': d['quality']}
        out.append(row)
        da = v_all[ours] - v_all[theirs]
        dc = '' if v_cons is None else f'{v_cons[ours] - v_cons[theirs]:+.3f} ({row["cons_best"]})'
        hc = '' if row['n_hcp_cons'] is None else f'{row["n_hcp_cons"]:.1f}'
        print(f"{key[0][:8]} {key[1]:>3} {n['scoring']:<3} {ours:>4} {theirs:>6} {m:>4} {int(cons.sum()):>4} {'ok' if row['real_ok'] else 'no':>4}  "
              f"{da:>+9.3f} ({row['all_best']:>3}) {dc:>17}  {row['n_hcp_all']:5.1f}/{hc:>5} ({row['n_hcp_real']})  {n['imp']:+d}")
    if args.out:
        with open(args.out, 'w') as f:
            for r in out:
                f.write(json.dumps(r) + '\n')
    rows = [r for r in out if r['cons'] is not None]
    print(f"\n{len(out)} positions; consistent samples {sum(r['consistent'] for r in out)} of {sum(r['n'] for r in out)} "
          f"({100 * sum(r['consistent'] for r in out) / max(1, sum(r['n'] for r in out)):.0f}%); "
          f"robots repeat the real auction on {sum(r['real_ok'] for r in out)}")
    for label, test in (('all samples prefer theirs', lambda r: r['all'][r['theirs']] > r['all'][r['ours']]),
                        ('consistent samples prefer theirs', lambda r: r['cons'][r['theirs']] > r['cons'][r['ours']])):
        sel = [r for r in rows if test(r)]
        print(f'  {label}: {len(sel)} of {len(rows)} (table result {sum(-r["imp"] for r in sel):+d} IMPs for those)')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mode', choices=['dump', 'brill', 'score', 'roll', 'roll-score'])
    p.add_argument('--conf')
    p.add_argument('--nodes', required=True)
    p.add_argument('--out')
    p.add_argument('--dump')
    p.add_argument('--brill')
    p.add_argument('--cache')
    p.add_argument('--select', default='contested-game')
    p.add_argument('--variant', default='base', choices=list(VARIANTS))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--dds-threads', type=int, default=3)
    p.add_argument('--tf-threads', type=int, default=2)
    p.add_argument('--max-requests', type=int, default=0)
    p.add_argument('--daily-budget', type=int, default=100000)
    p.add_argument('--min-consistent', type=int, default=5)
    args = p.parse_args()
    if args.mode == 'dump':
        return dump(args)
    if args.mode == 'brill':
        return brill(args)
    if args.mode == 'roll':
        return roll(args)
    if args.mode == 'roll-score':
        return roll_score(args)
    return score(args)


if __name__ == '__main__':
    sys.exit(main())
