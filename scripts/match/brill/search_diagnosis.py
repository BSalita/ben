#!/usr/bin/env python3
"""Why a replayed config's South call differs from a Brill entrant's: network, candidates or search.

For every board the entrant played, take South's first call where the config's
replay and the entrant part ways. Up to there the two auctions are the same, so
the config can be asked again in exactly that position, and what its bidder did
can be read off:

  one candidate     the network offered only the call we made; no search was run
  not a candidate   several calls were searched, the entrant's was not among them
  search chose ours the entrant's call was searched and scored below ours
  other             the network was followed because the samples were poor, or
                    the final-contract check changed the call

Then the position is bid once more with the entrant's call added to the
candidates, which shows what the search makes of it when it has to look.

Three steps, the middle one in the BEN container:

  elo/.venv/Scripts/python.exe search_diagnosis.py export FIELD.jsonl BOARDS.parquet CONFIG NODES.jsonl [ENTRANT]
  python search_diagnosis.py run --conf CONF --nodes NODES.jsonl --out OUT [--workers N]
  elo/.venv/Scripts/python.exe search_diagnosis.py summary NODES.jsonl OUT/diag.jsonl [TABLES.txt]

BOARDS.parquet is score_brill_replay.py's output (the config's auction and score
on each board). 'diff' is the config's matchpoint percentage against the other
tables minus the entrant's, as in gap_analysis.py; summed over a group and
divided by all compared boards it is that group's share of the gap.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
W = {}


# ---------------------------------------------------------------- export

def export(field_path, boards_path, config, nodes_path, entrant='R:BBOBen'):
    import polars as pl
    sys.path.insert(0, HERE)
    from gap_analysis import contract_class, direction, pct_against, stage, HCP

    field = {(b['tournament_id'], b['board']): b for b in map(json.loads, open(field_path, encoding='utf-8'))}
    df = pl.read_parquet(boards_path).filter(pl.col('config') == config)
    n_boards, gap, same, nodes = 0, 0.0, 0, []
    for r in df.select('tournament_id', 'board', 'auction', 'score', 'source').iter_rows(named=True):
        key = (r['tournament_id'], r['board'])
        b = field.get(key)
        if b is None:
            continue
        t = next((t for t in b['tables'] if t['player_id'] == entrant), None)
        others = [x['ns_score'] for x in b['tables'] if x['player_id'] != entrant]
        if t is None or not others:
            continue
        diff = pct_against(r['score'], others) - pct_against(t['ns_score'], others)
        n_boards += 1
        gap += diff
        a, e = r['auction'].split('-'), t['auction']
        i = next((i for i in range(min(len(a), len(e))) if a[i] != e[i]), None)
        if i is None:
            same += 1
            continue
        dealer_i = SEATS.index(b['dealer'])
        south = b['deal'][2:].split()[2]
        hcp = sum(HCP.get(c, 0) for c in south)
        oc, tc = contract_class(a, dealer_i), contract_class(e, dealer_i)
        nodes.append({
            'tournament_id': r['tournament_id'], 'board': r['board'], 'deal': b['deal'], 'dealer': b['dealer'],
            'vul': b['vul'], 'scoring': b['scoring'], 'seat': SEATS[(dealer_i + i) % 4],
            'prefix': a[:i], 'our_call': a[i], 'their_call': e[i], 'diff': diff,
            'stage': stage(a[:i], dealer_i), 'direction': direction(a[i], e[i]),
            'ours': f'{oc[0]} {oc[1]}{oc[2]}', 'theirs': f'{tc[0]} {tc[1]}{tc[2]}', 'south': south, 'hcp': hcp,
            'our_auction': r['auction'], 'their_auction': '-'.join(e), 'our_score': r['score'],
            'their_score': t['ns_score'], 'source': r['source'],
        })
    meta = {'meta': True, 'config': config, 'entrant': entrant, 'boards': n_boards, 'gap_sum': gap, 'same_auction': same}
    with open(nodes_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(meta) + '\n')
        for n in nodes:
            f.write(json.dumps(n) + '\n')
    print(f'{config} against {entrant}: {n_boards} boards, gap {gap / n_boards:+.2f} points a board; '
          f'{len(nodes)} boards where the auctions part ({sum(n["seat"] == "S" for n in nodes)} at a South call) -> {nodes_path}')


# ---------------------------------------------------------------- run (in the container)

def worker_init(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    import ab_match
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
    W['dds'] = DDSolver(max_threads=cfg['dds_threads'])
    print(f"worker {os.getpid()} ready: {cfg['conf']}")


def to_ben(call):
    return 'PASS' if call == 'P' else call


def from_ben(call):
    return 'P' if call == 'PASS' else call


def one_bid(node, add_call=None, probe=None):
    """Bid the node's position as brill_replay.py does and report what the bidder did.

    add_call: a call (BEN spelling) put among the candidates when the bidder did not pick it itself.
    probe: a call (P, X, 1C..) whose network probability is reported as 'probe'.
    """
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from objects import CandidateBid
    from sample import Sample

    models = W['models']
    hands = node['deal'][2:].split()
    dealer_i = SEATS.index(node['dealer'])
    models.matchpoint = node['scoring'] == 'MP'
    np.random.seed(42)
    bot = BotBid(VULN[node['vul']], hands[2], models, Sample.from_conf(W['conf']), 2, dealer_i, W['dds'], False, False)
    auction = ['PAD_START'] * dealer_i + [to_ben(c) for c in node['prefix']]
    seen = {'added': False}

    real_next = bot.next_bid_np

    def next_bid_np(a):
        r = real_next(a)
        if 'soft' not in seen and list(a) == auction:
            seen['soft'] = np.array(r[0], dtype=float).copy()
        return r

    real_candidates = bot.get_bid_candidates

    def get_bid_candidates(a):
        cands, passout = real_candidates(a)
        seen['offered'] = [(c.bid, c.insta_score, c.who) for c in cands]
        seen['passout'] = passout
        if add_call and all(c.bid != add_call for c in cands):
            cands.append(CandidateBid(bid=add_call, insta_score=float(seen['soft'][bidding.BID2ID[add_call]]), alert=None))
            seen['added'] = True
        return cands, passout

    real_rollout = bot.do_rollout

    def do_rollout(a, cands, max_score, sample_count):
        r = real_rollout(a, cands, max_score, sample_count)
        seen['searched'] = bool(r)
        seen['samples'] = int(sample_count)
        return r

    bot.next_bid_np, bot.get_bid_candidates, bot.do_rollout = next_bid_np, get_bid_candidates, do_rollout
    resp = bot.bid(auction)

    def sim(c):
        v = c.expected_mp if models.matchpoint else c.expected_imp
        if v is None:
            v = c.expected_score
        return v

    soft = seen['soft']
    return {
        'call': from_ben(resp.bid), 'who': resp.who, 'quality': None if resp.quality is None else round(float(resp.quality), 3),
        'searched': seen.get('searched'), 'samples': seen.get('samples'), 'passout': bool(seen['passout']),
        'added': seen['added'], 'bid_no': int(bot.my_bid_no), 'threshold': float(bot.get_min_candidate_score(bot.my_bid_no)),
        'offered': [{'call': from_ben(b), 'p': round(float(p), 4), 'who': w} for b, p, w in seen['offered']],
        'result': [{'call': from_ben(c.bid), 'p': None if c.insta_score is None else round(c.insta_score, 4),
                    'sim': None if sim(c) is None else round(sim(c), 3),
                    'adjust': None if c.adjust is None else round(c.adjust, 3),
                    'score': None if c.expected_score is None else round(c.expected_score, 1), 'who': c.who}
                   for c in resp.candidates],
        'net': {from_ben(bidding.ID2BID[i]): round(float(soft[i]), 4) for i in np.argsort(-soft)[:6]},
        'probe': None if probe is None else round(float(soft[bidding.BID2ID[to_ben(probe)]]), 5),
    }


def run_node(node):
    t0 = time.time()
    key = {'tournament_id': node['tournament_id'], 'board': node['board']}
    try:
        natural = one_bid(node, probe=node['their_call'])
        out = {**key, 'natural': natural, 'their_p': natural['probe']}
        if all(c['call'] != node['their_call'] for c in natural['offered']):
            out['forced'] = one_bid(node, add_call=to_ben(node['their_call']))
        return {**out, 'seconds': round(time.time() - t0, 1)}
    except Exception:
        return {**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 1)}


def run(args):
    nodes = [json.loads(line) for line in open(args.nodes, encoding='utf-8')]
    nodes = [n for n in nodes if not n.get('meta') and n['seat'] == 'S']
    if args.limit:
        nodes = nodes[:args.limit]
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    path = os.path.join(args.out, 'diag.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['tournament_id'], r['board']))
    tasks = [n for n in nodes if (n['tournament_id'], n['board']) not in done]
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    import ab_match
    cfg = {'conf': ab_match.resolve(args.conf), 'out': args.out, 'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads}
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump({'argv': sys.argv, 'cfg': cfg, 'nodes': len(nodes), 'conf_sha1': ab_match.sha1_file(cfg['conf'])}, f, indent=1)
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{len(tasks)} of {len(nodes)} positions to bid, {workers} workers, conf {cfg["conf"]}')
    sys.stdout.flush()
    t0, errors = time.time(), 0
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, r in enumerate(pool.imap_unordered(run_node, tasks, chunksize=1), start=1):
                out.write(json.dumps(r) + '\n')
                out.flush()
                errors += 'error' in r
                if k % 50 == 0 or k == len(tasks):
                    print(f'[{k}/{len(tasks)}] {time.time() - t0:6.0f} s  errors {errors}')
                    sys.stdout.flush()
    return 1 if errors else 0


# ---------------------------------------------------------------- summary

def group_of(node, d):
    """(group, detail) of one position."""
    nat = d['natural']
    offered = [c['call'] for c in nat['offered']]
    if nat['call'] != node['our_call']:
        return '0 not reproduced', ''
    if node['their_call'] in offered:
        if not nat['searched']:
            return '4 other', 'offered, no search'
        res = {c['call']: c for c in nat['result']}
        top = max(nat['result'], key=lambda c: (c['sim'] or -1e9) + (c['adjust'] or 0)) if nat['result'] else None
        if nat['who'] != 'Simulation':
            return '4 other', f"offered and searched, chosen by {nat['who']}"
        if top is not None and top['call'] != nat['call']:
            return '4 other', 'offered and searched, network followed (poor samples)'
        return '3 search chose ours', ''
    if len(offered) == 1:
        return '1 one candidate', ''
    return '2 not a candidate', ''


def summary(nodes_path, diag_path, out_path=None):
    import io
    import numpy as np
    lines = [json.loads(line) for line in open(nodes_path, encoding='utf-8')]
    meta = lines[0]
    nodes = {(n['tournament_id'], n['board']): n for n in lines[1:]}
    diag = {}
    for line in open(diag_path, encoding='utf-8'):
        r = json.loads(line)
        if 'error' not in r:
            diag[(r['tournament_id'], r['board'])] = r
    n_boards = meta['boards']
    buf = io.StringIO()

    def say(text=''):
        print(text)
        buf.write(text + '\n')

    def row(label, rows, width=44):
        d = np.array([r['diff'] for r in rows], dtype=float)
        if len(d) == 0:
            return f'  {label:<{width}} {0:>6}'
        ci = 1.96 * d.std() / len(d) ** 0.5 if len(d) > 1 else float('nan')
        return (f'  {label:<{width}} {len(d):>6} {100 * len(d) / n_boards:>6.1f}% {d.mean():>+8.1f} {ci:>6.1f} '
                f'{d.sum() / n_boards:>+8.2f}  {int((d > 0).sum()):>5} {int((d < 0).sum()):>5}')

    head = f'  {"":<44} {"boards":>6} {"of all":>7} {"mean":>8} {"+-":>6} {"gap":>8}  {"we>":>5} {"we<":>5}'

    say(f"{meta['config']} against {meta['entrant']}: {n_boards} boards, gap {meta['gap_sum'] / n_boards:+.2f} matchpoint points a board")
    say(f"same auction on {meta['same_auction']} boards; the auctions part on {len(nodes)}: "
        f"{sum(n['seat'] == 'S' for n in nodes.values())} at a South call, "
        f"{sum(n['seat'] != 'S' for n in nodes.values())} at a robot's call (the robots called differently at the two tables)")
    say("'mean': our matchpoint percentage minus the entrant's on those boards, with its 95% interval;")
    say("'gap': the group's sum over all compared boards, so the rows add up to the whole gap; 'we>' / 'we<': boards we score better / worse.")

    rows = []
    for key, n in nodes.items():
        if n['seat'] != 'S':
            rows.append({**n, 'group': '5 robots differed', 'detail': ''})
        elif key in diag:
            g, detail = group_of(n, diag[key])
            rows.append({**n, 'group': g, 'detail': detail, 'd': diag[key]})
        else:
            rows.append({**n, 'group': '6 not run', 'detail': ''})

    say('\n1. What the bidder did at the call where we part from the entrant')
    say(head)
    for g in sorted({r['group'] for r in rows}):
        say(row(g, [r for r in rows if r['group'] == g]))
    say(row('all boards where the auctions part', rows))
    details = sorted({r['detail'] for r in rows if r['detail']})
    if details:
        say('  of "other":')
        for dt in details:
            say(row('  ' + dt, [r for r in rows if r['detail'] == dt]))

    # 2. the network's number for the entrant's call
    say("\n2. The network's probability for the entrant's call, where it was not a candidate (groups 1 and 2)")
    say(head)
    miss = [r for r in rows if r['group'][0] in '12']
    for lo, hi in [(0, 0.001), (0.001, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]:
        say(row(f'{lo:g} to {hi:g}', [r for r in miss if r['d'].get('their_p') is not None and lo <= r['d']['their_p'] < hi]))
    say(row('all', miss))
    say("  the network's probability for the call we made, same boards:")
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        def ours_p(r):
            return next((c['p'] for c in r['d']['natural']['offered'] if c['call'] == r['our_call']), 0.0)
        say(row(f'  {lo:g} to {hi:g}', [r for r in miss if lo <= ours_p(r) < hi]))

    # 3. forced search
    say("\n3. The same positions bid again with the entrant's call added to the candidates (groups 1 and 2)")
    say("   'sim': the search's own value of a call (IMPs, or matchpoints from -1 to 1, by the event's scoring) before BEN's adjustments;")
    say("   'adj': after them (the bonus for the network's favorite and the penalties for unlikely calls).")
    say(head)
    forced = [r for r in miss if 'forced' in r['d']]

    def pair(r):
        res = {c['call']: c for c in r['d']['forced']['result']}
        return res.get(r['our_call']), res.get(r['their_call'])

    order = ['the bidder now makes: theirs', 'the bidder now makes: ours', 'the bidder now makes: a third call',
             'sim and adj both have theirs higher', 'sim has theirs higher, adj ours', 'sim has ours higher or equal',
             'theirs now chosen; we had one candidate', 'theirs now chosen; we had several',
             'no search even so (no samples)']
    cats = defaultdict(list)
    for r in forced:
        f = r['d']['forced']
        o, t = pair(r)
        if not f['searched'] or o is None or t is None or o['sim'] is None or t['sim'] is None:
            cats[order[8]].append(r)
            continue
        pick = 'theirs' if f['call'] == r['their_call'] else 'ours' if f['call'] == r['our_call'] else 'a third call'
        cats[f'the bidder now makes: {pick}'].append(r)
        if pick == 'theirs':
            cats[order[6] if r['group'][0] == '1' else order[7]].append(r)
        ta, oa = t['sim'] + (t['adjust'] or 0), o['sim'] + (o['adjust'] or 0)
        cats[order[5] if t['sim'] <= o['sim'] else order[3] if ta > oa else order[4]].append(r)
    for label in order:
        if cats[label]:
            say(row(label, cats[label]))

    # 4. group 3 margins
    say("\n4. Where the entrant's call was searched and ours chosen (group 3): the search's margin for our call")
    say(head)
    g3 = [r for r in rows if r['group'][0] == '3']

    def margin(r, adjusted):
        res = {c['call']: c for c in r['d']['natural']['result']}
        o, t = res.get(r['our_call']), res.get(r['their_call'])
        if o is None or t is None or o['sim'] is None or t['sim'] is None:
            return None
        return (o['sim'] + (o['adjust'] or 0) - t['sim'] - (t['adjust'] or 0)) if adjusted else o['sim'] - t['sim']

    say(row('sim alone had theirs higher (adjustment decided)', [r for r in g3 if margin(r, False) is not None and margin(r, False) < 0]))
    say(row('sim alone had ours higher or equal', [r for r in g3 if margin(r, False) is not None and margin(r, False) >= 0]))
    for sc, cuts in (('IMP', [0, 0.5, 1, 2, 4, 99]), ('MP', [0, 0.1, 0.25, 0.5, 2.01])):
        say(f'  events scored at {sc}, margin after adjustments:')
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            say(row(f'  {lo:g} to {hi:g}', [r for r in g3 if r['scoring'] == sc and margin(r, True) is not None and lo <= margin(r, True) < hi]))

    # 5. groups by stage and direction
    for title, keyf in (("5. Groups by the stage of the call", lambda r: r['stage']),
                        ("6. Groups by direction (our call, the entrant's call)", lambda r: r['direction'])):
        say(f'\n{title}')
        say(head)
        for g in sorted({r['group'] for r in rows if r['group'][0] in '123'}):
            say(f'  {g}')
            sub = [r for r in rows if r['group'] == g]
            for k in sorted({keyf(r) for r in sub}):
                part = [r for r in sub if keyf(r) == k]
                if len(part) >= 8:
                    say(row('  ' + k, part))

    if out_path:
        with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(buf.getvalue())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        return export(*sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        return summary(*sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=['run'])
    parser.add_argument('--conf', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dds-threads', type=int, default=3)
    parser.add_argument('--tf-threads', type=int, default=2)
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
