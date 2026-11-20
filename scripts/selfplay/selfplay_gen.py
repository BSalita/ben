#!/usr/bin/env python3
"""Self-play on a simulated table: bid deals, and try the learner's other calls on the same deal.

The learner (the bidding network of --conf) sits South. North, East and West are the network
of --robots, each making the legal call it rates highest (--table self: the learner's own
network in all four seats, and all four seats learn). Three modes:

  explore  The learner makes the call its network rates highest. At each of its turns every
           other candidate call (network output of at least --cand-min-p, at most --cand-max
           calls, plus the first choice of each --also network, plus the calls --inject names) is tried: the auction is
           finished from there by the same players and the contract reached is scored double
           dummy on this deal. So every candidate gets the result it would have had on the
           real cards, which the learner cannot see when it bids. selfplay_build.py turns
           these into training targets.
           With --sample T the learner does not always make its first choice: at a turn
           with other candidates of its own network the call made is drawn among them, each
           with a chance in proportion to its network output to the power 1/T. The deal then
           goes on from the call drawn, so the later turns are positions the first-choice
           line never reaches (the ones a changed network will meet). Every candidate of a
           turn, the call made included, is then scored with everybody making first choices
           from there on, so that the results of one turn compare like with like.
  net      The same table without the tries: only the auction and its result. Two runs on
           the same deals with different learners are compared by selfplay_compare.py.
  search   As net, but South bids with the whole engine of --conf (candidates, sampling,
           rollouts), the way it bids in play. For comparing two full settings.

    python selfplay_gen.py --conf CONF --out DIR --n 100000 --seed 1 [--mode explore]

Deals: --n and --seed (random, the same for the same seed), or --deals FILE with one board a
line (tournament_id, board, deal, dealer, vul, scoring: the format of brill_live.py).

Output: DIR/games.jsonl, one line a deal: hands, calls (P, X, XX, 1C..7N from the dealer),
contract, score (North-South, double dummy), dd (tricks: rows S H D C NT, columns N E S W)
and in explore mode decisions: for each learner turn with more than one candidate, n (calls
made before it), seat and cands [[call, network output, contract, North-South score, source],
...], the call made first (source: made, net, also, or the --inject kind). A run that was stopped continues with --resume.
"""
import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp_common  # noqa: E402
from sp_common import SEATS, VULN, from_ben  # noqa: E402

W = {}
CHUNK = 8
MAX_CALLS = 80


def worker_init(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '1' if cfg['mode'] != 'search' else '2'
    os.chdir(sp_common.SRC)
    log = open(os.path.join(cfg['out'], 'logs', f'worker-{os.getpid()}.log'), 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    from ddsolver.ddsolver import dds3
    W['cfg'] = cfg
    W['dds3'] = dds3
    nets = {}

    def net(path):
        if path not in nets:
            nets[path] = sp_common.Net(path)
        return nets[path]

    W['learner'] = net(cfg['conf'])
    W['robots'] = W['learner'] if cfg['table'] == 'self' else net(cfg['robots'])
    W['also'] = [net(path) for path in cfg['also']]
    W['learner_seats'] = (0, 1, 2, 3) if cfg['table'] == 'self' else (2,)
    if cfg['mode'] == 'search':
        sys.path.insert(0, os.path.join(sp_common.BEN_ROOT, 'scripts', 'match'))
        import ab_match
        from nn.models_tf2 import Models
        from ddsolver.ddsolver import DDSolver
        W['conf'] = ab_match.load_conf(cfg['conf'], None)
        W['models'] = Models.from_conf(W['conf'], sp_common.BEN_ROOT)
        ab_match.platform_fixups(W['models'])
        W['dds'] = DDSolver(max_threads=cfg['dds_threads'])
    print(f"worker {os.getpid()} ready: {cfg['mode']} {cfg['conf']} robots={cfg['robots']} table={cfg['table']}")


def run_chunk(boards):
    try:
        deals = [b['deal'] for b in boards]
        tables = [t['res_table'] for t in W['dds3'].calc_all_tables_pbn(deals, mode=-1, max_threads=1)['tables']]
    except Exception:
        error = traceback.format_exc()
        return [{'tournament_id': b['tournament_id'], 'board': b['board'], 'error': error} for b in boards]
    out = []
    for board, dd in zip(boards, tables):
        t0 = time.time()
        key = {'tournament_id': board['tournament_id'], 'board': board['board']}
        try:
            out.append({**key, **run_deal(board, [list(map(int, row)) for row in dd]),
                        'seconds': round(time.time() - t0, 2)})
        except Exception:
            out.append({**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 2)})
    return out


def run_deal(board, dd):
    from bidding import bidding
    cfg = W['cfg']
    hands = sp_common.hands_nesw(board['deal'])
    dealer_i = SEATS.index(board['dealer'])
    vuln = VULN[board['vul']]
    learner, robots = W['learner'], W['robots']
    seats = W['learner_seats']
    hand_of = {}

    def hand_bin(net, seat):
        if (id(net), seat) not in hand_of:
            hand_of[id(net), seat] = net.hand(hands[seat])
        return hand_of[id(net), seat]

    def ranked(net, auction, limit=None):
        seat = len(auction) % 4
        return sp_common.ranked_legal(net.probs(hand_bin(net, seat), vuln, auction), auction, limit)

    def finish(auction):
        """Everybody makes the call their network rates highest until the auction is over."""
        auction = list(auction)
        while not bidding.auction_over(auction):
            if len(auction) > MAX_CALLS:
                raise ValueError(f'auction does not end: {auction}')
            net = learner if len(auction) % 4 in seats else robots
            auction.append(ranked(net, auction, 1)[0][0])
        return auction

    def search_call(seat, auction):
        import numpy as np
        from botbidder import BotBid
        from sample import Sample
        W['models'].matchpoint = board['scoring'] == 'MP'
        np.random.seed(42)
        bot = BotBid(vuln, hands[seat], W['models'], Sample.from_conf(W['conf']), seat, dealer_i, W['dds'], False, False)
        call = bot.bid(auction).bid
        if not bidding.can_bid(call, auction):
            raise ValueError(f'illegal call {call} after {auction}')
        return call

    auction = ['PAD_START'] * dealer_i
    decisions = []
    sample = cfg.get('sample') or 0
    rng = random.Random(f"{board['tournament_id']}:{board['board']}")
    while not bidding.auction_over(auction):
        if len(auction) > MAX_CALLS:
            raise ValueError(f'auction does not end: {auction}')
        seat = len(auction) % 4
        if seat not in seats:
            call = ranked(robots, auction, 1)[0][0]
        elif cfg['mode'] == 'search':
            call = search_call(seat, auction)
        elif cfg['mode'] == 'net':
            call = ranked(learner, auction, 1)[0][0]
        else:
            order = ranked(learner, auction)
            output = dict(order)
            call = order[0][0]
            others = [(c, 'net') for c, p in order[1:] if p >= cfg['cand_min_p']][:cfg['cand_max'] - 1]
            if sample and others:
                pool = [call] + [c for c, _ in others]
                drawn = rng.choices(pool, weights=[output[c] ** (1 / sample) for c in pool])[0]
                if drawn != call:      # the first choice becomes a candidate; the rest stay in falling output
                    others = [(c, 'net') for c in pool if c != drawn]
                    call = drawn
            extra = [(ranked(net, auction, 1)[0][0], 'also') for net in W['also']]
            extra += sp_common.forced_candidates(cfg['inject'], auction, hands[seat])
            for c, why in extra:
                if c != call and all(c != o for o, _ in others) and len(others) < cfg['cand_limit'] - 1:
                    others.append((c, why))
            if others:
                cands = [[from_ben(call), round(output[call], 4), None, None, 'made']]
                if sample:
                    cands[0][2:4] = sp_common.result_of(finish(auction + [call]), dd, vuln)
                for c, why in others:
                    contract, score = sp_common.result_of(finish(auction + [c]), dd, vuln)
                    cands.append([from_ben(c), round(output[c], 4), contract, score, why])
                decisions.append({'n': len(auction) - dealer_i, 'seat': SEATS[seat], 'cands': cands})
        auction.append(call)

    contract, score = sp_common.result_of(auction, dd, vuln)
    if not sample:      # the call made went on with first choices: its result is the deal's
        for d in decisions:
            d['cands'][0][2:4] = [contract, score]
    rec = {'dealer': board['dealer'], 'vul': board['vul'], 'scoring': board['scoring'], 'hands': hands,
           'calls': [from_ben(c) for c in auction[dealer_i:]], 'contract': contract, 'score': score, 'dd': dd}
    if cfg['mode'] == 'explore':
        rec['decisions'] = decisions
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--conf', required=True, help="the learner's config (relative to src, or absolute)")
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['explore', 'net', 'search'], default='explore')
    parser.add_argument('--table', choices=['brill', 'self'], default='brill',
                        help='brill: learner South, --robots in the other seats. self: the learner in all four')
    parser.add_argument('--robots', default='config/Brill-21GF.conf', help='config whose bidder network plays the robots')
    parser.add_argument('--also', action='append', default=[],
                        help='explore: a config whose network also names a candidate (its first choice); may be repeated')
    parser.add_argument('--cand-min-p', type=float, default=0.03, help='explore: least network output of a candidate')
    parser.add_argument('--cand-max', type=int, default=4, help="explore: most candidates from the learner's network")
    parser.add_argument('--inject', action='append', default=[],
                        help='explore: calls tried whatever the network thinks of them: double, compete, game, all:N '
                             '(see sp_common.forced_candidates); may be repeated')
    parser.add_argument('--cand-limit', type=int, default=12, help='explore: most candidates at a turn, of every source')
    parser.add_argument('--sample', type=float, default=0,
                        help="explore: draw the call made among the network's candidates, output to the power 1/T (0 = first choice)")
    parser.add_argument('--n', type=int, help='number of deals to deal')
    parser.add_argument('--seed', type=int, help='seed of the deals')
    parser.add_argument('--board-cycle', type=int, default=16, help='dealer and vulnerability of boards 1..N (Brill events: 8)')
    parser.add_argument('--deals', help='deal file instead of --n and --seed')
    parser.add_argument('--limit', type=int, default=0, help='only the first N deals of --deals')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument('--tf-threads', type=int, default=0, help='TensorFlow threads a worker (default 1, search mode 2)')
    parser.add_argument('--dds-threads', type=int, default=4, help='search mode: double dummy threads a worker')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.mode == 'search' and args.table == 'self':
        raise SystemExit('search mode is for the brill table; two full settings in all four seats: scripts/match/ab_match.py')
    started = datetime.datetime.now()
    if args.deals:
        boards = [json.loads(line) for line in open(args.deals, encoding='utf-8')]
        boards = [{k: b[k] for k in ('tournament_id', 'board', 'deal', 'dealer', 'vul', 'scoring')} for b in boards]
        if args.limit:
            boards = boards[:args.limit]
        source = f'{args.deals}, {len(boards)} deals'
    elif args.n and args.seed is not None:
        boards = sp_common.deal_boards(args.n, args.seed, args.board_cycle)
        source = f'{args.n} random deals, seed {args.seed}, boards 1-{args.board_cycle}'
    else:
        raise SystemExit('Give --deals FILE, or --n and --seed.')

    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    path = os.path.join(args.out, 'games.jsonl')
    done = set()
    if args.resume and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['tournament_id'], r['board']))
    elif os.path.exists(path):
        raise SystemExit(f'{path} exists; use --resume or another --out')
    todo = [b for b in boards if (b['tournament_id'], b['board']) not in done]
    tasks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]

    cfg = {'conf': sp_common.resolve_conf(args.conf), 'robots': sp_common.resolve_conf(args.robots),
           'also': [sp_common.resolve_conf(p) for p in args.also], 'out': args.out, 'mode': args.mode,
           'table': args.table, 'cand_min_p': args.cand_min_p, 'cand_max': args.cand_max,
           'inject': args.inject, 'cand_limit': args.cand_limit, 'sample': args.sample,
           'tf_threads': args.tf_threads or (2 if args.mode == 'search' else 1), 'dds_threads': args.dds_threads}
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump({'argv': sys.argv, 'cfg': cfg, 'deals': source, 'started': started.isoformat(timespec='seconds'),
                   'learner_network': sp_common.NetSpec(cfg['conf']).bidder_path}, f, indent=1)
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{started:%Y-%m-%d %H:%M:%S} {args.mode}: {len(todo)} of {len(boards)} deals to bid ({source}), '
          f'{workers} workers, learner {cfg["conf"]}, table {args.table}')
    sys.stdout.flush()

    counts = {'deals': 0, 'errors': 0, 'decisions': 0, 'changed': 0}
    t0 = time.time()
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, records in enumerate(pool.imap_unordered(run_chunk, tasks, chunksize=1), start=1):
                for r in records:
                    out.write(json.dumps(r) + '\n')
                    if 'error' in r:
                        counts['errors'] += 1
                        continue
                    counts['deals'] += 1
                    for d in r.get('decisions', ()):
                        counts['decisions'] += 1
                        counts['changed'] += any(c[3] != d['cands'][0][3] for c in d['cands'][1:])
                out.flush()
                if k % 200 == 0 or k == len(tasks):
                    spent = time.time() - t0
                    print(f'[{counts["deals"] + counts["errors"]}/{len(todo)}] {spent:6.0f} s  {counts}  '
                          f'{3600 * (counts["deals"] + counts["errors"]) / max(spent, 1):.0f} deals an hour')
                    sys.stdout.flush()
    ended = datetime.datetime.now()
    print(f'{ended:%Y-%m-%d %H:%M:%S} end, elapsed {ended - started}, {counts}')
    return 1 if counts['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
