#!/usr/bin/env python3
"""A/B team match between two BEN configurations, scored in IMPs.

Every board is played at two tables: table 1 has config A as North-South and
config B as East-West, table 2 has them swapped. The IMP result of a board is
A's score at table 1 against B's score in the same seats at table 2.

Two modes:
  bidding  auction, opening lead, then the double dummy result (the existing
           scripts/match/bidding pieces: auction.py, lead.py, score.py)
  play     auction and all 13 tricks played by the engine (src/game.py Driver)

See README.md in this folder for usage.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BEN_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(BEN_ROOT, 'src')
BIDDING_SCRIPTS = os.path.join(HERE, 'bidding')

# src first: "bidding" must resolve to src/bidding, not to scripts/match/bidding
for p in (BIDDING_SCRIPTS, SRC):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

SEATS = ['north', 'east', 'south', 'west']
VULN_PBN = {'None': 'None', 'NS': 'N-S', 'EW': 'E-W', 'All': 'Both', 'Both': 'Both',
            'N-S': 'N-S', 'E-W': 'E-W', 'Love': 'None', '-': 'None'}


# ---------------------------------------------------------------- board sets

def load_boards(path):
    """Read a board set: the scripts/match/bidding .txt format or a PBN file."""
    boards = []
    if path.lower().endswith('.pbn'):
        from pbn2ben import load
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines() + ['\n']
        pbn_boards, _ = load(lines)
        for b in pbn_boards:
            dealer, vuln = b['auction'].split()[:2]
            boards.append((dealer, VULN_PBN[vuln], b['deal'].split()))
    else:
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                boards.append((parts[0], VULN_PBN[parts[1]], parts[2:6]))
    return boards


def generate_boards(n, seed):
    """Random deals, dealer and vulnerability rotating as on boards 1-16."""
    import numpy as np
    from create_set import generate
    np.random.seed(seed)
    return [(dealer, vuln, deal.split()) for dealer, vuln, deal in generate(n)]


def check_board(i, dealer, vuln, hands):
    ok = dealer in 'NESW' and len(hands) == 4
    cards = set()
    for h_i, hand in enumerate(hands if ok else []):
        suits = hand.split('.')
        ok = ok and len(suits) == 4 and sum(len(s) for s in suits) == 13
        for s_i, suit in enumerate(suits):
            cards.update((s_i, c) for c in suit)
    if not ok or len(cards) != 52:
        raise SystemExit(f"Board {i + 1} is not a complete deal: {dealer} {vuln} {' '.join(hands)}")


# ------------------------------------------------------------------- workers

class Side:
    def __init__(self, label, conf, models):
        self.label = label
        self.conf = conf
        self.models = models


W = {}  # per worker process state


def load_conf(conf_path, opponent_path):
    import conf
    configuration = conf.load(conf_path)
    if opponent_path:
        # the same override game.py --opponent applies
        if not configuration.read(opponent_path):
            raise FileNotFoundError(opponent_path)
    return configuration


def platform_fixups(models):
    """What game.py main() does after loading: switch off engines this platform lacks."""
    if sys.platform != 'win32' and (getattr(models, 'ace_use_declaring', False) or getattr(models, 'ace_use_defending', False)):
        models.ace_use_declaring = False
        models.ace_use_defending = False
    if models.pimc_use_declaring or models.pimc_use_defending:
        from pimc.PIMC import BGADLL
        if BGADLL.get_dll() is None:
            models.pimc_use_declaring = False
            models.pimc_use_defending = False


def worker_init(cfg):
    if not cfg['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    if cfg['tf_threads']:
        # TensorFlow otherwise starts a thread per core in every worker
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
        os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.chdir(SRC)
    # everything the engine prints (python and native) goes to a per worker log
    log = open(os.path.join(cfg['out'], 'logs', f'worker-{os.getpid()}.log'), 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    import numpy as np
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    from nn.models_tf2 import Models
    from ddsolver.ddsolver import DDSolver

    conf_a = load_conf(cfg['a'], cfg['opp_a'])
    models_a = Models.from_conf(conf_a, BEN_ROOT)
    platform_fixups(models_a)
    if cfg['same']:
        conf_b, models_b = conf_a, models_a
    else:
        conf_b = load_conf(cfg['b'], cfg['opp_b'])
        models_b = Models.from_conf(conf_b, BEN_ROOT)
        platform_fixups(models_b)
    if cfg['matchpoint'] is not None:
        models_a.matchpoint = cfg['matchpoint']
        models_b.matchpoint = cfg['matchpoint']

    W['cfg'] = cfg
    W['A'] = Side('A', conf_a, models_a)
    W['B'] = Side('B', conf_b, models_b)
    # one solver context (with its own memory) per thread: the configs' default of
    # a thread per core costs about 6 GB per worker on a 128 thread machine
    W['dds'] = DDSolver(max_threads=cfg['dds_threads'])
    if cfg['mode'] == 'bidding':
        import auction as auction_script
        import lead as lead_script
        import score as score_script
        W['auction'], W['lead'], W['score'] = auction_script, lead_script, score_script
    else:
        import game
        W['game'] = game
    print(f"worker {os.getpid()} ready: A={cfg['a']} B={cfg['b']} mode={cfg['mode']}")


def base_record(board):
    rec = {'board': board['board'], 'dealer': board['dealer'], 'vuln': board['vuln']}
    for seat, hand in zip(SEATS, board['hands']):
        rec[seat] = hand
    return rec


def play_bidding(board, ns, ew):
    """Auction, opening lead and double dummy score, with the scripts in bidding/."""
    import numpy as np
    from sample import Sample
    from bidding import bidding
    np.random.seed(42)  # as auction.py does for every board
    rec = base_record(board)
    samplers = [Sample.from_conf(ns.conf), Sample.from_conf(ew.conf)]
    auction, _ = W['auction'].bid_hand(board['hands'], board['dealer'], board['vuln'],
                                       [ns.models, ew.models], samplers, W['dds'], False)
    rec['auction'] = auction
    rec['contract'] = bidding.get_contract(auction)
    rec['lead'] = None
    if rec['contract'] is not None:
        # the opening lead is made by the side that is on lead, with its own config
        leader = [ns, ew][('NESW'.index(rec['contract'][-1]) + 1) % 2]
        rec['lead'] = W['lead'].lead(rec, leader.models, Sample.from_conf(leader.conf), W['dds'], False)
    tricks, score = W['score'].get_dd_score(rec)
    rec['tricks'] = int(tricks)
    rec['score'] = int(score)
    return rec, None


def play_full(board, ns, ew):
    """Auction and card play by the engine, through game.Driver."""
    import human
    import scoring
    from sample import Sample
    game = W['game']
    rec = base_record(board)
    driver = game.Driver(ns.models, human.ConsoleFactory(), Sample.from_conf(ns.conf), None, W['dds'], False,
                         models_ew=ew.models, sampler_ew=Sample.from_conf(ew.conf))
    driver.human = [False, False, False, False]
    driver.facit = False
    driver.set_deal(board['board'], ' '.join(board['hands']), f"{board['dealer']} {board['vuln']}",
                    play_only=False, bidding_only="False")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(driver.run(time.time()))
    finally:
        loop.close()
    driver.parscore = W['dds'].calculatepar(driver.deal_str, [driver.vuln_ns, driver.vuln_ew])
    rec['auction'] = driver.auction
    rec['contract'] = driver.contract
    rec['lead'] = None
    rec['tricks'] = 0
    rec['score'] = 0
    if driver.contract is not None:
        if driver.card_play is None:
            raise RuntimeError("the play did not finish")
        declarer_ns = driver.contract[-1] in 'NS'
        score = scoring.score(driver.contract, driver.vuln_ns if declarer_ns else driver.vuln_ew, driver.tricks_taken)
        rec['lead'] = driver.card_responses[0].card.symbol()
        rec['tricks'] = int(driver.tricks_taken)
        rec['score'] = int(score if declarer_ns else -score)
    rec['par'] = int(driver.parscore)
    return rec, driver.asPBN()


def run_task(task):
    board, table = task
    t0 = time.time()
    out = {'board': board['board'], 'table': table}
    try:
        ns, ew = (W['A'], W['B']) if table == 1 else (W['B'], W['A'])
        print(f"=== board {board['board']} table {table} NS={ns.label} EW={ew.label}")
        fn = play_bidding if W['cfg']['mode'] == 'bidding' else play_full
        out['record'], out['pbn'] = fn(board, ns, ew)
    except Exception:
        out['error'] = traceback.format_exc()
        print(out['error'])
    out['seconds'] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------- results

def t_critical(df, z=1.959964):
    """Two sided 95% Student t quantile (Cornish-Fisher expansion, good from df 3)."""
    if df <= 0:
        return float('nan')
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def summarize(compared, k_swings):
    imps = [c['imp'] for c in compared]
    n = len(imps)
    s = {'boards': n}
    if n == 0:
        return s
    mean = sum(imps) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in imps) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    half = t_critical(n - 1) * se if n > 1 else float('nan')
    diff_auction = [c['imp'] for c in compared if c['auction'][0] != c['auction'][1]]
    s.update({
        'imps_a': sum(x for x in imps if x > 0),
        'imps_b': -sum(x for x in imps if x < 0),
        'net_a': sum(imps),
        'boards_won_a': sum(1 for x in imps if x > 0),
        'boards_won_b': sum(1 for x in imps if x < 0),
        'boards_pushed': sum(1 for x in imps if x == 0),
        'boards_different_auction': len(diff_auction),
        'net_a_different_auction': sum(diff_auction),
        'imps_per_board': mean,
        'sd_per_board': sd,
        'standard_error': se,
        'ci95': [mean - half, mean + half],
        # boards for a 95% interval of +-1 IMP/board, and to detect a true
        # 1 IMP/board difference (5% two sided, 80% power)
        'boards_for_ci_1imp': math.ceil((1.96 * sd) ** 2) if sd > 0 else None,
        'boards_to_detect_1imp': math.ceil(((1.96 + 0.8416) * sd) ** 2) if sd > 0 else None,
        'swings': [c['board'] for c in sorted(compared, key=lambda c: (-abs(c['imp']), c['board']))[:k_swings] if c['imp'] != 0],
    })
    return s


def contract_text(c, i):
    if c['contract'][i] is None:
        return 'passed out'
    made = c['tricks'][i] - (int(c['contract'][i][0]) + 6)
    result = '=' if made == 0 else f'{made:+d}'
    lead = f" lead {c['lead'][i]}" if c['lead'][i] else ''
    return f"{c['contract'][i]}{result}{lead}"


def summary_text(meta, s, compared, failed):
    name_a, name_b = meta['name_a'], meta['name_b']
    lines = [
        f"A/B match, {meta['mode']} mode" + (" (double dummy score after the opening lead)" if meta['mode'] == 'bidding' else " (auction and card play)"),
        f"  A = {name_a}  ({meta['a']}" + (f", opponents: {meta['opp_a']}" if meta['opp_a'] else '') + ")",
        f"  B = {name_b}  ({meta['b']}" + (f", opponents: {meta['opp_b']}" if meta['opp_b'] else '') + ")",
        f"  boards: {meta['boards_source']}  ({meta['boards_sha1'][:12]})",
        "",
    ]
    n = s['boards']
    if n == 0:
        lines.append("No board finished at both tables.")
    else:
        lines += [
            f"Boards played      {n}" + (f"   ({len(failed)} failed, left out)" if failed else ''),
            f"IMPs               A {s['imps_a']}  -  B {s['imps_b']}   (net {s['net_a']:+d} for A)",
            f"Boards won         A {s['boards_won_a']}, B {s['boards_won_b']}, pushed {s['boards_pushed']}",
            f"IMPs per board     {s['imps_per_board']:+.2f} for A",
        ]
        if n > 1:
            lines += [
                f"  standard error   {s['standard_error']:.2f}",
                f"  95% interval     {s['ci95'][0]:+.2f} to {s['ci95'][1]:+.2f}",
                f"  spread (sd)      {s['sd_per_board']:.2f} IMPs per board",
            ]
        if s.get('boards_to_detect_1imp'):
            lines += [
                f"Boards needed      {s['boards_for_ci_1imp']} for a 95% interval of +-1 IMP/board,",
                f"                   {s['boards_to_detect_1imp']} to detect a real 1 IMP/board difference (80% power)",
            ]
        lines.append(f"Different auctions {s['boards_different_auction']} boards, net {s['net_a_different_auction']:+d} for A")
        if s['swings']:
            lines += ["", "Largest swings (IMPs for A; table 1 = A North-South, table 2 = B North-South; scores are North-South)"]
            by_board = {c['board']: c for c in compared}
            for b in s['swings']:
                c = by_board[b]
                lines.append(f"  board {b:>4} {c['dealer']} {c['vuln']:<5} {c['imp']:+3d}   "
                             f"table 1: {contract_text(c, 0):<18} {c['score'][0]:+6d}   "
                             f"table 2: {contract_text(c, 1):<18} {c['score'][1]:+6d}")
    if failed:
        lines += ["", "Failed (see logs/):"] + [f"  board {b} table {t}" for b, t in failed]
    lines += ["", f"Time: {meta.get('seconds', 0):.0f} s with {meta['workers']} workers"
              + (f", {meta['seconds_per_table']:.1f} s per board per table" if meta.get('seconds_per_table') else '')]
    return '\n'.join(lines) + '\n'


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(['git', '-c', 'safe.directory=*', '-C', BEN_ROOT, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def resolve(path, base=SRC):
    """Config paths are given the way game.py takes them: relative to src."""
    if path is None:
        return None
    for candidate in (path, os.path.join(base, path)):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(f"File not found: {path} (looked in the current folder and in {base})")


def main():
    parser = argparse.ArgumentParser(description="A/B team match between two BEN configurations, scored in IMPs")
    parser.add_argument('--a', required=True, help="config of side A, for example config/default.conf (relative to src)")
    parser.add_argument('--b', required=True, help="config of side B")
    parser.add_argument('--out', required=True, help="output folder (one per run)")
    parser.add_argument('--mode', choices=['bidding', 'play'], default='bidding',
                        help="bidding: auction + lead + double dummy score. play: auction + card play")
    parser.add_argument('--boards', help="board set: PBN file, or the scripts/match/bidding set format (.txt)")
    parser.add_argument('--n', type=int, help="number of boards: how many to deal, or how many to take from --boards")
    parser.add_argument('--seed', type=int, help="seed for dealing the boards when no --boards file is given")
    parser.add_argument('--opp-a', help="opponent description A plays against (a config/opponent/*.conf file, as game.py --opponent)")
    parser.add_argument('--opp-b', help="opponent description B plays against")
    parser.add_argument('--matchpoint', choices=['true', 'false'], help="override the matchpoint setting of both configs (scoring stays IMPs)")
    parser.add_argument('--workers', type=int, default=4, help="parallel processes, each loads both configs")
    parser.add_argument('--dds-threads', type=int, default=16, help="double dummy solver threads per worker (0 = one per core)")
    parser.add_argument('--tf-threads', type=int, default=8, help="TensorFlow threads per worker (0 = TensorFlow's default, one per core)")
    parser.add_argument('--swings', type=int, default=10, help="how many of the largest swings to list")
    parser.add_argument('--resume', action='store_true', help="continue a run that was interrupted in --out")
    parser.add_argument('--gpu', action='store_true', help="let TensorFlow use the GPU (default: CPU, as the BEN servers do)")
    args = parser.parse_args()

    conf_a, conf_b = resolve(args.a), resolve(args.b)
    opp_a, opp_b = resolve(args.opp_a), resolve(args.opp_b)
    if args.boards:
        boards_path = resolve(args.boards, BEN_ROOT)
        raw_boards = load_boards(boards_path)
        if args.n:
            raw_boards = raw_boards[:args.n]
        boards_source = f"{args.boards}, {len(raw_boards)} boards"
    else:
        if not args.n or args.seed is None:
            raise SystemExit("Give a board set with --boards, or --n and --seed to deal one.")
        raw_boards = generate_boards(args.n, args.seed)
        boards_source = f"{args.n} random deals, seed {args.seed}"
    if not raw_boards:
        raise SystemExit("The board set is empty.")
    boards = []
    for i, (dealer, vuln, hands) in enumerate(raw_boards):
        check_board(i, dealer, vuln, hands)
        boards.append({'board': i + 1, 'dealer': dealer, 'vuln': vuln, 'hands': list(hands)})
    set_text = ''.join(f"{b['dealer']} {b['vuln']} {' '.join(b['hands'])}\n" for b in boards)

    import configparser
    names = []
    for path in (conf_a, conf_b):
        cp = configparser.ConfigParser()
        cp.read(path)
        names.append(cp.get('models', 'name', fallback=os.path.basename(path)))

    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    meta = {
        'mode': args.mode, 'a': args.a, 'b': args.b, 'opp_a': args.opp_a, 'opp_b': args.opp_b,
        'name_a': names[0], 'name_b': names[1],
        'conf_a_sha1': sha1_file(conf_a), 'conf_b_sha1': sha1_file(conf_b),
        'opp_a_sha1': sha1_file(opp_a) if opp_a else None, 'opp_b_sha1': sha1_file(opp_b) if opp_b else None,
        'matchpoint': args.matchpoint,
        'boards_source': boards_source, 'boards_sha1': hashlib.sha1(set_text.encode()).hexdigest(),
    }
    identity = json.dumps(meta, sort_keys=True)
    raw_path = os.path.join(out, 'raw.jsonl')
    done = {}
    if os.path.exists(raw_path):
        if not args.resume:
            raise SystemExit(f"{out} already holds a run. Use another --out folder, or --resume to continue it.")
        with open(os.path.join(out, 'run.json')) as f:
            if json.load(f)['identity'] != identity:
                raise SystemExit(f"{out} holds a run with other configs, boards or mode; it cannot be resumed with these.")
        with open(raw_path) as f:
            for line in f:
                r = json.loads(line)
                if 'record' in r:
                    done[(r['board'], r['table'])] = r
    with open(os.path.join(out, 'boards.txt'), 'w', newline='\n') as f:
        f.write(set_text)
    with open(os.path.join(out, 'run.json'), 'w') as f:
        json.dump({'identity': identity, 'started': datetime.datetime.now().isoformat(timespec='seconds'),
                   'git': git_head(), 'command': sys.argv}, f, indent=1)

    cfg = {'a': conf_a, 'b': conf_b, 'opp_a': opp_a, 'opp_b': opp_b, 'out': out, 'mode': args.mode,
           'gpu': args.gpu, 'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads, 'same': conf_a == conf_b and opp_a == opp_b,
           'matchpoint': None if args.matchpoint is None else args.matchpoint == 'true'}
    tasks = [(b, t) for b in boards for t in (1, 2) if (b['board'], t) not in done]
    workers = max(1, min(args.workers, len(tasks)))
    print(f"{args.mode} match: A = {names[0]} ({args.a}), B = {names[1]} ({args.b}), {len(boards)} boards, "
          f"{len(tasks)} of {2 * len(boards)} deals to play, {workers} workers, output in {out}")
    sys.stdout.flush()

    t_start = time.time()
    task_seconds = []
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(raw_path, 'a') as raw:
            for k, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), start=1):
                raw.write(json.dumps(r) + '\n')
                raw.flush()
                task_seconds.append(r['seconds'])
                if 'record' in r:
                    done[(r['board'], r['table'])] = r
                    what = f"{r['record']['contract'] or 'passed out'} {r['record']['score']:+d}"
                else:
                    what = "FAILED: " + r['error'].strip().splitlines()[-1]
                print(f"[{k}/{len(tasks)}] {time.time() - t_start:6.0f} s  board {r['board']} table {r['table']}: {what}")
                sys.stdout.flush()

    # ---- result files: everything below is a pure function of raw.jsonl
    from compare import get_imps
    compared, failed = [], []
    tables = {1: [], 2: []}
    pbns = {1: [], 2: []}
    for b in boards:
        r1, r2 = done.get((b['board'], 1)), done.get((b['board'], 2))
        for t, r in ((1, r1), (2, r2)):
            if r is None:
                failed.append((b['board'], t))
            else:
                tables[t].append(r['record'])
                if r.get('pbn'):
                    pbns[t].append(r['pbn'])
        if r1 is None or r2 is None:
            continue
        rec1, rec2 = r1['record'], r2['record']
        c = {key: rec1[key] for key in ['board', 'dealer', 'vuln'] + SEATS}
        for key in ['auction', 'contract', 'lead', 'tricks', 'score']:
            c[key] = [rec1[key], rec2[key]]
        if args.mode == 'bidding':
            # the names scripts/match/bidding/printmatch*.py read
            c['dd_tricks'], c['dd_score'] = c['tricks'], c['score']
        else:
            c['par'] = rec1['par']
        c['imp'] = get_imps(rec1['score'], rec2['score'])
        compared.append(c)

    for t in (1, 2):
        with open(os.path.join(out, f'table{t}.jsonl'), 'w', newline='\n') as f:
            f.writelines(json.dumps(rec) + '\n' for rec in tables[t])
        if pbns[t]:
            with open(os.path.join(out, f'table{t}.pbn'), 'w', newline='\n') as f:
                ns, ew = (names[0], names[1]) if t == 1 else (names[1], names[0])
                f.write(f'% PBN 2.1\n% A/B match table {t}: North-South {ns}, East-West {ew}\n\n')
                f.writelines(pbns[t])
    with open(os.path.join(out, 'results.jsonl'), 'w', newline='\n') as f:
        f.writelines(json.dumps(c) + '\n' for c in compared)

    meta['workers'] = workers
    meta['seconds'] = round(time.time() - t_start, 1)
    if task_seconds:
        meta['seconds_per_table'] = round(sum(task_seconds) / len(task_seconds), 1)
    s = summarize(compared, args.swings)
    text = summary_text(meta, s, compared, failed)
    with open(os.path.join(out, 'summary.txt'), 'w', newline='\n') as f:
        f.write(text)
    with open(os.path.join(out, 'summary.json'), 'w', newline='\n') as f:
        json.dump({'meta': meta, 'git': git_head(), 'summary': s, 'failed': failed,
                   'results_sha1': sha1_file(os.path.join(out, 'results.jsonl'))}, f, indent=1)
    print()
    print(text)
    print(f"results.jsonl sha1 {sha1_file(os.path.join(out, 'results.jsonl'))[:12]}  (the same configs and boards give the same value)")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
