"""
Paired test of [adjustments] reward_lead_partner_suit: what bonus for leading partner's suit
makes BEN's full opening-lead choice (network candidates, simulation, bonus) take most tricks?

The bonus only reorders the simulated candidates, so one run of the lead choice per deal gives
the lead for every bonus value: the candidates and their simulated scores are saved and the
choice is replayed with the bonus changed. Each chosen lead is scored double dummy on the true
deal. Only deals where BEN's rule finds a partner suit (bidding.get_partner_suit: partner's
first bid, a suit) and the candidates include both a card of that suit and a card of another
can change; the others are counted and not simulated.

Deals and auctions come from results.jsonl files of ab_match.py (both tables of every board).

  python lead_bonus_test.py --conf config/default.conf --out <dir> <results.jsonl> [...]
      [--workers 16] [--dds-threads 8] [--tf-threads 4] [--limit N]
  python lead_bonus_test.py --out <dir> --summary        (tables from <dir>/leads.jsonl only)

--select nt runs the lead choice on every notrump contract instead and tabulates whether it
finds partner's four-card and longer suits, bid or not, against the network's first choice.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BEN_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(BEN_ROOT, 'src')
sys.path.insert(0, SRC)

VULN = {'None': [False, False], 'N-S': [True, False], 'E-W': [False, True], 'Both': [True, True]}
SEATS = ['north', 'east', 'south', 'west']
BONUSES = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 1000]
W = {}


def worker_init(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(cfg['tf_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.chdir(SRC)
    log = open(os.path.join(cfg['out'], 'logs', f'worker-{os.getpid()}.log'), 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    import conf
    from nn.models_tf2 import Models
    from ddsolver.ddsolver import DDSolver
    sys.path.insert(0, HERE)
    from ab_match import platform_fixups
    configuration = conf.load(cfg['conf'])
    models = Models.from_conf(configuration, BEN_ROOT)
    platform_fixups(models)
    assert models.use_real_imp_or_mp_opening_lead, 'only the IMP / MP lead scoring is replayed'
    W.update(conf=configuration, models=models, dds=DDSolver(max_threads=cfg['dds_threads']))


def run_task(t):
    import scoring
    from bidding import bidding
    from botopeninglead import BotLead
    from sample import Sample
    t0 = time.time()
    models = W['models']
    hands = [t[s] for s in SEATS]
    auction = t['auction']
    contract = bidding.get_contract(auction)
    lead_i = (bidding.get_decl_i(contract) + 1) % 4
    vuln = VULN[t['vuln']]
    psuit = t['psuit']
    rec = {'key': t['key'], 'contract': contract, 'leader': lead_i, 'psuit': psuit,
           'cfg_bonus': models.reward_lead_partner_suit, 'mp': bool(models.matchpoint)}
    try:
        bot = BotLead(vuln, hands[lead_i], models, Sample.from_conf(W['conf']), lead_i,
                      'NESW'.index(t['dealer']), W['dds'], False)
        codes, soft = bot.get_opening_lead_candidates(auction)
        codes = [int(c) for c in codes]
        in_p = [c // 8 == psuit for c in codes]
        every = t.get('select') == 'nt'                   # simulate every lead, keep the network's scores
        if every:
            rec['soft'] = [round(float(x), 4) for x in soft[0]]
        if not every and not any(in_p):
            rec['skip'] = 'no candidate in the suit'
            rec['top'] = codes[0]
        elif not every and all(in_p):
            rec['skip'] = 'every candidate in the suit'
            rec['top'] = codes[0]
        else:
            resp = bot.find_opening_lead(auction, {})
            cands = sorted(resp.candidates, key=lambda c: c.insta_score, reverse=True)
            mult = 10 if models.matchpoint else 1
            out = []
            for c in cands:
                code = int(c.card.code())
                score = c.expected_score_mp if models.matchpoint else c.expected_score_imp
                if score is not None and code // 8 == psuit:
                    score -= mult * models.reward_lead_partner_suit
                out.append([code, c.insta_score, None if score is None else round(score, 4)])
            rec.update(cands=out, who=resp.who, quality=float(resp.quality), n_samples=len(resp.samples),
                       top=int(resp.candidates[0].card.code()))
        # every lead on the true deal: tricks for the defence per 32-card code
        res = W['dds'].solve(bidding.get_strain_i(contract), lead_i, [], ['N:' + ' '.join(hands)], 3)
        t32 = [-1] * 32
        for c, v in res.items():
            b = (c // 13) * 8 + min(c % 13, 7)
            t32[b] = max(t32[b], int(v[0]))
        rec['t32'] = t32
        rec['decl_scores'] = [int(x) for x in scoring.contract_scores_by_trick(contract, tuple(vuln))]
    except Exception as ex:
        import traceback
        traceback.print_exc()
        rec['error'] = repr(ex)
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


def choose(rec, bonus):
    """The 32-card code BEN leads with this bonus (botopeninglead.find_opening_lead replayed)."""
    if 'skip' in rec:
        return rec['top']
    cands = rec['cands']                                  # sorted by network score
    if rec['who'].startswith('NN') or cands[0][2] is None:
        return cands[0][0]
    mult = 10 if rec['mp'] else 1
    key = lambda c: (round(c[2] + (mult * bonus if c[0] // 8 == rec['psuit'] else 0), 4), round(c[1], 2))
    return sorted(cands, key=key, reverse=True)[0][0]


def imps(diff):
    import scoring
    return (1 if diff > 0 else -1) * abs(scoring.diff_to_imps(diff)) if diff else 0


def summary(out_dir):
    import numpy as np
    recs = [json.loads(line) for line in open(os.path.join(out_dir, 'leads.jsonl'))]
    meta = json.load(open(os.path.join(out_dir, 'run.json')))
    errors = [r for r in recs if 'error' in r]
    recs = [r for r in recs if 'error' not in r]
    if meta.get('select') == 'nt':
        return summary_nt(out_dir, recs, errors, meta)
    sim = [r for r in recs if 'skip' not in r]
    lines = []
    emit = lines.append
    emit(f"Config {meta['conf']}; {meta['tables']} tables, {meta['contracts']} with a contract, "
         f"{meta['distinct']} distinct deal and auction pairs")
    emit(f"Partner's first bid was a suit the leader holds: {len(recs) + len(errors)} leads ({len(errors)} failed)")
    for why in ('no candidate in the suit', 'every candidate in the suit'):
        emit(f"  {why}: {sum(1 for r in recs if r.get('skip') == why)} (bonus cannot change the lead)")
    emit(f"  simulated: {len(sim)}; of those the network's lead was taken without the simulation's ranking in "
         f"{sum(1 for r in sim if r['who'].startswith('NN') or r['cands'][0][2] is None)}")
    cfg = recs[0]['cfg_bonus']
    wrong = sum(1 for r in sim if choose(r, cfg) != r['top'])
    emit(f"Replay check: with the configured bonus {cfg} the replayed lead differs from BEN's in {wrong} of {len(sim)}")
    emit(f"Seconds per simulated lead: median {np.median([r['seconds'] for r in sim]):.1f}")

    def score(r, code):
        t = r['t32'][code]
        return t, -r['decl_scores'][13 - t]              # defence tricks, defence score

    n = len(recs)
    emit('')
    emit(f"Each bonus against bonus 0, double dummy on the true deal, over all {n} leads (IMP sign: + is better for the defence)")
    emit(f"{'bonus':>7s} {'changed':>8s} {'to suit':>8s} {'gain':>6s} {'lose':>6s} {'tricks/lead':>20s} {'IMPs/lead':>20s} {'IMPs/changed':>13s}")
    for b in BONUSES[1:]:
        dt, di, changed, to_suit = np.zeros(n), np.zeros(n), 0, 0
        for i, r in enumerate(recs):
            c0, cb = choose(r, 0), choose(r, b)
            if c0 == cb:
                continue
            changed += 1
            to_suit += cb // 8 == r['psuit']
            (t0, s0), (tb, sb) = score(r, c0), score(r, cb)
            dt[i], di[i] = tb - t0, imps(sb - s0)
        name = 'always' if b == 1000 else f'{b:g}'
        emit(f"{name:>7s} {changed:8d} {to_suit:8d} {int((dt > 0).sum()):6d} {int((dt < 0).sum()):6d} "
             f"{dt.mean():+10.4f} +-{dt.std() / np.sqrt(n):.4f} {di.mean():+10.4f} +-{di.std() / np.sqrt(n):.4f} "
             f"{di.sum() / max(changed, 1):+13.3f}")
    emit("   'always': a card of partner's suit whenever one is among the candidates.")

    emit('')
    emit('Leads chosen with bonus 0, by suit and result')
    tot = {}
    for r in recs:
        c0 = choose(r, 0)
        k = 'partner suit' if c0 // 8 == r['psuit'] else 'other suit'
        best = max(r['t32'])
        a = tot.setdefault(k, [0, 0, 0.0])
        a[0] += 1; a[1] += r['t32'][c0] == best; a[2] += best - r['t32'][c0]
    for k, (cnt, ok, cost) in tot.items():
        emit(f"  {k:13s} {cnt:6d} leads, double-dummy best {100 * ok / cnt:.1f}%, mean cost {cost / cnt:.3f} tricks")
    text = '\n'.join(lines)
    open(os.path.join(out_dir, 'summary.txt'), 'w').write(text + '\n')
    print(text)


def summary_nt(out_dir, recs, errors, meta):
    """Notrump contracts: does the full lead choice find partner's 4+ card suits, bid or not?"""
    import numpy as np
    lines = []
    emit = lines.append
    PTS = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    n = len(recs)

    def score(r, code):
        t = r['t32'][code]
        return t, -r['decl_scores'][13 - t]

    def ms(x):
        x = np.asarray(x, dtype=float)
        return f"{x.mean():+.3f} +-{x.std() / np.sqrt(max(len(x), 1)):.3f}" if len(x) else '      n/a'

    emit(f"Config {meta['conf']}; {meta['tables']} tables, {meta['contracts']} with a contract, "
         f"{meta['distinct']} distinct deal and auction pairs")
    emit(f"Notrump contracts: {n + len(errors)} leads ({len(errors)} failed)")
    who = {}
    for r in recs:
        who[r['who']] = who.get(r['who'], 0) + 1
    emit('How BEN chose: ' + ', '.join(f'{k} {v}' for k, v in sorted(who.items(), key=lambda kv: -kv[1])))
    emit(f"Seconds per lead: median {np.median([r['seconds'] for r in recs]):.1f}; "
         f"candidates per lead: mean {np.mean([len(r['cands']) for r in recs]):.2f}")

    net = [r['cands'][0][0] for r in recs]               # the network's first choice
    full = [r['top'] for r in recs]                      # BEN's lead
    best = [max(r['t32']) for r in recs]
    emit('')
    emit('1. Network alone against the full lead choice (double dummy on the true deal)')
    for name, cs in (('network alone', net), ('full choice', full)):
        ok = np.mean([r['t32'][c] == b for r, c, b in zip(recs, cs, best)])
        cost = np.mean([b - r['t32'][c] for r, c, b in zip(recs, cs, best)])
        emit(f"  {name:14s} best lead {100 * ok:.1f}%, mean cost {cost:.3f} tricks")
    dt = [r['t32'][f] - r['t32'][c] for r, c, f in zip(recs, net, full)]
    di = [imps(score(r, f)[1] - score(r, c)[1]) for r, c, f in zip(recs, net, full)]
    emit(f"  full choice differs from the network on {sum(c != f for c, f in zip(net, full))} leads: "
         f"gains a trick or more on {sum(d > 0 for d in dt)}, loses on {sum(d < 0 for d in dt)}; "
         f"tricks/lead {ms(dt)}, IMPs/lead {ms(di)}")

    # one row per lead and suit in which partner holds four or more cards and the leader at least one
    rows = []
    for i, r in enumerate(recs):
        hands, auc = r['key'].split('|')
        hands, auc = hands.split(), auc.split()
        ld, pt = r['leader'], (r['leader'] + 2) % 4
        bid = {s: set() for s in range(4)}               # seat -> suits bid
        for k, c in enumerate(auc):
            if c[0].isdigit() and c[1] in 'SHDC':
                bid[k % 4].add('SHDC'.index(c[1]))
        soft = np.array(r['soft'])
        cand = {c[0]: c for c in r['cands']}
        for s in range(4):
            lh, ph = hands[ld].split('.')[s], hands[pt].split('.')[s]
            if not lh or len(ph) < 4:
                continue
            fav = s * 8 + int(np.argmax(soft[s * 8:s * 8 + 8]))
            if r['t32'][fav] < 0:
                fav = next(c for c in range(s * 8, s * 8 + 8) if r['t32'][c] >= 0)
            in_c = [c for c in cand.values() if c[0] // 8 == s]
            gap = None
            if in_c and cand[full[i]][2] is not None and full[i] // 8 != s:
                gap = max(c[2] for c in in_c) - cand[full[i]][2]
            rows.append(dict(
                i=i, pts=sum(PTS.get(x, 0) for x in ph), plen=len(ph), llen=len(lh),
                p_bid=s in bid[pt], l_bid=s in bid[ld], o_bid=s in bid[(ld + 1) % 4] or s in bid[(ld + 3) % 4],
                net=net[i] // 8 == s, full=full[i] // 8 == s, in_c=bool(in_c), gap=gap,
                d_net=0 if net[i] // 8 == s else r['t32'][fav] - r['t32'][net[i]],
                d_full=0 if full[i] // 8 == s else r['t32'][fav] - r['t32'][full[i]],
                i_full=0 if full[i] // 8 == s else imps(score(r, fav)[1] - score(r, full[i])[1]),
                best_here=max(r['t32'][c] for c in range(s * 8, s * 8 + 8)) == best[i]))

    def table(title, keep):
        emit('')
        emit(title)
        emit(f"  {'points':>7s} {'pairs':>6s} {'net leads':>9s} {'full leads':>10s} {'candidate':>9s} "
             f"{'switch network: tricks':>24s} {'switch full: gain':>18s} {'lose':>6s} {'tricks':>16s} {'IMPs':>16s} {'sim gap':>8s}")
        for name, lo, hi in (('0-2', 0, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7+', 7, 99), ('all', 0, 99)):
            g = [x for x in rows if keep(x) and lo <= x['pts'] <= hi]
            if not g:
                continue
            m = lambda k: 100 * np.mean([x[k] for x in g])
            gaps = [x['gap'] for x in g if x['gap'] is not None]
            emit(f"  {name:>7s} {len(g):6d} {m('net'):8.1f}% {m('full'):9.1f}% {m('in_c'):8.1f}% "
                 f"{ms([x['d_net'] for x in g]):>24s} {100 * np.mean([x['d_full'] > 0 for x in g]):17.1f}% "
                 f"{100 * np.mean([x['d_full'] < 0 for x in g]):5.1f}% {ms([x['d_full'] for x in g]):>16s} "
                 f"{ms([x['i_full'] for x in g]):>16s} {np.mean(gaps) if gaps else float('nan'):+8.2f}")

    emit('')
    emit("Pairs below: a lead and a suit in which partner holds 4+ cards and the leader at least one, by partner's points in the suit.")
    emit("'net leads' / 'full leads': the network alone / BEN leads the suit. 'candidate': a card of the suit was simulated.")
    emit("'switch': lead the network's favorite card of the suit instead, 0 where the suit is led already; over all pairs.")
    emit("'sim gap': simulated IMPs of the suit's best candidate minus BEN's lead, where the suit was simulated and not led.")
    table('2. Partner did not bid the suit, nor did anyone else', lambda x: not (x['p_bid'] or x['l_bid'] or x['o_bid']))
    table('3. Partner did not bid the suit, declarer or dummy did', lambda x: not x['p_bid'] and x['o_bid'])
    table('4. Partner did not bid the suit, the leader did', lambda x: not x['p_bid'] and x['l_bid'] and not x['o_bid'])
    table('5. Partner bid the suit', lambda x: x['p_bid'])

    # where the full choice's lost tricks are
    emit('')
    emit("6. Where the full choice loses tricks")
    cost = np.array([b - r['t32'][c] for r, c, b in zip(recs, full, best)])
    emit(f"  not the best lead on {int((cost > 0).sum())} of {n} leads, {cost.sum()} tricks in all")
    for name, keep in (("the best lead is in partner's 4+ suit nobody bid, 3+ points",
                        lambda x: x['pts'] >= 3 and not (x['p_bid'] or x['l_bid'] or x['o_bid'])),
                       ("the best lead is in partner's 4+ suit partner bid", lambda x: x['p_bid']),
                       ("the best lead is in any 4+ suit of partner's", lambda x: True)):
        hit = {}
        for x in rows:
            if keep(x) and x['best_here'] and cost[x['i']] > 0 and not x['full']:
                hit[x['i']] = hit.get(x['i'], False) or x['in_c']
        emit(f"  {name}: {len(hit)} leads, {int(sum(cost[i] for i in hit))} tricks; "
             f"the suit was among the simulated candidates on {sum(hit.values())}")
    text = '\n'.join(lines)
    open(os.path.join(out_dir, 'summary.txt'), 'w').write(text + '\n')
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--conf', default='config/default.conf')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--dds-threads', type=int, default=8)
    ap.add_argument('--tf-threads', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--select', choices=['partner', 'nt'], default='partner',
                    help="partner: leads where partner's first bid is a suit (bonus test); nt: every notrump contract")
    args = ap.parse_args()
    if args.summary:
        return summary(args.out)

    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    os.chdir(SRC)
    from bidding import bidding
    tasks, seen, tables, contracts = [], set(), 0, 0
    for path in args.inputs:
        for line in open(path):
            o = json.loads(line)
            for k in range(len(o['auction'])):
                tables += 1
                auction = o['auction'][k]
                contract = bidding.get_contract(auction)
                if contract is None:
                    continue
                contracts += 1
                key = ' '.join(o[s] for s in SEATS) + '|' + ' '.join(auction)
                if key in seen:
                    continue
                seen.add(key)
                lead_i = (bidding.get_decl_i(contract) + 1) % 4
                psuit = bidding.get_partner_suit(lead_i, auction)
                no_suit = psuit is None or psuit > 3 or not o[SEATS[lead_i]].split('.')[psuit]
                if args.select == 'nt':
                    if contract[1] != 'N':
                        continue
                    psuit = -1 if no_suit else psuit
                elif no_suit:
                    continue
                tasks.append({**{s: o[s] for s in SEATS}, 'dealer': o['dealer'], 'vuln': o['vuln'],
                              'auction': auction, 'psuit': int(psuit), 'key': key, 'select': args.select})
    if args.limit:
        tasks = tasks[:args.limit]
    json.dump({'conf': args.conf, 'inputs': args.inputs, 'tables': tables, 'contracts': contracts,
               'distinct': len(seen), 'select': args.select}, open(os.path.join(args.out, 'run.json'), 'w'))
    out_path = os.path.join(args.out, 'leads.jsonl')
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)['key'] for line in open(out_path)}
    tasks = [t for t in tasks if t['key'] not in done]
    print(f'{tables} tables, {contracts} contracts, {len(seen)} distinct; {len(done)} leads done, {len(tasks)} to run', flush=True)

    cfg = {'conf': args.conf, 'out': args.out, 'dds_threads': args.dds_threads, 'tf_threads': args.tf_threads}
    t0 = time.time()
    with mp.get_context('spawn').Pool(args.workers, initializer=worker_init, initargs=(cfg,)) as pool, \
            open(out_path, 'a') as f:
        for k, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), start=1):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if k % 100 == 0:
                print(f'{k}/{len(tasks)} leads, {time.time() - t0:.0f} s', flush=True)
    summary(args.out)


if __name__ == '__main__':
    main()
