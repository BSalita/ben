"""
Would leading partner's suit improve the lead network's opening leads?

On the rows lead_train.py held out (the last test_fraction of each data folder) every card the
leader holds is solved double dummy on the true deal. The lead network's top choice is compared
with the card the network likes best in each other suit, so "switch to this suit" is scored both
ways: tricks gained and tricks lost. Each (board, suit the leader holds) pair is one row of
<out_dir>/<label>_pairs.csv; the tables printed (and written to <out_dir>/<label>_tables.txt) are
cuts of those rows by what partner holds in the suit and by what the auction showed.

Usage (from scripts/training/opening lead):
  python lead_partner_suit.py <out_dir> <set> [<set> ...] -- <suit model.keras> <nt model.keras> [key=value]
  set: <label>=<data_dir>:<source.txt>[,<source.txt>...]   (as for lead_eval.py)
  options: test_fraction=0.05 max_rows=0 (0 = all held-out rows) workers=48
"""
import csv
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEATS = 'NESW'
RANKS = 'AKQJT98765432'
BID = {'PP': 'PASS', 'DD': 'X', 'RR': 'XX'}
HCP = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}

_dds = None


def _init_worker():
    global _dds
    from ddsolver.ddsolver import DDSolver
    _dds = DDSolver()


def _solve(job):
    """Tricks for the leader's side for each of the 52 cards (-1: not a card of the leader)."""
    strain, leader, pbn = job
    out = np.full(52, -1, dtype=np.int8)
    res = _dds.solve(strain, leader, [], [pbn], 3)
    if res:
        for c, v in res.items():
            out[c] = v[0]
    return out


def is_bid(call):
    return len(call) == 2 and call[0] in '1234567'


def read_sources(paths):
    out = []
    for p in paths:
        lines = open(p).read().split('\n')
        out.append([lines[i:i + 4] for i in range(0, len(lines) - 3, 4)])
    return out


def parse(rec):
    """The deal, the leader and what the auction showed about each suit (S, H, D, C)."""
    from bidding import bidding
    deal, meta, auction_str, play = rec
    dealer_c = meta.split()[0]
    hands = dict(zip('WNES', deal[2:].split()))
    nesw = [hands[s] for s in SEATS]
    auction = ['PAD_START'] * SEATS.index(dealer_c) + [BID.get(b, b) for b in auction_str.split()]
    contract = bidding.get_contract(auction)
    strain = bidding.get_strain_i(contract)          # 0 = notrump, 1..4 = S, H, D, C
    lead_i = (bidding.get_decl_i(contract) + 1) % 4
    partner = (lead_i + 2) % 4
    p_bid, p_dbl, l_bid, o_bid = [0] * 4, [0] * 4, [0] * 4, [0] * 4
    p_first = -1                                     # BEN's rule: partner's first bid (4 = notrump)
    for i, call in enumerate(auction):
        seat = i % 4
        if is_bid(call):
            s = 'SHDCN'.index(call[1])
            if seat == partner:
                if p_first == -1:
                    p_first = s
                if s < 4:
                    p_bid[s] = 1
            elif s < 4:
                (l_bid if seat == lead_i else o_bid)[s] = 1
        elif call == 'X' and seat == partner and i >= 1 and is_bid(auction[i - 1]) and auction[i - 1][1] != 'N':
            # A double that asks for the lead: of a suit bid that is not the final strain and
            # looks artificial (an answer to partner's notrump bid, or at the four level or above)
            prev = auction[i - 1]
            s = 'SHDC'.index(prev[1])
            over_nt = i >= 3 and is_bid(auction[i - 3]) and auction[i - 3][1] == 'N'
            if s != strain - 1 and (over_nt or int(prev[0]) >= 4):
                p_dbl[s] = 1
    return dict(pbn='N:' + ' '.join(nesw), strain=strain, lead_i=lead_i, level=int(contract[0]),
                leader=nesw[lead_i].split('.'), partner=nesw[partner].split('.'),
                p_bid=p_bid, p_dbl=p_dbl, l_bid=l_bid, o_bid=o_bid, p_first=p_first,
                lead_card='SHDC'.index(play[0].upper()) * 13 + RANKS.index(play[1].upper()))


def hand32(suits):
    x = np.zeros(32, dtype=np.float16)
    for s, cards in enumerate(suits):
        for c in cards:
            x[s * 8 + min(RANKS.index(c), 7)] += 1
    return x


FIELDS = ['row', 'level', 'strain', 'suit', 'trump', 'l_len', 'p_len', 'p_hcp', 'p_bid', 'p_first', 'p_dbl',
          'l_bid', 'o_bid', 'model_led', 'actual_led', 't_best', 't_model', 't_suit', 't_actual']


def build_pairs(label, data_dir, sources, model, test_fraction, max_rows, pool, out_dir):
    X = np.load(os.path.join(data_dir, 'X.npy')); B = np.load(os.path.join(data_dir, 'B.npy'))
    src = np.load(os.path.join(data_dir, 'src.npy'))
    start = len(X) - int(len(X) * test_fraction)
    X, B, src = X[start:], B[start:], src[start:]
    if max_rows:
        X, B, src = X[:max_rows], B[:max_rows], src[:max_rows]
    n = len(X)
    recs = read_sources(sources)
    rows = [parse(recs[s_i][r_i]) for s_i, r_i in src]
    del recs
    bad = sum(1 for i, r in enumerate(rows) if not np.array_equal(hand32(r['leader']), X[i, 10:42]))
    print(f'{label}: {n} held-out rows; leader hand differs from the data row in {bad}', flush=True)

    cache = os.path.join(data_dir, f'dd_tricks_{n}.npy')
    if os.path.exists(cache):
        tricks = np.load(cache)
    else:
        jobs = [(r['strain'], r['lead_i'], r['pbn']) for r in rows]
        tricks = np.array(pool.map(_solve, jobs, chunksize=64), dtype=np.int8)
        np.save(cache, tricks)
    print(f'{label}: double dummy done', flush=True)

    # Tricks per 32-card bucket (the best card of the bucket, as lead_eval.py does)
    t32 = np.full((n, 32), -1, dtype=np.int8)
    for c in range(52):
        b = (c // 13) * 8 + min(c % 13, 7)
        t32[:, b] = np.maximum(t32[:, b], tricks[:, c])
    p = np.asarray(model.predict([X.astype(np.float16), B.astype(np.float16)], batch_size=4096, verbose=0))
    held = X[:, 10:42] > 0
    q = np.where(held, p, 0)
    top = q.argmax(1)
    best = t32.max(1)
    t_model = t32[np.arange(n), top]
    top_safe8000 = (t_model[:8000] == best[:8000]).mean()

    out = []
    for i, r in enumerate(rows):
        t_actual = tricks[i, r['lead_card']]
        for s in range(4):
            if not r['leader'][s]:
                continue
            pref = s * 8 + q[i, s * 8:s * 8 + 8].argmax()
            ps = r['partner'][s]
            out.append((i, r['level'], r['strain'], s, int(s == r['strain'] - 1), len(r['leader'][s]), len(ps),
                        sum(HCP.get(c, 0) for c in ps), r['p_bid'][s], int(r['p_first'] == s), r['p_dbl'][s],
                        r['l_bid'][s], r['o_bid'][s], int(top[i] // 8 == s), int(r['lead_card'] // 13 == s),
                        best[i], t_model[i], t32[i, pref], t_actual))
    out = np.array(out, dtype=np.int32)
    with open(os.path.join(out_dir, f'{label}_pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(out.tolist())
    return n, out, top_safe8000


def tables(label, n, a, top_safe8000, emit):
    c = {k: a[:, j] for j, k in enumerate(FIELDS)}
    d = c['t_suit'] - c['t_model']                       # tricks gained by switching to this suit
    led = c['model_led'] == 1
    side = c['trump'] == 0                               # not the trump suit
    cost = c['t_best'] - c['t_model']
    first = np.unique(c['row'], return_index=True)[1]    # one pair per board
    bad_rows = set(c['row'][first][cost[first] > 0].tolist())
    n_bad = len(bad_rows)
    is_bad = cost > 0
    act_cost = (c['t_best'] - c['t_actual'])[first]

    emit(f'\n===== {label}: {n} boards =====')
    emit(f'Network lead not double-dummy best: {n_bad} boards = {100 * n_bad / n:.1f}%; '
         f'mean cost {cost[first].mean():.3f} tricks a board (on the first 8,000 rows best = {top_safe8000:.3f}, '
         f'the DDOLAR of lead_eval.py)')
    emit(f'Lead actually made not best: {100 * (act_cost > 0).mean():.1f}%; mean cost {act_cost.mean():.3f} tricks')

    emit('\nA. Of the non-optimal leads: partner holds a side suit of 4+ cards and its lead takes more tricks')
    emit(f"{'points in the suit':20s} {'boards':>8s} {'% of non-optimal':>17s} {'lead better':>12s} {'% of non-optimal':>17s}")
    hb = np.minimum(c['p_hcp'], 9)
    long4 = side & (c['p_len'] >= 4) & is_bad
    cuts = [('0-2', hb <= 2)] + [(str(h) if h < 9 else '9+', hb == h) for h in range(3, 10)] + \
           [('any', hb >= 0), ('3 or more', hb >= 3)]
    for name, m in cuts:
        have = len(set(c['row'][long4 & m].tolist()))
        better = len(set(c['row'][long4 & m & (d > 0)].tolist()))
        emit(f'{name:20s} {have:8d} {100 * have / max(n_bad, 1):16.1f}% {better:12d} {100 * better / max(n_bad, 1):16.1f}%')

    def line(name, m):
        k = int(m.sum())
        if k == 0:
            emit(f'{name:34s} {0:8d}')
            return
        nl = m & ~led
        dn = d[nl]
        allp = d[m]                                      # 0 where the network already leads the suit
        se = allp.std() / np.sqrt(k)
        emit(f'{name:34s} {k:8d} {100 * led[m].mean():6.1f}% {100 * (dn > 0).mean():7.1f}% {100 * (dn < 0).mean():7.1f}% '
             f'{dn.mean():+8.3f} {allp.mean():+8.3f} +-{se:.3f}')

    head = f"{'':34s} {'pairs':>8s} {'led':>7s} {'gains':>8s} {'loses':>8s} {'net':>8s} {'net, all':>8s}"
    note = ('   led: the network already leads the suit. gains / loses / net: when it does not, share of boards\n'
            '   where switching to the suit takes more / fewer tricks, and mean tricks. net, all: mean tricks over\n'
            '   every pair if the suit were always led (+- one standard error).')

    emit('\nB. Both directions, all boards: partner holds 4+ cards in a side suit, by points in the suit')
    emit(head)
    for name, m in cuts[:-2]:
        line(f'{name} points, partner bid it', side & (c['p_len'] >= 4) & m & (c['p_bid'] == 1))
    for name, m in cuts[:-2]:
        line(f'{name} points, not bid', side & (c['p_len'] >= 4) & m & (c['p_bid'] == 0))
    emit(note)

    emit('\nC. By partner\'s length in a side suit')
    emit(head)
    for name, m in [('0-2 cards', c['p_len'] <= 2), ('3 cards', c['p_len'] == 3), ('4 cards', c['p_len'] == 4),
                    ('5 cards', c['p_len'] == 5), ('6+ cards', c['p_len'] >= 6)]:
        line(f'{name}, partner bid it', side & m & (c['p_bid'] == 1))
        line(f'{name}, not bid', side & m & (c['p_bid'] == 0))

    emit('\nD. What the leader can know (side suits, whatever partner holds)')
    emit(head)
    line('partner bid the suit', side & (c['p_bid'] == 1))
    line("  it was partner's first bid (BEN)", side & (c['p_first'] == 1))
    line('  bid, and opponents bid it too', side & (c['p_bid'] == 1) & (c['o_bid'] == 1))
    line('  bid, opponents did not', side & (c['p_bid'] == 1) & (c['o_bid'] == 0))
    line('  bid, leader holds 1 card', side & (c['p_bid'] == 1) & (c['l_len'] == 1))
    line('  bid, leader holds 2 cards', side & (c['p_bid'] == 1) & (c['l_len'] == 2))
    line('  bid, leader holds 3+ cards', side & (c['p_bid'] == 1) & (c['l_len'] >= 3))
    line('partner doubled it for the lead', side & (c['p_dbl'] == 1))
    line('partner did not bid or double it', side & (c['p_bid'] == 0) & (c['p_dbl'] == 0))
    line('  and nobody bid it', side & (c['p_bid'] == 0) & (c['p_dbl'] == 0) & (c['l_bid'] == 0) & (c['o_bid'] == 0))


def main():
    argv = sys.argv[1:]
    sep = argv.index('--')
    out_dir, sets, rest = argv[0], argv[1:sep], argv[sep + 1:]
    model_paths = [a for a in rest if '=' not in a]
    opts = dict(a.split('=', 1) for a in rest if '=' in a)
    test_fraction = float(opts.get('test_fraction', 0.05))
    max_rows = int(opts.get('max_rows', 0))
    workers = int(opts.get('workers', 48))
    os.makedirs(out_dir, exist_ok=True)

    # The double-dummy workers are forked before TensorFlow is loaded
    pool = get_context('fork').Pool(workers, initializer=_init_worker)
    import keras
    suit_model = keras.models.load_model(model_paths[0], compile=False)
    nt_model = keras.models.load_model(model_paths[1], compile=False)

    for s in sets:
        label, spec = s.split('=', 1)
        data_dir, srcs = spec.rsplit(':', 1)
        is_nt = np.load(os.path.join(data_dir, 'X.npy'), mmap_mode='r')[0, 1] == 1
        n, a, top_safe = build_pairs(label, data_dir, srcs.split(','), nt_model if is_nt else suit_model,
                                     test_fraction, max_rows, pool, out_dir)
        with open(os.path.join(out_dir, f'{label}_tables.txt'), 'w') as f:
            def emit(text):
                print(text, flush=True)
                f.write(text + '\n')
            tables(label, n, a, top_safe, emit)
    pool.close()


if __name__ == '__main__':
    main()
