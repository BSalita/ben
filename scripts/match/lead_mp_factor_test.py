"""
Paired test of the factor that scales the lead rules' suit adjustments at matchpoints
(botopeninglead.py: expected_score_mp + factor * suit_adjust).

The factor 10 was made for the old 0..100 MP scale; calculate.calculate_mp_score now returns -1..+1, so
10 x (+0.5 partner suit, -0.2 trumps, -1 / -0.5 slam Q/J leads) outweighs any simulated difference.

The adjustment only reorders the simulated candidates, so one capture of BEN's lead per board gives the
lead for every factor. The capture is the evidence of /lead details=true (BEN branch lead-explain;
src/play-explain/capture_lead.py): the sampled layouts, every candidate's double-dummy tricks on each
layout, the suit adjustments, the network scores and the rule that picked the lead. This script
recomputes BEN's MP score of each candidate exactly as calculate.calculate_mp_score does (round robin
over the candidates, sign of the score difference on each layout, equal weights, rounded to 4 decimals),
adds factor x suit_adjust and sorts as find_opening_lead does: (score, network score rounded to 2
decimals), after a first sort by network score. Each chosen lead is then scored double dummy on the
true deal (endplay), at matchpoints against the lead the baseline factor chooses.

The layouts and DD tables do not depend on the scoring (an IMP capture and an MP capture of the same
board are identical), so an IMP capture can be replayed at MP. --check verifies the replay against
a capture's own leads (MP capture at factor 10, IMP capture at factor 1 with IMP scoring).

  python lead_mp_factor_test.py --leads imp.jsonl --boards boards_def.jsonl --out <dir> [--check mp.jsonl]

Needs numpy and endplay (dev: src\\bbo-bidding\\.venv).
"""
import argparse
import json
import os
import sys

import numpy as np

FACTORS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 10]
BASELINES = [10, 0, 0.2]
SUITS = 'SHDC'


def diff_to_imps(d):
    t = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000,
         2250, 2500, 3000, 3500, 4000]
    a = abs(d)
    i = sum(1 for x in t if a >= x)
    return i if d >= 0 else -i


class Lead:
    def __init__(self, rec):
        ev = rec['evidence']
        L = ev['lead']
        self.id = rec['id']
        self.card = rec['card']
        self.who = rec.get('who') or ''
        self.dd = {c: np.array(v, int) for c, v in ev['dd'].items()}
        self.sbt = list(ev['score_by_tricks'])
        self.need = int(ev['tricks_needed'])
        self.sym = dict(L['candidate'])
        self.adjust = {s: float(a) for s, a in L['suit_adjust'].items()}
        self.nn = {c: float(L['nn'][c]) for c in self.dd}
        assert not ev.get('weighted'), 'weighted layouts are not replayed'
        self.rule = 'nn' if self.who.startswith('NN') else 'sim'      # 'NN - best', 'NN - bad quality samples'
        self.contract = ev['contract']
        self.n = len(ev['layouts'])
        self.sc = {'MP': self._scores('MP'), 'IMP': self._scores('IMP')}

    def _scores(self, scoring):
        cards = list(self.dd)
        if len(cards) == 1:
            return {cards[0]: 0.0}
        rs = {c: -np.array([self.sbt[13 - t] for t in self.dd[c]]) for c in cards}
        out = {}
        for c in cards:
            acc = 0
            for o in cards:
                if o != c:
                    d = rs[c] - rs[o]
                    acc += int(np.sign(d).sum()) if scoring == 'MP' else sum(diff_to_imps(int(x)) for x in d)
            v = acc / (self.n * (len(cards) - 1))
            out[c] = round(v, 4) if scoring == 'MP' else round(v, 2)
        return out

    def choose(self, factor, scoring='MP'):
        """The solved card BEN leads with this factor."""
        first = sorted(self.dd, key=lambda c: -self.nn[c])
        if self.rule == 'nn':
            return first[0]
        sc = self.sc[scoring]
        key = lambda c: (sc[c] + factor * self.adjust[c[0]], round(self.nn[c], 2))
        return sorted(first, key=key, reverse=True)[0]


def true_dd(boards, ids, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [i for i in ids if i not in cache]
    if todo:
        from endplay.types import Deal, Denom, Player
        from endplay.dds import solve_board
        denom = {'S': Denom.spades, 'H': Denom.hearts, 'D': Denom.diamonds, 'C': Denom.clubs, 'N': Denom.nt}
        rank = {f'R{r}': r for r in '23456789TJQKA'}
        for k, i in enumerate(todo, 1):
            b = boards[i]
            decl = 'NESW'.index(b['declarer'])
            leader = [Player.north, Player.east, Player.south, Player.west][(decl + 1) % 4]
            d = Deal('N:' + b['hands'], first=leader, trump=denom[b['contract'][1]])
            cache[i] = {SUITS['spades hearts diamonds clubs'.split().index(c.suit.name)] + rank[c.rank.name]: int(t)
                        for c, t in solve_board(d)}
            if k % 200 == 0:
                print(f'true DD {k}/{len(todo)}', flush=True)
                json.dump(cache, open(cache_path, 'w'))
        json.dump(cache, open(cache_path, 'w'))
    return cache


def check(path, factor, scoring):
    bad = n = 0
    for line in open(path):
        r = json.loads(line)
        if not r.get('evidence') or not r['evidence'].get('dd'):
            continue
        t = Lead(r)
        n += 1
        # BEN lists its candidates in its final order: the first is its choice (the card led can be another card
        # of the suit when the choice is an 8 or a small card: Morty's carding rule)
        if t.sym[t.choose(factor, scoring)] != r['candidates'][0]['card']:
            bad += 1
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leads', required=True, help='capture_lead.py leads.jsonl with evidence')
    ap.add_argument('--boards', required=True, help='the boards (true hands) of the capture')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', help='an MP capture to verify the replay against (factor 10)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    emit = lambda s='': (lines.append(s), print(s, flush=True))

    if a.check:
        n, bad = check(a.check, 10, 'MP')
        emit(f'Replay check, MP capture {os.path.basename(a.check)} at factor 10: {n - bad} of {n} leads reproduced')
    n, bad = check(a.leads, 1, 'IMP')
    emit(f'Replay check, {os.path.basename(a.leads)} at IMPs (factor 1): {n - bad} of {n} leads reproduced')

    boards = {}
    for line in open(a.boards):
        b = json.loads(line)
        boards[b['id']] = b
    recs = [json.loads(line) for line in open(a.leads)]
    leads = [Lead(r) for r in recs if r.get('evidence') and r['evidence'].get('dd')]
    n_all = len(recs)
    tdd = true_dd(boards, [t.id for t in leads], os.path.join(a.out, 'true_dd.json'))

    def defence(t, c):
        tr = tdd[t.id][c]
        return tr, -t.sbt[13 - tr]

    sim = [t for t in leads if t.rule == 'sim']
    emit(f'{n_all} boards; {len(leads)} with layouts; {len(sim)} decided by the simulation '
         f'({len(leads) - len(sim)} by the network rules, {n_all - len(leads)} without layouts: same lead at every factor)')
    touched = sum(1 for t in sim if len({t.adjust[c[0]] for c in t.dd}) > 1)
    emit(f'Boards where the adjustment differs between candidates: {touched}')
    emit()
    emit('Each factor against a baseline, double dummy on the true deal, per board over all '
         f'{n_all} boards (+ is better for the defence).')
    emit('MP = matchpoints against a table that leads with the baseline factor: 50% + 50% x mean(sign of the score '
         'difference); +- one standard error.')
    res = {}
    for base in BASELINES:
        emit()
        emit(f'Baseline factor {base:g}')
        emit(f"{'factor':>7s} {'changed':>8s} {'better':>7s} {'worse':>6s} {'MP %':>16s} {'tricks/board':>18s} {'IMPs/board':>18s}")
        for f in FACTORS:
            if f == base:
                continue
            sg, dt, di = np.zeros(n_all), np.zeros(n_all), np.zeros(n_all)
            changed = 0
            for i, t in enumerate(leads):
                cb, cf = t.choose(base), t.choose(f)
                if cb == cf:
                    continue
                changed += 1
                (tb, sb), (tf, sf) = defence(t, cb), defence(t, cf)
                sg[i], dt[i], di[i] = np.sign(sf - sb), tf - tb, diff_to_imps(sf - sb)
            se = lambda x: x.std(ddof=1) / np.sqrt(n_all)
            mp_pct, mp_se = 50 + 50 * sg.mean(), 50 * se(sg)
            res[f'{base:g}->{f:g}'] = dict(changed=changed, better=int((sg > 0).sum()), worse=int((sg < 0).sum()),
                                           mp_pct=mp_pct, mp_se=mp_se, tricks=dt.mean(), tricks_se=se(dt),
                                           imps=di.mean(), imps_se=se(di))
            emit(f'{f:7g} {changed:8d} {int((sg > 0).sum()):7d} {int((sg < 0).sum()):6d} '
                 f'{mp_pct:9.2f} +-{mp_se:4.2f} {dt.mean():+10.4f} +-{se(dt):.4f} {di.mean():+10.4f} +-{se(di):.4f}')

    def shares(f, scoring='MP'):
        p_has = p_led = t_has = t_led = best = 0
        for t in sim:
            c = t.choose(f, scoring)
            pos = [s for s, v in t.adjust.items() if v > 0]
            trump = t.contract[1] if t.contract[1] in SUITS else None
            if pos and any(x[0] == pos[0] for x in t.dd):
                p_has += 1
                p_led += c[0] == pos[0]
            if trump and any(x[0] == trump for x in t.dd):
                t_has += 1
                t_led += c[0] == trump
            best += tdd[t.id][c] == max(tdd[t.id][x] for x in t.dd)
        return (f'partner suit {p_led}/{p_has}, trumps {t_led}/{t_has}, '
                f'DD-best candidate {100 * best / len(sim):.1f}%')

    emit()
    emit("Leads by factor: led in partner's suit / in trumps, of the leads where such a candidate exists; "
         "the lead is the DD best candidate on the true deal")
    for f in FACTORS:
        emit(f'  MP factor {f:5g}: {shares(f)}')
    emit(f'  IMPs (weight 1): {shares(1, "IMP")}')
    emit(f'  IMPs (weight 0): {shares(0, "IMP")}')

    emit()
    emit('Fitting the factor to IMPs, where the adjustment is added at weight 1 to a -24..+24 IMP score')
    ratios = []
    for t in sim:
        mp_v, imp_v = np.array(list(t.sc['MP'].values())), np.array(list(t.sc['IMP'].values()))
        if len(mp_v) > 1 and np.ptp(imp_v) > 0:
            ratios.append(np.ptp(mp_v) / np.ptp(imp_v))
    emit(f'  MP score range / IMP score range among the candidates of a board: median {np.median(ratios):.3f} '
         f'(quartiles {np.percentile(ratios, 25):.3f} .. {np.percentile(ratios, 75):.3f}, {len(ratios)} boards)')
    imp_changed = sum(1 for t in sim if t.choose(1, 'IMP') != t.choose(0, 'IMP'))
    emit(f'  leads the adjustment changes: IMPs {imp_changed} of {len(sim)}; MP by factor: ' +
         ', '.join(f'{f:g}: {sum(1 for t in sim if t.choose(f) != t.choose(0))}' for f in FACTORS[1:]))

    led_by_id = {r['id']: r['card'] for r in recs}
    same = differ = 0
    for t in sim:
        c, led = t.choose(1, 'IMP'), led_by_id[t.id]
        if led != c:
            if tdd[t.id][led] == tdd[t.id][c]:
                same += 1
            else:
                differ += 1
    emit()
    emit(f'Carding: in this capture the card led differs from the solved card of the choice on {same + differ} of '
         f'{len(sim)} leads; on the true deal it takes other DD tricks on {differ}. Leads here are scored by the solved card.')
    open(os.path.join(a.out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
