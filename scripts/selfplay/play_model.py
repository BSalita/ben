#!/usr/bin/env python3
"""Real card play instead of double dummy: the tricks a contract takes when robots play it.

Explore scores every candidate call by the contract it leads to, at the double dummy number of
tricks. Real play differs: declarers take more than double dummy in some contracts (the
defenders do not see all the cards), fewer in others, and a doubled sacrifice goes down by a
different number than double dummy says. This learns, from played boards, the distribution of
(tricks taken - double dummy tricks) for a contract, and rescores explore's candidates by their
expected score under it.

    python play_model.py fit FIELD.jsonl OUT.json [--types Robot] [--holdout 5]
    python play_model.py check OUT.json FIELD.jsonl            (held-out events: model against double dummy)
    python play_model.py rescore OUT.json GAMES.jsonl OUT_GAMES.jsonl

fit      FIELD is a Brill field file (brill_field.py: one board a line, the deal and every
         table's contract, declarer and tricks). Tables of the given types only (robots play
         their own cards; Brill's robots defend). Every --holdout-th event (by tournament id)
         is left out for `check`. The model is a table of trick-difference counts by
         (strain kind, level kind, doubled, margin), margin = double dummy tricks - tricks
         needed (-4..+3), with a back-off to (doubled, margin) and to margin alone where a
         cell has few boards. Needs endplay (double dummy tables of the deals).
check    on the held-out events: mean absolute error of the North-South score, model's
         expected score against the double dummy score, and the make rate by margin.
rescore  every candidate's score (cands[i][3]) and the game's own score become the expected
         North-South score under the model (rounded to an integer), for selfplay_build.py.
         The double dummy scores are kept as cands[i][5] and dd_score.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

DELTA = range(-3, 4)          # tricks taken - double dummy tricks, clipped
MARGINS = range(-4, 4)        # double dummy tricks - tricks needed, clipped
PRIOR = 20                    # pseudo-boards of the back-off distribution in a cell
STRAINS = 'SHDCN'             # rows of the games' dd tables
SEATS = 'NESW'


def parse_contract(text):
    """'4SXE' -> (4, 'S', 'X', 'E'); None for a pass-out."""
    if not text or text in ('PASS', 'P'):
        return None
    text = text.replace('NT', 'N')
    level, strain, decl = int(text[0]), text[1], text[-1]
    return level, strain, text[2:-1], decl


def keys(level, strain, dbl, margin):
    """The model's cells for a contract, most specific first."""
    m = max(min(margin, MARGINS[-1]), MARGINS[0])
    strain_kind = 'N' if strain == 'N' else 'M' if strain in 'SH' else 'm'
    level_kind = 'slam' if level >= 6 else 'game' if (level >= 3 and strain == 'N') or (level >= 4 and strain in 'SH') \
        or level >= 5 else 'part'
    dbl_kind = dbl or '-'
    return [f'{strain_kind}|{level_kind}|{dbl_kind}|{m}', f'{dbl_kind}|{m}', f'{m}']


def score_ns(level, strain, dbl, decl, tricks, vul):
    """North-South score; vul is None / NS / EW / All (or the games' form: '-', 'N-S', ...)."""
    vul = {'-': 'None', 'N-S': 'NS', 'E-W': 'EW', 'Both': 'All', 'both': 'All'}.get(vul, vul)
    vulnerable = vul == 'All' or (vul == 'NS' and decl in 'NS') or (vul == 'EW' and decl in 'EW')
    over = tricks - level - 6
    if over < 0:
        down = -over
        if not dbl:
            points = -down * (100 if vulnerable else 50)
        elif vulnerable:
            points = -(200 + 300 * (down - 1))
        else:
            points = -(100 + 200 * min(down - 1, 2) + 300 * max(down - 3, 0))
        if dbl == 'XX':
            points *= 2
    else:
        per = 30 if strain in 'SHN' else 20
        trick_points = per * level + (10 if strain == 'N' else 0)
        mult = {'': 1, 'X': 2, 'XX': 4}[dbl]
        trick_points *= mult
        points = trick_points
        points += (500 if vulnerable else 300) if trick_points >= 100 else 50
        if level == 6:
            points += 750 if vulnerable else 500
        if level == 7:
            points += 1500 if vulnerable else 1000
        if dbl:
            points += 50 * (2 if dbl == 'XX' else 1)
            points += over * (200 if vulnerable else 100) * (2 if dbl == 'XX' else 1)
        else:
            points += over * (30 if strain in 'SHN' else 20)
    return points if decl in 'NS' else -points


class Model:
    def __init__(self, data):
        self.counts = {k: Counter({int(d): n for d, n in v.items()}) for k, v in data['counts'].items()}
        self.meta = data.get('meta', {})
        self._dist = {}

    def dist(self, level, strain, dbl, margin):
        """{delta: probability} for a contract with this double dummy margin."""
        key = (level, strain, dbl, max(min(margin, MARGINS[-1]), MARGINS[0]))
        if key in self._dist:
            return self._dist[key]
        cells = keys(level, strain, dbl, margin)
        p = {0: 1.0}
        for k in reversed(cells):          # global margin cell first, then refine
            c = self.counts.get(k)
            if not c:
                continue
            n = sum(c.values())
            p = {d: (c.get(d, 0) + PRIOR * p.get(d, 0)) / (n + PRIOR) for d in DELTA}
        self._dist[key] = p
        return p

    def expected(self, level, strain, dbl, decl, dd_tricks, vul):
        margin = dd_tricks - level - 6
        return sum(q * score_ns(level, strain, dbl, decl, max(0, min(13, dd_tricks + d)), vul)
                   for d, q in self.dist(level, strain, dbl, margin).items() if q > 0)


def held_out(tournament_id, every):
    return every and int(hashlib.md5(tournament_id.encode()).hexdigest(), 16) % every == 0


def played_tables(field, types):
    """(board, table, contract) of every played table of these types."""
    for line in open(field, encoding='utf-8'):
        b = json.loads(line)
        for t in b['tables']:
            if types and t.get('type') not in types:
                continue
            c = parse_contract((t.get('contract') or '') + (t.get('declarer') or ''))
            if c is None or t.get('tricks') is None:
                continue
            yield b, t, c


def dd_table(deal, cache):
    if deal not in cache:
        from endplay.dds import calc_dd_table
        from endplay.types import Deal, Denom, Player
        table = calc_dd_table(Deal(deal))
        cache[deal] = {(s, p): table[getattr(Denom, dn), getattr(Player, pn)]
                       for s, dn in zip(STRAINS, ('spades', 'hearts', 'diamonds', 'clubs', 'nt'))
                       for p, pn in zip(SEATS, ('north', 'east', 'south', 'west'))}
    return cache[deal]


def fit(args):
    counts = defaultdict(Counter)
    cache, stats = {}, Counter()
    for b, t, (level, strain, dbl, decl) in played_tables(args.field, args.types):
        if held_out(b['tournament_id'], args.holdout):
            stats['held-out tables'] += 1
            continue
        dd = dd_table(b['deal'], cache)[strain, decl]
        delta = max(min(t['tricks'] - dd, DELTA[-1]), DELTA[0])
        for k in keys(level, strain, dbl, dd - level - 6):
            counts[k][delta] += 1
        stats['tables'] += 1
        stats['tricks above double dummy'] += t['tricks'] - dd
    meta = {'field': args.field, 'types': args.types, 'holdout': args.holdout, **stats,
            'mean tricks above double dummy': round(stats['tricks above double dummy'] / max(stats['tables'], 1), 3)}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'counts': {k: dict(v) for k, v in counts.items()}}, f, indent=0)
    print(json.dumps(meta))


def check(args):
    model = Model(json.load(open(args.model)))
    types, every = model.meta.get('types'), model.meta.get('holdout')
    cache = {}
    err_dd = err_m = n = 0
    sq_dd = sq_m = bias_dd = bias_m = 0.0
    by_margin = defaultdict(lambda: [0, 0, 0.0])      # tables, made, predicted make rate
    for b, t, (level, strain, dbl, decl) in played_tables(args.field, types):
        if not held_out(b['tournament_id'], every):
            continue
        dd = dd_table(b['deal'], cache)[strain, decl]
        actual = score_ns(level, strain, dbl, decl, t['tricks'], b['vul'])
        e_dd = score_ns(level, strain, dbl, decl, dd, b['vul']) - actual
        e_m = model.expected(level, strain, dbl, decl, dd, b['vul']) - actual
        sign = 1 if decl in 'NS' else -1          # errors from the declarer's side
        err_dd += abs(e_dd)
        err_m += abs(e_m)
        sq_dd += e_dd * e_dd
        sq_m += e_m * e_m
        bias_dd += sign * e_dd
        bias_m += sign * e_m
        n += 1
        margin = max(min(dd - level - 6, MARGINS[-1]), MARGINS[0])
        p = model.dist(level, strain, dbl, dd - level - 6)
        row = by_margin[margin]
        row[0] += 1
        row[1] += t['tricks'] >= level + 6
        row[2] += sum(q for d, q in p.items() if margin + d >= 0)
    print(f'{n} held-out tables: mean |score error| double dummy {err_dd / n:.1f}, model {err_m / n:.1f} '
          f'(the model gives an expectation, not the likeliest result: judge it by the next two)')
    print(f'root mean square error: double dummy {(sq_dd / n) ** 0.5:.1f}, model {(sq_m / n) ** 0.5:.1f}')
    print(f"mean error for the declarer's side: double dummy {bias_dd / n:+.1f}, model {bias_m / n:+.1f}")
    print('margin  tables  made  double dummy says  model says')
    for m in sorted(by_margin):
        t, made, pred = by_margin[m]
        print(f'{m:+5d} {t:8d} {100 * made / t:5.1f}% {100.0 * (m >= 0):14.0f}% {100 * pred / t:10.1f}%')


def rescore(args):
    model = Model(json.load(open(args.model)))
    stats = Counter()
    with open(args.games, encoding='utf-8') as f, open(args.out, 'w', encoding='utf-8') as out:
        for line in f:
            g = json.loads(line)
            if 'error' not in g:
                dd = g['dd']
                vul = g['vul']

                def exp_score(text):
                    c = parse_contract(text)
                    if c is None:
                        return 0
                    level, strain, dbl, decl = c
                    return int(round(model.expected(level, strain, dbl, decl, dd[STRAINS.index(strain)][SEATS.index(decl)], vul)))
                g['dd_score'] = g['score']
                g['score'] = exp_score(g.get('contract'))
                for d in g.get('decisions', []):
                    for c in d['cands']:
                        new = exp_score(c[2])
                        stats['candidates'] += 1
                        stats['score changed'] += new != c[3]
                        c.append(c[3])
                        c[3] = new
                stats['games'] += 1
            out.write(json.dumps(g) + '\n')
    print(json.dumps(dict(stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='what', required=True)
    p = sub.add_parser('fit')
    p.add_argument('field')
    p.add_argument('out')
    p.add_argument('--types', nargs='*', default=['Robot'])
    p.add_argument('--holdout', type=int, default=5)
    p = sub.add_parser('check')
    p.add_argument('model')
    p.add_argument('field')
    p = sub.add_parser('rescore')
    p.add_argument('model')
    p.add_argument('games')
    p.add_argument('out')
    args = parser.parse_args()
    {'fit': fit, 'check': check, 'rescore': rescore}[args.what](args)


if __name__ == '__main__':
    sys.exit(main())
