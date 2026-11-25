#!/usr/bin/env python3
"""Why a replayed setting misses the games (and slams, and strains) most Brill tables bid.

A leak board: the setting's contract is below what most tables reached, in one of three ways:

  game     we play a North-South partscore (or pass it out), most tables a North-South game
  slam     we play a North-South game, most tables a North-South slam
  strain   we play the same kind of contract as most tables (part, game, slam), in another strain

On each, take the table in the leak's target (most tables' kind of contract) whose auction
shares the longest start with ours. Where the two first part is the deciding turn; if it is
South's, the position can be bid again with search_diagnosis.py's run step, which shows
whether the bidder offered the table's call at all (one candidate / not a candidate), or
searched it and chose ours, and then what the search makes of the table's call when it has to
look at it.

  elo/.venv/Scripts/python.exe leak_diagnosis.py export FIELD.jsonl REPLAY.jsonl NODES.jsonl
  python search_diagnosis.py run --conf CONF --nodes NODES.jsonl --out OUT          (container)
  elo/.venv/Scripts/python.exe leak_diagnosis.py summary NODES.jsonl OUT/diag.jsonl [REPORT.txt]

Scores: ours is the replay's result as brill30.py takes it (a real table with the same
auction, else the same contract, else double dummy); 'imp' is our score against the chosen
game table's in IMPs (negative: the leak cost us), 'imp_field' our cross-IMPs against the
field minus that table's.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_brill_replay as sbr  # noqa: E402
from gap_analysis import stage, HCP  # noqa: E402

SEATS = 'NESW'
GAME_TRICKS = {'C': 5, 'D': 5, 'H': 4, 'S': 4, 'N': 3}
LIVE_ID = 'R:MortyAI'


def kind(contract):
    """('pass'|'ns'|'ew', 'part'|'game'|'slam'|'grand'|None, strain)"""
    if contract is None:
        return 'pass', None, None
    level, strain, _, decl = contract
    side = 'ns' if decl in 'NS' else 'ew'
    k = 'grand' if level == 7 else 'slam' if level == 6 else 'game' if level >= GAME_TRICKS[strain] else 'part'
    return side, k, strain


def board_score(board, walk, tables):
    contract = sbr.contract_of(walk, board['dealer'])
    same = [t for t in tables if t['auction'] == walk]
    if same:
        return int(sbr.pick(same)), 'auction'
    same = [t for t in tables if sbr.table_contract(t) == contract]
    if same:
        return int(sbr.pick(same)), 'contract'
    if contract is None:
        return 0, 'dd'
    tricks = sbr.dd_tricks(board['deal'], contract[1], contract[3])
    return int(sbr.score_ns(*contract, tricks, board['vul'])), 'dd'


def cross_imps(x, others):
    return float(np.mean(sbr.imps(np.asarray([x - o for o in others])))) if others else 0.0


def export(field_path, replay_path, nodes_path):
    field = {}
    for line in open(field_path, encoding='utf-8'):
        b = json.loads(line)
        if b['event'].startswith('Daily'):
            field[(b['tournament_id'], b['board'])] = b
    nodes, counts = [], Counter()
    for line in open(replay_path, encoding='utf-8'):
        r = json.loads(line)
        b = field.get((r['tournament_id'], r['board']))
        if b is None or 'walk' not in r:
            continue
        tables = [t for t in b['tables'] if t['player_id'] != LIVE_ID]
        if len(tables) < 3:
            continue
        counts['boards'] += 1
        walk = r['walk']
        ours = kind(sbr.contract_of(walk, b['dealer']))
        tk = [kind(sbr.table_contract(t)) for t in tables]
        n = len(tables)
        leak, target = None, None
        ns_game_up = [i for i, k in enumerate(tk) if k[0] == 'ns' and k[1] in ('game', 'slam', 'grand')]
        ns_slam_up = [i for i, k in enumerate(tk) if k[0] == 'ns' and k[1] in ('slam', 'grand')]
        if (ours[0] == 'pass' or (ours[0] == 'ns' and ours[1] == 'part')) and len(ns_game_up) * 2 > n \
                and len(ns_slam_up) * 2 <= n:
            leak, target = 'game', [i for i in ns_game_up if tk[i][1] == 'game']
        elif ours[0] == 'ns' and ours[1] == 'game' and len(ns_slam_up) * 2 > n:
            leak, target = 'slam', ns_slam_up
        elif ours[0] == 'ns':
            same_kind = Counter(tk[i][2] for i in range(n) if tk[i][0] == 'ns' and tk[i][1] == ours[1])
            if same_kind:
                strain, cnt = same_kind.most_common(1)[0]
                if strain != ours[2] and cnt * 2 > n:
                    leak, target = 'strain', [i for i in range(n) if tk[i][:3] == ('ns', ours[1], strain)]
        if leak is None:
            continue
        counts[leak] += 1
        our_score, source = board_score(b, walk, tables)
        others = [t['ns_score'] for t in tables]

        def common(t):
            k = 0
            while k < min(len(walk), len(t['auction'])) and walk[k] == t['auction'][k]:
                k += 1
            return k
        best = max((tables[i] for i in target), key=lambda t: (common(t), t['player_id'] == 'R:Brill'))
        i = common(best)
        dealer_i = SEATS.index(b['dealer'])
        seat = SEATS[(dealer_i + i) % 4]
        their_kind = kind(sbr.table_contract(best))
        south = b['deal'][2:].split()[2]
        node = {
            'tournament_id': b['tournament_id'], 'board': b['board'], 'date': b['date'][:10], 'deal': b['deal'],
            'dealer': b['dealer'], 'vul': b['vul'], 'scoring': b['scoring'], 'leak': leak, 'seat': seat,
            'prefix': walk[:i], 'our_call': walk[i] if i < len(walk) else None,
            'their_call': best['auction'][i] if i < len(best['auction']) else None,
            'our_auction': '-'.join(walk), 'their_auction': '-'.join(best['auction']), 'their_player': best['player_id'],
            'ours': f"{ours[0]} {ours[1]} {ours[2]}", 'theirs': f"{their_kind[0]} {their_kind[1]} {their_kind[2]}",
            'our_score': our_score, 'our_source': source, 'their_score': best['ns_score'],
            'imp': int(sbr.imps(our_score - best['ns_score'])),
            'imp_field': cross_imps(our_score, others) - cross_imps(best['ns_score'], others),
            'target_share': len(target) / n, 'tables': n, 'stage': stage(walk[:i], dealer_i),
            'hcp': sum(HCP.get(c, 0) for c in south), 'south': south,
        }
        if seat != 'S':
            counts[f'{leak}: parts at a robot call'] += 1
        nodes.append(node)
    with open(nodes_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'meta': True, 'replay': replay_path, **counts}) + '\n')
        for n_ in nodes:
            f.write(json.dumps(n_) + '\n')
    print(dict(counts))
    for leak in ('game', 'slam', 'strain'):
        g = [x for x in nodes if x['leak'] == leak]
        if g:
            print(f"{leak}: {len(g)} boards, at a South call {sum(x['seat'] == 'S' for x in g)}, "
                  f"IMPs against the chosen table {np.mean([x['imp'] for x in g]):+.2f} a board, "
                  f"against the field {np.mean([x['imp_field'] for x in g]):+.2f}")


# ---------------------------------------------------------------- summary

def group(d, node):
    nat = d['natural']
    offered = [c['call'] for c in nat['offered']]
    theirs = node['their_call']
    if theirs not in offered:
        return '1 one candidate' if len(offered) == 1 else '2 not a candidate'
    if nat['call'] == theirs:
        return '0 now bids theirs'
    if nat.get('searched'):
        return '3 searched, ours chosen'
    return '4 offered, no search'


def sim_of(res, call):
    for c in res['result']:
        if c['call'] == call:
            return c['sim'], c.get('adjust')
    return None, None


def summary(nodes_path, diag_path, out_path=None):
    nodes = {}
    for line in open(nodes_path, encoding='utf-8'):
        n = json.loads(line)
        if not n.get('meta'):
            nodes[(n['tournament_id'], n['board'])] = n
    diag = {}
    for line in open(diag_path, encoding='utf-8'):
        d = json.loads(line)
        if 'error' not in d:
            diag[(d['tournament_id'], d['board'])] = d
    lines = []

    def row(title, xs):
        if not xs:
            lines.append(f'  {title:<52} {0:>6}')
            return
        imp = np.array([x['imp'] for x in xs], float)
        f = np.array([x['imp_field'] for x in xs], float)
        ci = 1.96 * imp.std(ddof=1) / math.sqrt(len(imp)) if len(imp) > 1 else float('nan')
        won = int((imp > 0).sum()); lost = int((imp < 0).sum())
        lines.append(f'  {title:<52} {len(xs):>6} {imp.mean():>+7.2f} {ci:>5.2f} {f.mean():>+7.2f} {imp.sum():>+7.0f} {won:>5} {lost:>5}')

    head = f"  {'':<52} {'boards':>6} {'IMPs':>7} {'+-':>5} {'field':>7} {'sum':>7} {'we>':>5} {'we<':>5}"
    for leak in ('game', 'slam', 'strain'):
        g = [n for n in nodes.values() if n['leak'] == leak]
        if not g:
            continue
        lines.append(f'== {leak} ({len(g)} boards; IMPs = ours against the table that reached the target, a board) ==')
        lines.append(head)
        row('all', g)
        row('parts at a robot call', [n for n in g if n['seat'] != 'S'])
        south = [n for n in g if n['seat'] == 'S']
        row('parts at a South call', south)
        lines.append('  at the South call, what the bidder does now:')
        by = defaultdict(list)
        for n in south:
            d = diag.get((n['tournament_id'], n['board']))
            if d is None:
                by['9 not bid again'].append(n)
                continue
            if d['natural']['call'] != n['our_call']:
                by['8 now bids neither saved call' if d['natural']['call'] != n['their_call'] else '0 now bids theirs'].append(n)
                continue
            by[group(d, n)].append(n)
        for k in sorted(by):
            row('  ' + k, by[k])
        # what the search makes of the table's call when it has to look at it
        lines.append("  table's call added to the candidates (groups 1 and 2), and searched ones (group 3):")
        ch = defaultdict(list)
        margins = []
        for n in south:
            d = diag.get((n['tournament_id'], n['board']))
            if d is None or d['natural']['call'] != n['our_call']:
                continue
            g_ = group(d, n)
            res = d.get('forced') if g_[0] in '12' else d['natural'] if g_[0] == '3' else None
            if res is None:
                continue
            if not res.get('searched'):
                ch['no search even so'].append(n)
                continue
            so, ao = sim_of(res, n['our_call'])
            st, at = sim_of(res, n['their_call'])
            label = 'forced' if g_[0] in '12' else 'searched'
            if res['call'] == n['their_call']:
                ch[f'{label}: now chooses theirs'].append(n)
            elif so is not None and st is not None and st > so:
                ch[f'{label}: search has theirs higher, adjustments keep ours'].append(n)
            else:
                ch[f'{label}: search has ours higher or equal'].append(n)
            if so is not None and st is not None:
                margins.append((st - so, n))
        for k in sorted(ch):
            row('  ' + k, ch[k])
        if margins:
            lines.append("  the search's value of theirs minus ours, against what really happened (their table minus ours):")
            agree = sum((m > 0) == (n['imp'] < 0) for m, n in margins if n['imp'] != 0)
            dec = sum(1 for m, n in margins if n['imp'] != 0)
            corr = np.corrcoef([m for m, _ in margins], [-n['imp'] for _, n in margins])[0, 1] if len(margins) > 2 else float('nan')
            lines.append(f'    sign agrees on {agree} of {dec} boards with a swing; correlation {corr:+.2f}')
        by_stage = defaultdict(list)
        for n in south:
            by_stage[n['stage']].append(n)
        lines.append('  by the stage of the deciding South call:')
        for k in sorted(by_stage):
            row('  ' + k, by_stage[k])
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if out_path:
        with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text + '\n')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        return export(*sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        return summary(*sys.argv[2:])
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
