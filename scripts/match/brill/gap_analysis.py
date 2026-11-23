"""Where a replayed config loses to (or gains on) a real Brill entrant, board by board.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/gap_analysis.py FIELD.jsonl BOARDS.parquet CONFIG[,CONFIG...] [ENTRANT ...]

BOARDS.parquet is the output of score_brill_replay.py. With several configs
(training runs of one recipe) their results are averaged, so what is left is
what the recipe does, not one run's luck.

For every board the entrant played, the config's first South call that differs
from the entrant's is classified (stage of the auction, direction), and so is
the pair of final contracts. Each table gives boards, and the summed matchpoint
difference divided by all compared boards: the rows add up to the whole gap.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import polars as pl

SEATS = 'NESW'
HCP = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}


def is_bid(call):
    return call[0].isdigit()


def rank(call):
    return (int(call[0]) - 1) * 5 + 'CDHSN'.index(call[1])


def contract_class(calls, dealer_i):
    """('NS'|'EW'|'-', 'pass'|'part'|'game'|'slam', doubled) of an auction."""
    last = None
    for i, c in enumerate(calls):
        if is_bid(c):
            last = i
    if last is None:
        return '-', 'pass', ''
    level, strain = int(calls[last][0]), calls[last][1]
    dbl = ''
    for c in calls[last + 1:]:
        if c in ('X', 'XX'):
            dbl = 'X'
    side = 'NS' if (dealer_i + last) % 2 == 0 else 'EW'
    game = {'N': 3, 'H': 4, 'S': 4, 'C': 5, 'D': 5}[strain]
    kind = 'slam' if level >= 6 else 'game' if level >= game else 'part'
    return side, kind, dbl


def stage(prefix, dealer_i):
    """What kind of South turn this is."""
    ns = [c for i, c in enumerate(prefix) if (dealer_i + i) % 2 == 0]
    ew = [c for i, c in enumerate(prefix) if (dealer_i + i) % 2 == 1]
    south = [c for i, c in enumerate(prefix) if (dealer_i + i) % 4 == 2]
    north = [c for i, c in enumerate(prefix) if (dealer_i + i) % 4 == 0]
    ns_bid = any(c != 'P' for c in ns)
    ew_bid = any(c != 'P' for c in ew)
    first = not any(c != 'P' for c in south)
    north_acted = any(c != 'P' for c in north)
    if not ns_bid and not ew_bid:
        return '1 opening'
    if first and not north_acted:
        return '2 they opened, first action'
    if first and north_acted and not ew_bid:
        return '3 response, no interference'
    if first and north_acted:
        return '4 response in competition'
    if not ew_bid:
        return '5 later round, uncontested'
    return '6 later round, contested'


def direction(ours, theirs):
    def kind(c):
        return 'pass' if c == 'P' else 'dbl' if c in ('X', 'XX') else 'bid'
    a, b = kind(ours), kind(theirs)
    if a == 'bid' and b == 'bid':
        ra, rb = rank(ours), rank(theirs)
        if ours[1] == theirs[1]:
            return 'we bid higher, same strain' if ra > rb else 'we bid lower, same strain'
        return 'other strain, we higher' if ra > rb else 'other strain, we lower'
    return f'we {a}, they {b}'


def pct_against(mine, others):
    others = np.asarray(others)
    return float((np.sum(others < mine) + 0.5 * np.sum(others == mine)) / len(others) * 100)


def table(rows, keys, total, title, min_boards=8):
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        k = tuple(r[x] for x in keys)
        agg[k][0] += r['w']
        agg[k][1] += r['w'] * r['diff']
        agg[k][2] += r['w'] * r['diff'] ** 2
    out = []
    for k, (n, s, ss) in agg.items():
        if n < min_boards:
            continue
        mean = s / n
        sd = max(ss / n - mean ** 2, 0) ** 0.5
        out.append({**dict(zip(keys, k)), 'boards': round(n, 1), 'mean diff': round(mean, 1),
                    '+-': round(1.96 * sd / n ** 0.5, 1), 'share of gap': round(s / total, 2)})
    out.sort(key=lambda r: r['share of gap'])
    print(f'\n{title}')
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=60):
        print(pl.DataFrame(out))


def main():
    field_path, boards_path, configs = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    entrants = sys.argv[4:] or ['R:BBOBen']
    field = {(b['tournament_id'], b['board']): b for b in map(json.loads, open(field_path, encoding='utf-8'))}
    df = pl.read_parquet(boards_path).filter(pl.col('config').is_in(configs))
    ours = defaultdict(list)
    for r in df.select('config', 'tournament_id', 'board', 'auction', 'score', 'source').iter_rows(named=True):
        ours[(r['tournament_id'], r['board'])].append(r)
    print(f'configs {configs}: {df.height} rows, {len(ours)} boards')

    for entrant in entrants:
        rows, n_boards, gap = [], 0, 0.0
        for key, b in field.items():
            t = next((t for t in b['tables'] if t['player_id'] == entrant), None)
            if t is None or key not in ours:
                continue
            dealer_i = SEATS.index(b['dealer'])
            others = [x['ns_score'] for x in b['tables'] if x['player_id'] != entrant]
            if not others:
                continue
            their_pct = pct_against(t['ns_score'], others)
            n_boards += 1
            south = b['deal'][2:].split()[2]
            hcp = sum(HCP.get(c, 0) for c in south)
            for r in ours[key]:
                w = 1 / len(ours[key])
                diff = pct_against(r['score'], others) - their_pct
                gap += w * diff
                a, e = r['auction'].split('-'), t['auction']
                i = next((i for i in range(min(len(a), len(e))) if a[i] != e[i]), None)
                if i is None:
                    continue
                oc, tc = contract_class(a, dealer_i), contract_class(e, dealer_i)
                rows.append({
                    'w': w, 'diff': diff,
                    'stage': stage(a[:i], dealer_i), 'direction': direction(a[i], e[i]),
                    'ours': f'{oc[0]} {oc[1]}{oc[2]}', 'theirs': f'{tc[0]} {tc[1]}{tc[2]}',
                    'scoring': b['scoring'], 'vul': b['vul'], 'source': r['source'],
                    'hcp': '0-7' if hcp < 8 else '8-11' if hcp < 12 else '12-14' if hcp < 15 else '15-17' if hcp < 18 else '18+',
                    'key': key, 'prefix': '-'.join(a[:i]), 'our_call': a[i], 'their_call': e[i],
                    'our_auction': r['auction'], 'their_auction': '-'.join(e),
                    'south': south, 'deal': b['deal'], 'dealer': b['dealer'],
                    'our_score': r['score'], 'their_score': t['ns_score'],
                })
        differ = sum(r['w'] for r in rows)
        print(f'\n===== {configs[0]}{"+" if len(configs) > 1 else ""} against {entrant}: {n_boards} boards, '
              f'gap {gap / n_boards:+.2f} points per board; auction differs on {differ:.0f} boards '
              f'({100 * differ / n_boards:.1f}%), mean {sum(r["w"] * r["diff"] for r in rows) / differ:+.1f} there')
        table(rows, ['stage'], n_boards, 'By the stage of the first differing South call (share of gap = points per board over all boards)')
        table(rows, ['direction'], n_boards, 'By direction of the first differing call')
        table(rows, ['stage', 'direction'], n_boards, 'Stage x direction', min_boards=12)
        table(rows, ['ours', 'theirs'], n_boards, 'By final contract: ours and theirs', min_boards=10)
        table(rows, ['scoring'], n_boards, 'By event scoring')
        table(rows, ['hcp'], n_boards, "By South's points")
        table(rows, ['source'], n_boards, 'By how our result was obtained')
        out = pl.DataFrame([{k: (str(v) if k == 'key' else v) for k, v in r.items()} for r in rows])
        name = f'gap_{configs[0]}_{entrant.replace(":", "_")}.parquet'
        out.write_parquet(boards_path.replace('boards.parquet', name))
        print(f'rows -> {name}')


if __name__ == '__main__':
    main()
