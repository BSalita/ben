"""Compare groups of replayed training runs (seeds) from a scored Brill replay.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/compare_seeds.py OUT_DIR GROUP=name,name,... [GROUP=...]

Reads OUT_DIR/boards.parquet (score_brill_replay.py). Prints every run, every
group, and each pair of groups two ways:
  boards  the difference of the group means on each board, averaged over
          boards (the interval reflects which boards were dealt)
  runs    the difference of the group means over runs, with a Welch interval
          from the spread between runs (reflects training luck)
"""
import pathlib
import sys

import numpy as np
import polars as pl
from scipy import stats


def ci_boards(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def main():
    out_dir = pathlib.Path(sys.argv[1])
    groups = {}
    for arg in sys.argv[2:]:
        name, members = arg.split('=', 1)
        groups[name] = members.split(',')
    scored = pl.read_parquet(out_dir / 'boards.parquet').sort('tournament_id', 'board')
    configs = scored['config'].unique(maintain_order=True).to_list()
    missing = [m for members in groups.values() for m in members if m not in configs]
    if missing:
        raise SystemExit(f'not in boards.parquet: {missing}')

    # board x config matrices
    keys = scored.filter(pl.col('config') == configs[0]).select('tournament_id', 'board')
    pct, imp, score = {}, {}, {}
    for c in configs:
        rows = keys.join(scored.filter(pl.col('config') == c), on=['tournament_id', 'board'], how='left', validate='1:1')
        if rows['pct_all'].null_count():
            raise SystemExit(f'{c}: boards missing')
        pct[c], imp[c], score[c] = rows['pct_all'].to_numpy(), rows['imp_all'].to_numpy(), rows['score'].to_numpy()
    n = keys.height

    group_of = {m: g for g, members in groups.items() for m in members}
    print(f'{n} boards\n\nEvery run (matchpoint % and IMPs per board against the real field; +- is the board interval)')
    print(f'{"run":<24}{"group":<12}{"pct":>7}{"+-":>6}{"imp":>8}{"+-":>7}')
    for c in configs:
        m_p, ci_p = ci_boards(pct[c])
        m_i, ci_i = ci_boards(imp[c])
        print(f'{c:<24}{group_of.get(c, ""):<12}{m_p:>7.2f}{ci_p:>6.2f}{m_i:>8.3f}{ci_i:>7.3f}')

    print('\nGroups (mean over runs, spread between runs as standard deviation)')
    print(f'{"group":<12}{"runs":>5}{"pct":>8}{"sd":>6}{"min":>7}{"max":>7}{"imp":>8}{"sd":>7}')
    means = {}
    for g, members in groups.items():
        p = np.array([pct[m].mean() for m in members])
        i = np.array([imp[m].mean() for m in members])
        means[g] = (p, i)
        sd_p = p.std(ddof=1) if len(p) > 1 else float('nan')
        sd_i = i.std(ddof=1) if len(i) > 1 else float('nan')
        print(f'{g:<12}{len(members):>5}{p.mean():>8.2f}{sd_p:>6.2f}{p.min():>7.2f}{p.max():>7.2f}{i.mean():>8.3f}{sd_i:>7.3f}')

    print('\nGroup against group (first minus second)')
    names = list(groups)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            d_pct = np.mean([pct[m] for m in groups[a]], axis=0) - np.mean([pct[m] for m in groups[b]], axis=0)
            d_imp = np.mean([imp[m] for m in groups[a]], axis=0) - np.mean([imp[m] for m in groups[b]], axis=0)
            m_p, ci_p = ci_boards(d_pct)
            m_i, ci_i = ci_boards(d_imp)
            line = f'{a} - {b}: boards: pct {m_p:+.2f} +-{ci_p:.2f}, imp {m_i:+.3f} +-{ci_i:.3f}'
            (pa, ia), (pb, ib) = means[a], means[b]
            if len(pa) > 1 and len(pb) > 1:
                t_p = stats.ttest_ind(pa, pb, equal_var=False)
                t_i = stats.ttest_ind(ia, ib, equal_var=False)
                lo, hi = t_p.confidence_interval()
                line += f'; runs: pct {pa.mean() - pb.mean():+.2f} (95% {lo:+.2f} to {hi:+.2f}, p={t_p.pvalue:.3f}), imp p={t_i.pvalue:.3f}'
            print(line)

    print('\nHow often two runs of one group give a different score on a board (%)')
    for g, members in groups.items():
        if len(members) < 2:
            continue
        diffs = [np.mean(score[x] != score[y]) * 100 for k, x in enumerate(members) for y in members[k + 1:]]
        print(f'{g:<12}{np.mean(diffs):6.1f}')


if __name__ == '__main__':
    main()
