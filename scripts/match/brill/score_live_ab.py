"""Compare configs bid on the same deals with Brill's robots (brill_live.py), double dummy.

Run on Windows with the Elo venv:

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/score_live_ab.py DEALS.jsonl OUT_DIR base=replay.jsonl name=replay.jsonl [...]

Every config's final contract on a deal gets its double-dummy score. Each config
after the first is set against the first on the deals both finished:
  IMPs   the IMP value of the score difference, mean per deal
  MP %   100 for the better score, 50 for the same, 0 for the worse, mean per deal
with 95% intervals over deals. Also each config against par (IMPs), and the
split by the deal's scoring (the config bids to that scoring).

With the environment variable PLAY_MODEL set to a model of scripts/selfplay/play_model.py, a
contract's score is instead its expected score in real card play (the distribution of tricks
taken around the double-dummy number, learnt from played boards).

Writes OUT_DIR/dd.json (20 trick counts and par per deal, reused), OUT_DIR/boards.parquet
and OUT_DIR/compare.txt.
"""
import io
import json
import multiprocessing as mp
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_brill_replay import contract_of, imps, score_ns  # noqa: E402

STRAINS = 'CDHSN'
SEATS = 'NESW'


def dd_of(args):
    key, deal, dealer, vul = args
    from endplay.dds import calc_dd_table, par
    from endplay.types import Deal, Denom, Player, Vul
    table = calc_dd_table(Deal(deal))
    denoms = {'S': Denom.spades, 'H': Denom.hearts, 'D': Denom.diamonds, 'C': Denom.clubs, 'N': Denom.nt}
    players = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}
    tricks = {s + p: int(table[denoms[s], players[p]]) for s in STRAINS for p in SEATS}
    vuls = {'None': Vul.none, 'NS': Vul.ns, 'EW': Vul.ew, 'All': Vul.both}
    return key, {'tricks': tricks, 'par': int(par(table, vuls[vul], players[dealer]).score)}


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')


def main():
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}')
    deals_path, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    replays = dict(arg.split('=', 1) for arg in sys.argv[3:])
    deals = {f"{b['tournament_id']}|{b['board']}": b for b in map(json.loads, deals_path.open(encoding='utf-8'))}

    runs = {}
    for name, path in replays.items():
        runs[name] = {}
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                runs[name][f"{r['tournament_id']}|{r['board']}"] = r
    needed = sorted(set().union(*runs.values()))

    dd_path = out_dir / 'dd.json'
    dd = json.loads(dd_path.read_text()) if dd_path.exists() else {}
    todo = [(k, deals[k]['deal'], deals[k]['dealer'], deals[k]['vul']) for k in needed if k not in dd]
    if todo:
        print(f'double dummy for {len(todo)} deals ...', flush=True)
        with mp.Pool(min(24, mp.cpu_count())) as pool:
            for key, value in pool.imap_unordered(dd_of, todo, chunksize=16):
                dd[key] = value
        dd_path.write_text(json.dumps(dd))

    play = None
    if os.environ.get('PLAY_MODEL'):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'selfplay'))
        import play_model
        play = play_model.Model(json.load(open(os.environ['PLAY_MODEL'])))

    rows = []
    for name, run in runs.items():
        for key, r in run.items():
            b = deals[key]
            contract = contract_of(r['walk'], b['dealer'])
            if contract is None:
                score, text = 0, 'Pass'
            else:
                tricks = dd[key]['tricks'][contract[1] + contract[3]]
                if play is None:
                    score = score_ns(*contract, tricks, b['vul'])
                else:
                    score = round(play.expected(*contract, tricks, b['vul']))
                text = f'{contract[0]}{contract[1]}{contract[2]} {contract[3]}'
            rows.append({'config': name, 'key': key, 'scoring': b['scoring'], 'vul': b['vul'],
                         'auction': '-'.join(r['walk']), 'contract': text, 'score': int(score),
                         'par': dd[key]['par'], 'imp_vs_par': float(imps(score - dd[key]['par']))})
    boards = pl.DataFrame(rows)
    boards.write_parquet(out_dir / 'boards.parquet')

    out = io.StringIO()

    def say(text=''):
        print(text)
        out.write(text + '\n')

    names = list(runs)
    base = names[0]
    say(f"{len(deals)} deals in {deals_path.name}; "
        f"{'expected real-play scores (' + os.environ['PLAY_MODEL'] + ')' if play else 'double-dummy scores'}, North-South view")
    say()
    say('Each config (boards finished, against par)')
    say(f"{'config':24s} {'boards':>6s} {'mean score':>10s} {'IMPs vs par':>11s} {'+-':>6s}")
    for name in names:
        d = boards.filter(pl.col('config') == name)
        say(f"{name:24s} {len(d):6d} {d['score'].mean():10.1f} {d['imp_vs_par'].mean():11.3f} {ci(d['imp_vs_par']):6.3f}")
    b0 = boards.filter(pl.col('config') == base).select('key', pl.col('score').alias('b_score'),
                                                        pl.col('auction').alias('b_auction'),
                                                        pl.col('contract').alias('b_contract'))
    for name in names[1:]:
        d = boards.filter(pl.col('config') == name).join(b0, on='key', how='inner')
        diff = (d['score'] - d['b_score']).to_numpy()
        d = d.with_columns(pl.Series('imp', imps(diff).astype(float)), pl.Series('mp', 50.0 + 50.0 * np.sign(diff)))
        say()
        say(f'{name} against {base}: {len(d)} deals both finished; auction differs on '
            f"{(d['auction'] != d['b_auction']).mean() * 100:.1f}%, contract on {(d['contract'] != d['b_contract']).mean() * 100:.1f}%, "
            f"score on {(diff != 0).mean() * 100:.1f}%")
        say(f"{'deals':16s} {'n':>6s} {'IMPs/deal':>9s} {'+-':>6s} {'MP %':>6s} {'+-':>5s} {'better':>6s} {'worse':>6s}")
        for label, part in (('all', d), ('MP scoring', d.filter(pl.col('scoring') == 'MP')),
                            ('IMP scoring', d.filter(pl.col('scoring') == 'IMP')),
                            ('nobody vul', d.filter(pl.col('vul') == 'None')), ('NS vul', d.filter(pl.col('vul') == 'NS')),
                            ('EW vul', d.filter(pl.col('vul') == 'EW')), ('all vul', d.filter(pl.col('vul') == 'All'))):
            if len(part) < 2:
                continue
            say(f"{label:16s} {len(part):6d} {part['imp'].mean():+9.3f} {ci(part['imp']):6.3f} {part['mp'].mean():6.2f} "
                f"{ci(part['mp']):5.2f} {(part['mp'] == 100).sum():6d} {(part['mp'] == 0).sum():6d}")
    (out_dir / 'compare.txt').write_text(out.getvalue(), encoding='utf-8')
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}')


if __name__ == '__main__':
    main()
