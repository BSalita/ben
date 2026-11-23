#!/usr/bin/env python3
"""Would another network's first choice, added to the candidates, bring a config nearer a Brill entrant?

search_diagnosis.py showed where a config's bidder never looks at the entrant's
call. This asks whether a second network would have named it. At every South
turn of the entrant's auction the config bids as usual, each second network
gives its first choice, and where that call is not already a candidate the
config bids again with it added (its probability is the config's own network's,
as in search_diagnosis.py).

  elo/.venv/Scripts/python.exe second_net_check.py export FIELD.jsonl BOARDS.parquet CONFIG TURNS.jsonl [ENTRANT]
  python second_net_check.py run --conf CONF --turns TURNS.jsonl --out OUT --second NAME=CONF [NAME=CONF ...]
  python second_net_check.py summary TURNS.jsonl OUT/turns.jsonl [TABLES.txt]

A turn is 'on our way' when the config's own replay reaches it (the calls so far
are the same); the first turn where the replay parts from the entrant is the
'parting turn'. 'diff' is the config's matchpoint percentage minus the entrant's
on the board (gap_analysis.py).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import search_diagnosis as sd  # noqa: E402

SEATS = 'NESW'


def export(field_path, boards_path, config, turns_path, entrant='R:BBOBen'):
    import polars as pl
    from gap_analysis import pct_against

    field = {(b['tournament_id'], b['board']): b for b in map(json.loads, open(field_path, encoding='utf-8'))}
    df = pl.read_parquet(boards_path).filter(pl.col('config') == config)
    turns, n_boards = [], 0
    for r in df.select('tournament_id', 'board', 'auction', 'score').iter_rows(named=True):
        b = field.get((r['tournament_id'], r['board']))
        if b is None:
            continue
        t = next((t for t in b['tables'] if t['player_id'] == entrant), None)
        others = [x['ns_score'] for x in b['tables'] if x['player_id'] != entrant]
        if t is None or not others:
            continue
        n_boards += 1
        diff = pct_against(r['score'], others) - pct_against(t['ns_score'], others)
        a, e = r['auction'].split('-'), t['auction']
        first = next((i for i in range(min(len(a), len(e))) if a[i] != e[i]), None)
        dealer_i = SEATS.index(b['dealer'])
        for j, call in enumerate(e):
            if (dealer_i + j) % 4 != 2:
                continue
            turns.append({
                'tournament_id': r['tournament_id'], 'board': r['board'], 'turn': j, 'deal': b['deal'],
                'dealer': b['dealer'], 'vul': b['vul'], 'scoring': b['scoring'], 'prefix': e[:j], 'their_call': call,
                'our_call': a[j] if (first is None or j <= first) and j < len(a) else None,
                'on_way': first is None or j <= first, 'parting': first == j, 'diff': diff,
            })
    with open(turns_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({'meta': True, 'config': config, 'entrant': entrant, 'boards': n_boards}) + '\n')
        for t in turns:
            f.write(json.dumps(t) + '\n')
    print(f'{config} against {entrant}: {n_boards} boards, {len(turns)} South turns of the entrant, '
          f'{sum(t["on_way"] for t in turns)} on our way, {sum(t["parting"] for t in turns)} parting turns -> {turns_path}')


def worker_init(cfg):
    sd.worker_init(cfg)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    import ab_match
    from nn.models_tf2 import Models
    sd.W['second'] = {}
    for name, path in cfg['second']:
        models = Models.from_conf(ab_match.load_conf(path, None), ab_match.BEN_ROOT)
        ab_match.platform_fixups(models)
        sd.W['second'][name] = models
    print(f"worker {os.getpid()}: second networks {list(sd.W['second'])}")


def first_choice(models, node):
    """The network's most likely legal call in the node's position, and its probability."""
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from sample import Sample

    hands = node['deal'][2:].split()
    dealer_i = SEATS.index(node['dealer'])
    bot = BotBid(sd.VULN[node['vul']], hands[2], models, Sample.from_conf(sd.W['conf']), 2, dealer_i, sd.W['dds'], False, False)
    auction = ['PAD_START'] * dealer_i + [sd.to_ben(c) for c in node['prefix']]
    soft = np.array(bot.next_bid_np(auction)[0], dtype=float)
    for i in np.argsort(-soft):
        if bidding.can_bid(bidding.ID2BID[i], auction):
            return sd.from_ben(bidding.ID2BID[i]), round(float(soft[i]), 4)
    raise ValueError('no legal call')


def run_turn(node):
    t0 = time.time()
    key = {'tournament_id': node['tournament_id'], 'board': node['board'], 'turn': node['turn']}
    try:
        natural = sd.one_bid(node, probe=node['their_call'])
        offered = [c['call'] for c in natural['offered']]
        out = {**key, 'natural': natural, 'second': {}, 'added': {}}
        for name, models in sd.W['second'].items():
            call, p = first_choice(models, node)
            out['second'][name] = {'call': call, 'p': p}
            if call not in offered and call not in out['added']:
                out['added'][call] = sd.one_bid(node, add_call=sd.to_ben(call))
        return {**out, 'seconds': round(time.time() - t0, 1)}
    except Exception:
        return {**key, 'error': traceback.format_exc(), 'seconds': round(time.time() - t0, 1)}


def run(args):
    turns = [json.loads(line) for line in open(args.turns, encoding='utf-8')]
    turns = [t for t in turns if not t.get('meta')]
    if args.limit:
        turns = turns[:args.limit]
    os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)
    path = os.path.join(args.out, 'turns.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['tournament_id'], r['board'], r['turn']))
    tasks = [t for t in turns if (t['tournament_id'], t['board'], t['turn']) not in done]
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    import ab_match
    second = [(s.split('=', 1)[0], ab_match.resolve(s.split('=', 1)[1])) for s in args.second]
    cfg = {'conf': ab_match.resolve(args.conf), 'out': args.out, 'dds_threads': args.dds_threads,
           'tf_threads': args.tf_threads, 'second': second}
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump({'argv': sys.argv, 'cfg': cfg, 'turns': len(turns)}, f, indent=1)
    workers = max(1, min(args.workers, len(tasks)))
    print(f'{len(tasks)} of {len(turns)} turns to bid, {workers} workers, conf {cfg["conf"]}, second {second}')
    sys.stdout.flush()
    t0, errors = time.time(), 0
    if tasks:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, initializer=worker_init, initargs=(cfg,)) as pool, open(path, 'a') as out:
            for k, r in enumerate(pool.imap_unordered(run_turn, tasks, chunksize=1), start=1):
                out.write(json.dumps(r) + '\n')
                out.flush()
                errors += 'error' in r
                if k % 200 == 0 or k == len(tasks):
                    print(f'[{k}/{len(tasks)}] {time.time() - t0:6.0f} s  errors {errors}')
                    sys.stdout.flush()
    return 1 if errors else 0


def summary(turns_path, out_path, tables_path=None):
    import io
    import numpy as np
    lines = [json.loads(line) for line in open(turns_path, encoding='utf-8')]
    meta = lines[0]
    turns = {(t['tournament_id'], t['board'], t['turn']): t for t in lines[1:]}
    res = {}
    for line in open(out_path, encoding='utf-8'):
        r = json.loads(line)
        if 'error' not in r:
            res[(r['tournament_id'], r['board'], r['turn'])] = r
    buf = io.StringIO()

    def say(text=''):
        print(text)
        buf.write(text + '\n')

    rows = [{**turns[k], **res[k]} for k in turns if k in res]
    on_way = [r for r in rows if r['on_way']]
    wrong = sum(r['natural']['call'] != r['our_call'] for r in on_way)
    say(f"{meta['config']} against {meta['entrant']}: {meta['boards']} boards, {len(rows)} South turns of the entrant bid, "
        f"{len(on_way)} on our way")
    say(f"turns on our way where the call made now is not the saved replay's: {wrong}")
    names = sorted({n for r in rows for n in r['second']})

    def pct(a, b):
        return f'{100 * a / b:5.1f}%' if b else '   - '

    def diffs(part):
        d = np.array([r['diff'] for r in part], dtype=float)
        if len(d) < 2:
            return f'{len(d):>5}'
        return f'{len(d):>5}  board diff {d.mean():+6.1f} +-{1.96 * d.std() / len(d) ** 0.5:5.1f}'

    for scope, part in (('turns on our way (what the replay would meet)', on_way), ('all South turns of the entrant', rows)):
        say(f'\n===== {scope}: {len(part)}')
        nat_ok = sum(r['natural']['call'] == r['their_call'] for r in part)
        say(f"our call is the entrant's on {nat_ok} ({pct(nat_ok, len(part))})")
        for name in names:
            sub = [r for r in part if name in r['second']]
            sec_ok = sum(r['second'][name]['call'] == r['their_call'] for r in sub)
            differs = [r for r in sub if r['second'][name]['call'] != r['natural']['call']]
            cand = [r for r in differs if r['second'][name]['call'] not in r['added']]
            added = [r for r in differs if r['second'][name]['call'] in r['added']]

            def new_call(r):
                return r['added'][r['second'][name]['call']]['call']

            say(f"\n-- second network '{name}'")
            say(f"   its first choice is the entrant's call on {sec_ok} ({pct(sec_ok, len(sub))}); it differs from our call on {len(differs)}")
            say(f"   of those, already one of our candidates: {len(cand)}; added as a candidate: {len(added)}")
            kept = [r for r in added if new_call(r) == r['natural']['call']]
            moved = [r for r in added if new_call(r) != r['natural']['call']]
            say(f"   with it added the bidder keeps its call on {len(kept)} and changes on {len(moved)}")
            toward = [r for r in moved if r['natural']['call'] != r['their_call'] and new_call(r) == r['their_call']]
            away = [r for r in moved if r['natural']['call'] == r['their_call']]
            side = [r for r in moved if r['natural']['call'] != r['their_call'] and new_call(r) != r['their_call']]
            say(f"     toward: we differed, now make the entrant's call   {diffs(toward)}")
            say(f"     away:   we made the entrant's call, now another     {diffs(away)}")
            say(f"     aside:  we differed, now make a third call          {diffs(side)}")
            say(f"     agreement with the entrant: {nat_ok} -> {nat_ok + len(toward) - len(away)} of {len(part)}")
            say("   by the second network's probability for its call (added turns): turns, changed, toward, away, aside")
            for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
                band = [r for r in added if lo <= r['second'][name]['p'] < hi]
                ids = {id(r) for r in band}
                say(f"     {lo:g} to {hi:g}: {len(band):>5} {sum(id(r) in ids for r in moved):>5} "
                    f"{sum(id(r) in ids for r in toward):>5} {sum(id(r) in ids for r in away):>5} {sum(id(r) in ids for r in side):>5}")
            say("   by our network's probability for our own call (added turns): turns, changed, toward, away, aside")
            for lo, hi in [(0, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]:
                def ours_p(r):
                    return next((c['p'] for c in r['natural']['offered'] if c['call'] == r['natural']['call']), 0.0)
                band = [r for r in added if lo <= ours_p(r) < hi]
                ids = {id(r) for r in band}
                say(f"     {lo:g} to {hi:g}: {len(band):>5} {sum(id(r) in ids for r in moved):>5} "
                    f"{sum(id(r) in ids for r in toward):>5} {sum(id(r) in ids for r in away):>5} {sum(id(r) in ids for r in side):>5}")
    if tables_path:
        with open(tables_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(buf.getvalue())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        return export(*sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        return summary(*sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=['run'])
    parser.add_argument('--conf', required=True)
    parser.add_argument('--turns', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--second', nargs='+', required=True, help='NAME=CONF of each second network')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dds-threads', type=int, default=3)
    parser.add_argument('--tf-threads', type=int, default=2)
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
