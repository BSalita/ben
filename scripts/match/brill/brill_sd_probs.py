"""Single-dummy trick probabilities for every board of a Brill field file.

    elo/.venv/Scripts/python.exe ben/scripts/match/brill/brill_sd_probs.py FIELD.jsonl OUT.json [--produce 40] [--workers 24]

The method of mlBridgeAugmentLib.estimate_sd_trick_distributions: keep one
side's two hands, deal the other two at random `produce` times (seed 42), solve
each deal double dummy, and count how often each declarer of the kept side takes
0..13 tricks in each strain. OUT.json maps a deal to {declarer+strain: [14 shares]}
(N and S from the deals that keep North-South, E and W from those that keep
East-West). Deals already in OUT.json are skipped.
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
from datetime import datetime

from tqdm import tqdm

SEATS = 'NESW'
STRAINS = 'CDHSN'


def probabilities(args):
    deal, produce = args
    from endplay.dds import calc_all_tables
    from endplay.dealer import generate_deals
    from endplay.types import Deal, Denom, Player
    denoms = {'C': Denom.clubs, 'D': Denom.diamonds, 'H': Denom.hearts, 'S': Denom.spades, 'N': Denom.nt}
    players = {'N': Player.north, 'E': Player.east, 'S': Player.south, 'W': Player.west}
    out = {}
    for side in ('NS', 'EW'):
        hands = deal[2:].split()
        for i, seat in enumerate(SEATS):
            if seat not in side:
                hands[i] = '...'
        sample = tuple(generate_deals(lambda d: True, predeal=Deal('N:' + ' '.join(hands)), swapping=0,
                                      show_progress=False, produce=produce, seed=42, max_attempts=1000000, strict=True))
        tables = []
        for start in range(0, len(sample), 32):
            tables.extend(calc_all_tables(sample[start:start + 32]))
        for declarer in side:
            for strain in STRAINS:
                counts = [0] * 14
                for table in tables:
                    counts[int(table[denoms[strain], players[declarer]])] += 1
                out[declarer + strain] = [c / len(tables) for c in counts]
    return deal, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('field')
    parser.add_argument('out')
    parser.add_argument('--produce', type=int, default=40, help='random deals per side (the library default is 40)')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}', flush=True)
    out_path = pathlib.Path(args.out)
    cache = json.loads(out_path.read_text(encoding='utf-8')) if out_path.exists() else {'produce': args.produce, 'deals': {}}
    if cache['produce'] != args.produce:
        raise SystemExit(f"{out_path} was made with --produce {cache['produce']}")
    deals = []
    for line in pathlib.Path(args.field).open(encoding='utf-8'):
        deal = json.loads(line)['deal']
        if deal not in cache['deals'] and deal not in deals:
            deals.append(deal)
    if args.limit:
        deals = deals[:args.limit]
    print(f'{len(deals)} deals to simulate, {args.produce} random deals per side, {args.workers} workers', flush=True)
    if deals:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            jobs = pool.imap_unordered(probabilities, [(d, args.produce) for d in deals], chunksize=4)
            for k, (deal, probs) in enumerate(tqdm(jobs, total=len(deals), mininterval=30, file=sys.stdout), start=1):
                cache['deals'][deal] = probs
                if k % 500 == 0:
                    out_path.write_text(json.dumps(cache), encoding='utf-8')
        out_path.write_text(json.dumps(cache), encoding='utf-8')
    ended = datetime.now()
    print(f'end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {len(cache["deals"])} deals in {out_path}', flush=True)


if __name__ == '__main__':
    main()
