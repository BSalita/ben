# A/B match

`ab_match.py` plays two BEN configurations against each other and scores the
result in IMPs, so a change to a model or to a config setting gets a number.

Every board is played at two tables. At table 1 config A sits North-South and
config B East-West; at table 2 they swap. The board's result is the IMP
difference between the two tables, from A's side.

## Run it

From the folder `scripts/match`, with the Python that runs BEN:

```
python ab_match.py --a config/default.conf --b config/BEN-Sayc.conf --n 64 --seed 1 --mode bidding --out /var/lib/ben/ab/sayc-64
```

In the `ben` container (from a Windows shell):

```
wslc exec -w /ben/scripts/match ben python ab_match.py --a config/default.conf --b config/BEN-Sayc.conf --n 64 --seed 1 --mode bidding --out /var/lib/ben/ab/sayc-64
```

| Option | Meaning |
|--------|---------|
| `--a`, `--b` | The two config files, written as for `game.py` (relative to `src`). |
| `--mode bidding` | Auction, opening lead, then the double dummy result. Fast. Default. |
| `--mode play` | Auction and all 13 tricks played by the engine. |
| `--boards FILE` | Board set: a PBN file (for example `Challenges/pav_first120.pbn`, relative to the repo root) or the `bidding/set*.txt` format. |
| `--n N --seed S` | Without `--boards`: deal N random boards from seed S. With `--boards`: use the first N. |
| `--out DIR` | Output folder, one per run. |
| `--opp-a FILE`, `--opp-b FILE` | What each side is told about its opponents: a `config/opponent/*.conf` file, applied as `game.py --opponent` does. Without it each side assumes the opponents its own config names. |
| `--workers W` | Parallel processes (default 4). Each loads both configs: about 2.5 GB in bidding mode, 3 to 4 GB in play mode. |
| `--dds-threads T` | Double dummy solver threads per worker (default 16). The configs' own default, a thread per core, costs about 6 GB per process on a 128 thread machine. |
| `--tf-threads T` | TensorFlow threads per worker (default 8). |
| `--resume` | Continue an interrupted run in the same `--out`. |
| `--matchpoint true/false` | Override the `matchpoint` setting of both configs. The result is still counted in IMPs. |
| `--gpu` | Let TensorFlow use the GPU. Default is the CPU, as the BEN servers do. |

The opponent to measure against is whatever `--b` is: another BEN model
(`config/BEN-Sayc.conf`, `config/BEN-21GF.conf`, `config/GIB-BBO.conf`), or
BBA/EPBot bidding with BEN card play (`config/BBA-Sayc.conf`,
`config/BBA-21GF.conf`). A robot that is not a BEN config (GIB itself,
WBridge5, ...) cannot sit at these tables; that needs a table manager
(`src/table_manager_client.py`).

## How long, and how many boards

Measured on the P620 (64 cores) on 2026-11-16, with `config/default.conf`:

| Mode | One deal at one table | 100 boards (200 deals), 4 workers |
|------|-----------------------|-----------------------------------|
| bidding | about 10 s | about 10 minutes |
| play | about 45 s | about 40 minutes |

More workers finish sooner, with shrinking returns: with 16 to 26 workers
running at once each deal took two to three times as long as with 2 to 4. The
`ben` container is limited to 24 GB, shared with the BEN servers and anything
else running in it, which is room for 4 to 6 workers.

The spread of the board results decides how many boards a comparison needs.
Measured: 3.6 to 4.5 IMPs per board in bidding mode, 5.4 in play mode.

| To see | Bidding mode | Play mode |
|--------|--------------|-----------|
| 95% interval of +-1 IMP/board | 50 to 80 boards | about 115 boards |
| a real 1 IMP/board difference, 4 times out of 5 | 100 to 160 boards | about 235 boards |
| a real 0.5 IMP/board difference, 4 times out of 5 | 400 to 640 boards | about 930 boards |

Two configs that differ little agree on most boards, so their spread is smaller
and they need fewer boards; `summary.txt` prints the figures for the run at hand.

## Output

| File | Content |
|------|---------|
| `summary.txt` | Boards played, IMPs for each side, IMPs per board with standard error and 95% interval, boards needed for a given precision, the largest swings. |
| `summary.json` | The same numbers, the configs' checksums, the git commit, and `results_sha1`. |
| `results.jsonl` | One line per board: both auctions, contracts, leads, tricks, North-South scores and `imp` (for A). A superset of what `src/compare.py` writes, so `bidding/printmatch.py` reads it. |
| `table1.jsonl`, `table2.jsonl` | Each table on its own. |
| `table1.pbn`, `table2.pbn` | Play mode only: the full deals with auction and play. |
| `boards.txt` | The board set that was used, in the `bidding/set*.txt` format. |
| `raw.jsonl`, `run.json`, `logs/` | Results as they arrived (what `--resume` reads), the command, and everything the engine printed. |

## How the two modes score

- `bidding`: `bidding/auction.py` bids the board, `bidding/lead.py` picks the
  opening lead, `bidding/score.py` solves the rest double dummy. The lead is
  the lead network's first choice, without the simulation the full engine runs,
  and it is made with the config of the side that is on lead. The card play
  settings and play models of the configs have no effect in this mode.
- `play`: `game.py`'s `Driver` bids and plays the board. Each seat uses the
  config of its side for bidding, the opening lead and the card play.

## Reading the result

- `IMPs per board` is the mean over boards; the interval is the mean plus or
  minus about two standard errors. A result whose interval includes zero has not
  shown a difference.
- The engine seeds its sampling from the cards, not from the clock, so the same
  configs and boards give the same `results_sha1`, whatever the number of
  workers. A config against itself therefore scores exactly zero. The noise of a
  comparison is which boards were dealt: use the same seed for runs you want to
  compare with each other, and more boards (or another seed) to check that a
  result holds.
- Card play in play mode asks PIMC for an answer within `pimc_wait` seconds (3 in
  the shipped configs). A search cut short by that limit could change a card. It
  did not happen in the runs of 2026-11-16, including on a saturated machine,
  but it is the one place where the clock can enter.
- The auction is the same in both modes; play mode adds the engine's lead and
  card play in place of the double dummy result. On 128 boards of
  `default.conf` against `BEN-Sayc.conf` the contracts were identical in the two
  modes and the board results correlated 0.67.
- If A and B bid different systems, each side samples the other's hands with the
  opponent model from its own config unless `--opp-a` / `--opp-b` say otherwise.

## The older scripts in `bidding/`

`ab_match.py` calls the functions in `bidding/auction.py`, `lead.py`,
`score.py`, `create_set.py` and `src/compare.py`. The step by step procedure in
`bidding/README.md` still works with these corrections:

- `match.sh` does not run as written (it calls `lead.py` without `--bidder`,
  expects `A.conf` and `B.conf`, and has a Windows path to `compare.py`).
- `auction.py` has no `--search` option; whether the bidding searches is set in
  the config (`search_threshold`).
- `match.cmd` and `vs.cmd` make every opening lead at a table with one config,
  whichever side is on lead.
- `src/compare.py` scores a board as 0 IMPs whenever the two auctions are the
  same, even when the leads and the double dummy results differ.
- `create_set.py` deals from an unseeded generator, so a set cannot be dealt
  again; keep the file, or use `ab_match.py --seed`.
