# Self-play improvement loop for the bidding network

A loop that lets a bidding network learn from the results of its own bidding, and that only
calls a new network better when a match on fresh deals shows it.

```
incumbent ──► explore ──► targets ──► fine-tune ──► gate 1 ──► gate 2 ──► real robots ──► promote
 (setting      self-play    best call    candidate     networks    whole      (by hand)      next generation
  that plays   with every   on the real  network       alone       settings                  starts from it
  now)         candidate    cards
               tried
```

One generation is one run of `run_generation.sh`. It never changes the incumbent; `promote.sh`
does, after the checks below.

## The idea

The network was trained to copy a bidding system. It cannot see what a call leads to. Self-play
gives it that:

1. **Explore** (`selfplay_gen.py --mode explore`). The learner sits South on random deals. North,
   East and West are a network that models the robots it will meet (default `Brill-21GF`), each
   making the call it rates highest. At every South turn the calls the learner's network also
   considers (output of 0.03 or more, at most four, plus the robots' network's own first choice
   with South's cards) are each tried: the auction is finished from there by the same players and
   the contract reached is scored double dummy **on the real deal**. The learner bids without
   seeing the other hands; the judging afterwards sees them. That is what makes the results
   cheap and exact for the deal: one double dummy table a deal, no sampling.
2. **Targets** (`selfplay_build.py games`). At a turn where the candidates' results differ, the
   target is the call with the best result, weighted by the IMPs at stake. One deal says little
   about a call; over many deals a network trained on these weighted targets comes to prefer,
   of two calls, the one with the better expected result. Turns where every candidate ends the
   same teach nothing and get weight 0.
3. **Fine-tune** (`selfplay_train.py`). The incumbent's network is trained on, at a small
   constant learning rate, with a mix of its original training boards (so it stays in its
   system: North has to understand it) and the self-play targets of this and all earlier
   generations.
4. **Gate 1** (`selfplay_gen.py --mode net`, `selfplay_compare.py`). Candidate and incumbent
   network each sit South at the same table on the same fresh deals, no search. Cheap, so it
   can use 100,000 deals.
5. **Gate 2** (`--mode search`). The two whole settings (network, candidate calls, sampling,
   rollouts, as in play) on the same deals. This is the result that counts: a network that is
   better alone can be no better once the search is on top.
6. **Real robots, then promote.** The table of the loop is a model. Before a candidate
   replaces the network that plays, bid a set of deals against the real robots with both
   settings (`../match/brill/brill_live.py`, scored by `score_live_ab.py`). Then
   `promote.sh <generation>` makes it the loop's incumbent and the next generation explores
   with it. Putting it into a server is a separate step.

A generation that fails a gate is not wasted: its targets stay and the next generation trains
on them too.

## Three ways from the explore games to a network (`METHOD`)

The explore games hold, for every South turn with candidates, each tried call and the result
it had. `METHOD` says how a network is made from that.

- **`targets`** (default, steps 2 and 3 above). The best call on each deal is the target.
  Simple, but one deal is one layout of the hidden cards, and the call that was best on it is
  often not the best call.
- **`value`** (`selfplay_value.py`). A value network first learns what every tried call is
  worth on average with this hand after this auction: the incumbent's network with its last
  layer replaced by one value a call, trained on the results of the tried calls against the
  call made. The deals are split into four parts and each part is rated by a network that was
  trained on the other three, so no turn is rated by a network that has seen its result. The
  highest-rated call of a turn is then the target, weighted by its lead over the next call,
  and the fine-tuning is as before. `value/value.json` says how good the ratings are on the
  unseen deals: what bidding the highest-rated call instead of the call made would have
  brought on the real cards.
- **`pg`** (`selfplay_train.py --pg`), policy gradient. No target is named. At each turn the
  loss is minus the expected result of the network's own choice among the tried calls (their
  results weighted by the output the network gives them). The network so moves output towards
  the calls that did better, in proportion to how much better and to the output they already
  have, and is not pulled anywhere on calls that were not tried. This is the exact gradient
  over South's choice at the turn: every tried call counts, not one sampled call. The turns of
  every 20th deal are kept out of the training and the log says after each epoch what the
  changed first choices brought on them.

`build` writes the turns once as a decision file (`selfplay_build.py decisions`, `rl.npz`) for
both.

**Sampled calls** (`SAMPLE=1`, explore). In the default explore South always makes its first
choice, so every turn it learns from lies on the first-choice line. A network that has changed
some calls reaches positions that line never shows. With `SAMPLE` the call made at a turn is
drawn among the network's own candidates, each in proportion to its output (to the power
1/`SAMPLE`), and the deal goes on from the call drawn. Every candidate of a turn, the call made
included, is then scored with first choices from there on. The sampling decides which
positions are visited; what is learnt at each still uses every tried call.

First figures on g001's 500,000 games (2026-11-20, deals not seen in training, one changed
call at a time): the value network's highest-rated call instead of the call made +0.25 IMPs a
deal (changed at 48% of the turns with candidates; +0.18 from the 12% of turns where it leads
by more than 1), against +1.60 for the best call in hindsight; its ratings explain 6% of the
spread of the results, the rest being the lie of the cards. After the policy-gradient
fine-tuning the network's first choice had changed at 11% of the turns kept out, for the
better on 1,034 and for the worse on 502. Whether the networks so made bid better is what the
gates say.

Full self-play, the learner in all four seats (`--table self`), is there as well. It is not the
default because a partnership that only ever plays with itself drifts into its own system, and
the learner's partner here is a fixed robot.

## Run

In a BEN container with a GPU, `SP_ROOT` (default `/var/lib/ben/selfplay`) holding

- `conf/incumbent.conf`: the whole setting that plays now. Its one `bidder =` line names the
  network to improve.
- `base.npz`: the network's original training set in compact form, made once with
  `python selfplay_build.py base <bin dir of bidding_binary_keras.py> $SP_ROOT/base.npz`.

```bash
bash run_generation.sh g001          # long runs: nohup setsid bash run_generation.sh g001 > /dev/null 2>&1 &
cat $SP_ROOT/gen/g001/verdict.txt
bash promote.sh g001                 # only after the check against the real robots
```

Every step is skipped when its result exists, so a stopped run continues when started again.
`run.log` in the generation's folder has the step times.

| Setting | Default | Meaning |
|---------|---------|---------|
| `N_EXPLORE` | 200000 | self-play deals |
| `SEED` | from the generation's name | seed of those deals |
| `ROBOTS` | `config/Brill-21GF.conf` | network of North, East and West |
| `ALSO` | the same | configs whose first choice is one more candidate |
| `CAND_MIN_P`, `CAND_MAX` | 0.03, 4 | which of the learner's calls are tried |
| `TABLE` | `brill` | `self`: the learner in all four seats |
| `WEIGHT` | `mix` | `imp`: weight = IMPs at stake; `mp`: every difference alike; `mix`: between |
| `METHOD` | `targets` | `value`: targets named by a value network; `pg`: policy gradient (see above) |
| `VALUE_ARGS` | none | `METHOD=value`: arguments of `selfplay_value.py` (`--folds 4 --epochs 6 --scale 3 --margin 0.25`) |
| `PG_COEF`, `PG_ANCHOR` | 2, 0 | `METHOD=pg`: weight of the expected result in the loss; pull towards the call made |
| `SAMPLE` | off | explore: draw the call made among the network's candidates (1 = in proportion to output) |
| `INJECT` | none | explore: calls tried whatever the network thinks: `double`, `compete`, `game`, `all:N` |
| `TARGET` | `best` | `METHOD=targets`: `pair` makes a target of every tried call against the call made |
| `EXPLORE_FROM`, `EXPLORE_ALSO` | none | learn from the explore games of another generation (name or folder) instead of, or as well as, new ones |
| `TRAIN_MODE` | `finetune` | `scratch`: a new network from random weights (`EPOCHS=30 LR=0.001`) |
| `EPOCHS`, `LR`, `SP_SHARE` | 3, 0.0001, 0.25 | fine-tuning; `SP_SHARE` is the share of self-play rows in an epoch |
| `PRIOR` | `auto` | targets of earlier generations to train on as well (all there are) |
| `N_GATE1`, `GATE1_SEED` | 100000, 900001 | gate 1 deals |
| `GATE2` | `auto` | gate 2 only after gate 1 says better; `always`; `never` |
| `GATE2_DEALS` / `N_GATE2`, `GATE2_SEED` | 10000 random deals | gate 2 deals: a file (the format of `brill_live.py`) or random |
| `WORKERS`, `SEARCH_WORKERS` | 3/4 of the processors, 40 | processes for the network modes and for search mode |
| `TEST_PBN` | none | held-out boards of the system: `system_check.txt` shows how far the candidate moved from it |

A setting that uses the auction model needs what it needs in play (`AUCTION_MODEL_CODE`,
PyTorch); that only matters for gate 2.

## Reading a verdict

`selfplay_compare.py` gives the mean IMPs a deal with its 95% interval, the same as a
percentage of deals won, and both by scoring and vulnerability. A gate says `better` only
when the interval lies above 0. "No difference shown" is the usual result of a change that is
too small for the number of deals; it is a reason to keep the incumbent, not a finding that
the two are equal.

The incumbent's gate runs are kept (`$SP_ROOT/gate/<checksum of its config>/`) and used again
by later generations, so the gate deals stay the same from generation to generation. After
many generations judged on the same deals, change `GATE1_SEED` and the gate 2 deals: a
candidate chosen among many on one set of deals looks better on that set than it is.

## What the loop cannot see

- The robots are a network that copies them. On 320 deals bid with the same South setting at
  both tables (2026-11-20) the copy made the real robots' call on 96.1% of their turns, and
  the whole auction was the same on 74% of the deals. A call that gains against the copy may
  not gain against the robots; a call the copy has never met is where the two differ most.
  Hence the check against the real robots. `selfplay_gen.py --table live` plays explore games
  against Brill's own robots instead (through `brill_live.py`'s cache and daily budget), and
  `refit_robots.py` fine-tunes the copy on the robots' saved answers. Neither has helped yet
  (2026-11-22): 3,000 live explore games gave targets no better than the copy's, and the
  refitted copy agrees more with the answers it was fitted on but not with unseen ones.
- Results are double dummy, of the contract reached. Leads, defence and declarer play are not
  in it. `play_model.py` learns real card play (tricks taken around the double dummy number)
  from played robot boards and rescores explore's candidates by their expected real-play score;
  `score_live_ab.py` scores the same way with `PLAY_MODEL` set. A model trained on rescored
  targets (2026-11-22) gained +0.04 ±0.04 IMPs a deal by that scoring and lost 0.03 by double
  dummy; Brill's own card play of 286 deals where its contract differed put it in between,
  with no difference shown.
- In explore, South's later calls on a tried line are the network's alone, without search.
- The targets come from South's seat with these robots. The network is not asked to bid well
  with a partner who plays something else.

## Files

| File | |
|------|---|
| `run_generation.sh`, `promote.sh` | one generation; make its candidate the incumbent |
| `selfplay_gen.py` | the table: explore, net and search modes |
| `selfplay_build.py` | targets, or the decision file, from explore games; the base set in compact form |
| `selfplay_value.py` | the value network, its figures on unseen deals, targets from it |
| `selfplay_train.py` | fine-tuning on base set plus targets, or by policy gradient |
| `selfplay_compare.py` | two runs on the same deals, with the verdict |
| `sp_common.py` | deals, IMPs, a bidding network as a plain policy |
| `refit_robots.py` | training rows from the real robots' answers; the copy's own calls as anchor; agreement on held-out answers |
| `play_model.py` | tricks in real play against double dummy: fit, check on held-out events, rescore explore games |

Speed on a 64-core machine with an RTX 5080 (2026-11-20): explore about 0.4 processor-seconds
a deal (500,000 deals in under an hour on 64 processes), fine-tuning under a minute an epoch,
gate 1 about 12 minutes a side for 100,000 deals, search mode about 7 processor-seconds a deal
(10,000 deals in about 35 minutes on 32 processes).
