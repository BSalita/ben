# Opening leads weighted by the belief model (2026-11-25)

**Answer: no.** Weighting the opening lead's sampled layouts by the belief model, the way the defence's are weighted,
changes the lead on 15% of boards and gains nothing: over 4,410 boards (1,410 Brill, 3,000 held-out ACBL), declarer
**+0.003 ±0.008 double-dummy tricks, +0.022 ±0.030 IMPs a board** (positive = declarer gains, so the defence if
anything loses). A corrected `ratio` weighting also gains nothing (−0.000 ±0.026 IMPs). Cost: +0.2 s a lead (+1.5%).
Real play on the 188 changed Brill boards agrees: at normal load, declarer +0.32 ±0.58 IMPs a changed board, +0.043
±0.077 IMPs a board over all 1,410. Not deployed, and not to be deployed.

## Question

The belief model (a 25M-parameter transformer, run `full-d512`) improved Morty's defence when it weights the
defenders' sampled layouts (−0.045 ±0.036 declarer tricks, −0.198 ±0.129 IMPs a board on 1,410 Brill boards). Does
the opening lead gain from the same weighting?

## How BEN chooses the opening lead

`botopeninglead.py`:
1. The lead networks give candidates: at least 4 cards (`min_opening_leads`), more while their score is above
   `lead_threshold` (0.20). The network's choice is taken alone only above `lead_accept_nn` (0.999).
2. `sample.generate_samples_iterative` draws layouts of the three hidden hands that fit the auction, until about
   200 pass the bid check (`sample_hands_opening_lead`; median 163 to 200 found). If more pass, 200 are drawn in
   proportion to their bid score.
3. Each candidate is solved double dummy on every layout. With `use_real_imp_or_mp_opening_lead` (on in
   `default_api.conf`), the candidates are compared by IMPs against each other, layout by layout, and averaged;
   a fixed partner-suit and trump-honour adjustment is added.

The defence's belief weighting (branch `belief-play`: `play_weighted_seats`, `belief_weights.layout_weights_log`)
lives in the play sampler (`init_rollout_states_iterative`) and does not touch the lead.

The belief model's first query (slot `S_START`) reads only the vulnerability, the calls and the leader's 13 cards;
the lead and dummy come after it in the model's causal order. It was trained on that query from every seat, the
leader's included (viewer drawn among declarer and both defenders, loss on every query).

## What was built

In a private copy of the `belief-play` code, new `[sampling]` keys, all off by default:

| Key | Meaning |
|---|---|
| `lead_belief_model` | model file (.npz numpy or .pt torch, as `play_belief_model`); empty = off |
| `lead_belief_code` | folder of `belief.py` (torch only) |
| `lead_belief_weight` | β, scales the log-weight |
| `lead_belief_mode` | `replace`: weight = exp(β · model log-likelihood of the layout); `ratio`: exp(β · (model log-likelihood − the layouts' own log-frequency)) |

- `belief_weights.py`:
  - `lead_probabilities`: the model's [52, 4] beliefs before the lead;
  - `lead_layout_loglik`: honours one by one; a suit's small cards by the probability of the three hands'
    counts (a product of trinomials);
  - `layouts_own_loglik`: the layouts' own card frequencies, smoothed;
  - `lead_layout_weights`.
- `botopeninglead.py`: the same layouts are drawn and solved; only the averages over them are weighted
  (`calculate_imp_score(real_scores, weights)`, `np.average` in the trick-based branch).
- The log-likelihood matches brute-force enumeration to 1e-14 on three random deals.

The code is not committed: it gains nothing.

### Off = identical

100 Brill defence boards played in full (`play_ab.py`, belief defence on, both sides). The copy with the keys off was
played against an untouched copy of the `belief-play` source, 10 workers each:
- 100/100 same lead, 100/100 same tricks, 99/100 same card sequences;
- the one board that differs (card 28, mid-play) plays differently on every run, even with the same code: replayed
  alone, the reference copy differed from its own first run. It is PIMC's 3 s clock, not the change.

Leads have no clock. The off setting gave the same lead on all 1,410 Brill boards in two separate screens.

## Lead-only screens

Each board is run up to the opening lead under each setting. The seeds are the same, so the layouts are the same.
The lead is scored double dummy on the real deal (declarer's tricks after it). Results are paired against today's
lead, ± 95%; positive = declarer gains.

| Setting | Boards | Leads that change | Declarer DD tricks a board | Declaring side IMPs a board |
|---|---|---|---|---|
| `replace` β 1 (the defence's way), Brill | 1,410 | 13.3% | +0.008 ±0.013 | +0.044 ±0.053 |
| `replace` β 1, ACBL held out | 3,000 | 15.9% | +0.001 ±0.010 | +0.012 ±0.036 |
| **`replace` β 1, pooled** | 4,410 | 15.1% | **+0.003 ±0.008** | **+0.022 ±0.030** |
| `replace` β 0.5, Brill | 1,410 | 7.3% | +0.003 ±0.009 | +0.007 ±0.028 |
| `ratio` β 1, Brill | 1,410 | 18.4% | +0.000 ±0.015 | +0.001 ±0.054 |
| `ratio` β 0.5, Brill | 1,410 | 9.9% | −0.007 ±0.011 | −0.036 ±0.039 |
| `ratio` β 0.5, ACBL held out | 3,000 | 11.4% | +0.005 ±0.008 | +0.016 ±0.034 |
| `ratio` β 0.5, pooled | 4,410 | 10.9% | +0.001 ±0.006 | −0.000 ±0.026 |

- Among the changed leads, the defence does better double dummy about as often as worse: 68 against 67 boards on
  ACBL, 26 against 29 on Brill (`replace` β 1).
- `ratio` β 0.5 was the only setting leaning towards the defence on Brill. It was picked after looking at four
  settings, so ACBL was its confirmation, and ACBL does not confirm it.
- Time: one belief query a lead; +0.2 ±0.2 s on a lead of about 15 s (+1.5%). The leads took 7 s on an idle
  machine and 19 s under heavy load.

### Why: the weighted layouts believe worse

Brier score of honour placement (A to T, each hidden hand) at trick 0, against the real deal:

| | Brill (1,360 boards with layouts) | ACBL (2,836) |
|---|---|---|
| BEN's layouts, equal (today) | 0.2098 ±0.0016 | 0.2128 ±0.0014 |
| `replace` β 1 | 0.2203 (worse by 0.0106 ±0.0010) | 0.2243 (worse by 0.0115 ±0.0009) |
| `replace` β 0.5 | 0.2131 (worse by 0.0034) | — |
| `ratio` β 1 | 0.2204 (worse by 0.0107) | — |
| `ratio` β 0.5 | 0.2099 (same) | 0.2125 (same, −0.0003 ±0.0005) |
| **the belief model's own marginals** | **0.2035** (better by 0.0063 ±0.0012) | **0.2015** (better by 0.0113 ±0.0012) |

- BEN's layouts were drawn to fit the auction already. Multiplying them by a model that also reads the auction
  counts the auction twice, so the layouts become more confident than either source. Effective layouts: median 42
  to 46 of about 200 at β 1, under 10 on a quarter of the boards.
- In the defence this matters less: later in the play the cards seen dominate both sources.
- `ratio` divides by the layouts' own card frequencies, to pull them towards the model's marginals. With about 200
  layouts and 39 unseen cards those frequencies are too noisy: at β 1 it is as bad as `replace`, and at β 0.5 it
  only matches today.
- The model really does know more than BEN's layouts at the lead, most on human (ACBL) auctions: 0.0113 better.
  Reweighting a pool of about 200 layouts cannot carry that over.

## Real play on the changed boards (Brill, `replace` β 1)

The 188 Brill boards whose lead changes, played out with `play_ab.py` in both arms. In both arms the defence after
the lead uses the belief model and declarer plays as today. The other 1,222 boards have the same lead and the same
code, so their difference is zero except for clock noise.

First run, under heavy load (load about 210 on 128 threads, 181 to 186 s a board, both arms alike):

| | Changed boards (187 paired) | Over all 1,410 boards |
|---|---|---|
| Declarer's real tricks | +0.080 ±0.155 | +0.011 ±0.021 |
| Declaring side IMPs | +0.38 ±0.59 | +0.050 ±0.078 |
| Double dummy, same boards (screen) | +0.059 ±0.098 tricks | |
| Contracts made | +5.3 ±6.3% (base 58.3%) | |

Replayed at normal load (7 workers, 44 to 47 s a board), all 188 changed boards:

| | Changed boards (188) | Over all 1,410 boards |
|---|---|---|
| Declarer's real tricks | +0.069 ±0.153 | +0.009 ±0.020 |
| Declaring side IMPs | +0.32 ±0.58 | **+0.043 ±0.077** |
| Contracts made | +4.8 ±6.1% (base 58.5%) | |

- Real play agrees with double dummy: no gain, and the direction slightly favours declarer.
- The lead played in each arm is the screen's lead on every board.
- About half the changed boards end with different tricks. With so few boards the interval is wide, but nothing
  points to a gain that double dummy misses.

`ratio` β 0.5 was not played out: ACBL did not confirm its Brill lean.

## What would be needed for a lead gain

The belief model's lead-time beliefs are better than BEN's layouts. Using them would mean drawing the layouts from
the model itself (sample card by card from its marginals under the 13-card and shape constraints, or many more
candidate layouts to reweight), not reweighting the 200 the bid sampler found. That is a new sampler, not a
setting. Its ceiling is unknown. On this data the lead is already at field level.
