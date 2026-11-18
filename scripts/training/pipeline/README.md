# Training pipeline for the bidding networks

Two scripts that take a boards PBN (for example an `OK_boards` file from the `training-data`
release) to a trained and evaluated pair of networks, with a held-out test set.

```bash
python prep_split.py <boards.pbn> <run folder>      # train.pbn, test.pbn, sample.pbn (+ double dummy tables)
bash run_baseline.sh <run name>                      # conversion, both trainers, evaluation
```

`run_baseline.sh` works in `$TRAIN_ROOT/<run name>` (default `/var/lib/ben/training`) and runs
the scripts in `../bidding` and `../bidding_info`. Start it again after a stop and it continues.
It ends by writing `done` or `failed` in the run folder; `run.log` has the step times,
`eval.txt` the evaluation of the new model and of a shipped model on the held-out boards
(`test.pbn`) and on a sample of the trained boards (`sample.pbn`).

| Setting | Default | Meaning |
|---------|---------|---------|
| `BID_ARGS` | `lstm_size=256 decay_steps=12000 keep=0.95` | arguments of `bidding_nn_keras.py` (batch 512 and learning rate 0.001 are its defaults) |
| `EPOCHS` | 30 | epochs of both networks |
| `TRAIN_ROOT` | `/var/lib/ben/training` | where the run folders are |
| `SHIPPED` | `models/TF2models/GIB-BBO-8730_2025-04-19-E30.keras` | the model evaluated next to the new one |

`prep_split.py` takes every k-th board as the held-out test set, leaves out training boards
with the same four hands as a test board, and adds `[OptimumResultTable]` to `test.pbn` and
`sample.pbn`, which `eval_bidding_nn.py` needs.

On a 64-core machine with an RTX 5080 (2026-11-17), 1.15 million boards: conversion 50 s,
both networks 30 epochs about 16 minutes, evaluation 80 s. The default recipe scored 95.6%
same call and 66.2% identical auctions against the boards' own (EPBot) auctions on 20,253
held-out boards; the trainers' own defaults (`decay_steps=1000`, `keep=0.8`) are unchanged.

## Recipe runs of the info, play and lead networks

Three queue drivers train one model per line of a queue file and score it on held-out data.
Each appends a line per run to `results.txt` in its output folder, skips names that are
finished and reads the queue again after every run, so lines can be added while it runs.

```bash
bash info_hp.sh <train bin dir> <held-out bin dir> <out dir> <queue file>      # name | arguments
bash play_hp.sh <binary dir> <out dir> <queue file>                             # name | hand | arguments
bash lead_hp.sh <lead data dir> <WBC play.txt> <Jack dir> <out dir> <queue file>  # name | suit or nt | arguments
```

| Network | Trainer | Score | Held-out data |
|---------|---------|-------|---------------|
| bidding info | `../bidding_info/binfo_nn_keras.py` | `eval_binfo_nn.py`: mean error in points and in cards per suit of the three hidden hands, during the bidding and after it | `test.pbn` of a `prep_split.py` run, converted with `max_occurrences=1000000` |
| card play | `../playing/play_nn_keras.py` | `eval_play_nn.py`: share of cards the model has on top, and `nll` of the card played | the last 5% of the deals of each binary file |
| opening lead | `../opening lead/lead_train.py` | `lead_eval.py`: the same two for the card led, plus cover, DDOLAR and ADDOLAR from double dummy | the last 5% of the rows of each data folder |

`../bidding_info/eval_binfo_sampling.py` measures what the info network is for: with each info
model BEN deals the same samples at the same decision points, and the deals that fit the
auction are counted.

The runs of 2026-11-17 and 2026-11-18 (RTX 5080; the numbers are from held-out data):

**Bidding info**, 1.15 million boards, 20,253 held out. Error during the bidding in points /
cards per suit; without any information it is 3.397 / 1.033.

| Arguments | Points | Cards |
|-----------|--------|-------|
| defaults (`lstm_size=128 keep=0.8 learning_rate=0.0005`), three runs | 1.804 to 1.806 | 0.7909 to 0.7922 |
| `keep=0.95` | 1.791 | 0.7836 |
| `keep=1.0` | 1.789 | 0.7825 |
| `keep=0.95 learning_rate=0.001 schedule=cosine` | 1.784 | 0.7797 |
| `keep=0.95 lstm_size=256 learning_rate=0.001 schedule=cosine`, two runs | 1.773, 1.774 | 0.7696 |
| the same with `lstm_size=384` | 1.774 | 0.7667 |
| the same with `lstm_size=512` | 1.778 | 0.7657 |
| the same with `lstm_size=256 n_layers=4` | 1.770 | 0.7663 |
| the same with `lstm_size=256 epochs=60` | 1.775 | 0.7690 |

With `eval_binfo_sampling.py` (600 decision points, 2,000 samples each) the default recipe
gave 150.5 samples that fit the auction, `keep=0.95 lstm_size=256 learning_rate=0.001
schedule=cosine` 3.0 more (standard error 0.7).

**Card play**, declarer in suit contracts, 1.21 million deals, 63,909 held out. The shipped
`decl_suit_hybrid_2026-10-16.keras` scores 0.7684 / 0.5943. All runs with
`dense_layers=2 dense_units=512 lstm_size=256 n_layers=2`, as the shipped model has.

| Arguments | Epochs run | Top card | nll |
|-----------|-----------|----------|-----|
| defaults (`keep=0.8 epochs=30`) | 30 | 0.7646 | 0.6045 |
| `keep=0.9` | 30 | 0.7683 | 0.5944 |
| `keep=0.95` | 30 | 0.7691 | 0.5970 |
| `keep=1.0` | 19 | 0.7568 | 0.6281 |
| `keep=0.95 schedule=cosine epochs=60` | 60 | 0.7681 | 0.6013 |
| `keep=0.8 epochs=60` | 58 | 0.7710 | 0.5864 |
| `keep=0.85 epochs=60` | 56 | 0.7747 | 0.5785 |
| `keep=0.9 epochs=60` | 46 | 0.7744 | 0.5794 |
| `keep=0.95 epochs=60` | 33 | 0.7696 | 0.5953 |
| `keep=0.9 epochs=60 dense_units=768 lstm_size=384` | 25 | 0.7715 | 0.5901 |

With `epochs=30` training is cut off before the learning rate schedule has finished. The best
dropout depends on the number of deals: `keep=0.9` for the suit seats (1.28 million deals),
`keep=0.8` for the NT seats (608,000), where `keep=0.9` scores below the shipped models.
All eight seats with `epochs=60`, top card / nll:

| Seat | Shipped `*_hybrid_2026-10-16` | Retrained | `keep` |
|------|-------------------------------|-----------|--------|
| `decl_suit` | 0.7684 / 0.5943 | 0.7744 / 0.5794 | 0.9 |
| `lefty_suit` | 0.7752 / 0.5483 | 0.7766 / 0.5468 | 0.9 |
| `righty_suit` | 0.8068 / 0.4655 | 0.8074 / 0.4662 | 0.9 |
| `dummy_suit` | 0.7954 / 0.5075 | 0.7954 / 0.5088 | 0.9 |
| `decl_nt` | 0.7851 / 0.5488 | 0.7852 / 0.5473 | 0.8 |
| `lefty_nt` | 0.7495 / 0.6154 | 0.7510 / 0.6114 | 0.8 |
| `righty_nt` | 0.7838 / 0.5191 | 0.7867 / 0.5119 | 0.8 |
| `dummy_nt` | 0.8098 / 0.4787 | 0.8122 / 0.4717 | 0.8 |

A match of 160 boards with `play_nn_only = True` (`../../match/ab_match.py --mode play`),
the four retrained suit networks against the shipped ones, ended +0.11 IMPs per board for the
retrained (95% interval -0.20 to +0.41): no difference shown.

**Opening lead**, WBC and Jack data together. `nll` of the card led on the WBC / Jack held-out
rows; the shipped `LeadV3-Suit-Both_2026-10-16.keras` scores 1.205 / 0.972. Cover, DDOLAR and
ADDOLAR were the same to 0.003 in every run.

| Arguments (suit network) | WBC | Jack |
|--------------------------|-----|------|
| defaults (`layers=3 units=512 keep=0.7`), two runs | 1.206, 1.201 | 0.974, 0.972 |
| `keep=0.8` | 1.200 | 0.967 |
| `keep=0.9` | 1.208 | 0.969 |
| `keep=0.5` | 1.239 | 1.007 |
| `units=1024` | 1.202 | 0.976 |
| `layers=4` | 1.213 | 0.980 |
| `layers=2` | 1.204 | 0.981 |

| Arguments (NT network; shipped 1.349 / 1.022) | WBC | Jack |
|-----------------------------------------------|-----|------|
| defaults, two runs | 1.345, 1.347 | 1.019, 1.027 |
| `keep=0.8` | 1.349 | 1.012 |
| `units=1024` | 1.355 | 1.013 |

No setting of the lead trainer beats its defaults by more than two runs of the defaults
differ.
