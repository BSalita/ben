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
