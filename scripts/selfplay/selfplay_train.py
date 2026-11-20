#!/usr/bin/env python3
"""Fine-tune a bidding network on self-play targets, anchored by its original training set.

    python selfplay_train.py --start MODEL.keras --base base.npz --sp sp.npz [--sp more.npz] --out DIR --name NAME

Starts from the weights of --start and trains on a mix of the rows of --base (the calls of
the system the network was trained on, weight 1) and the self-play rows of --sp
(selfplay_build.py: the call with the best result, weighted by the IMPs at stake). The
self-play rows are repeated so that they make up --sp-share of every epoch. The loss is the
trainer's categorical cross-entropy; a target of weight w counts w times, weight 0 not at all.

The learning rate is constant and small (--lr): the network keeps what it knows and moves
where the self-play results say so. Models are saved after every epoch as
DIR/model/NAME_<date>-E<epoch>.keras, in the form the engine loads.

With --scratch no network is started from: one of the trainer's form (bidding_nn_keras.py:
--layers LSTM layers of --lstm-size, dropout 1 - --keep) is trained from random weights with
the trainer's learning rate schedule (--lr times 0.96 every --decay-steps batches), for
example --scratch --epochs 30 --lr 0.001. Without --sp that is a plain retrain of the base
set, the control that shows what the self-play rows add to a retrain.

Policy gradient (--pg rl.npz, a decision file of selfplay_build.py decisions): no target call
is named. At each turn of the file the network is asked for the expected result of its own
choice among the calls that were tried there: the sum of output times utility over the tried
calls, divided by the output they hold together (utility: sp_common.utilities of a call's
double dummy result against the call made, --pg-weight). The loss is minus --pg-coef times
that, so the network moves output towards calls in proportion to how much better they did
and to the output they already have: the exact gradient of the expected result over the
learner's choice at that turn, with every tried call counted, not one sampled call. Calls
that were not tried are neither pulled nor pushed. --pg-anchor adds a cross-entropy pull of
that weight to the call made, against moving on little evidence. The turns of every
--pg-holdout-th deal are kept out of the training; after each epoch the log says, on them,
the expected utility before and now, how many first choices among the tried calls changed
and the IMPs a deal those changes made on the real cards. --pg and --sp can be combined.

Needs a GPU (the data sits on it, as in scripts/training/gpu_epochs.py).
"""
import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, optimizers  # noqa: E402
from tensorflow.keras.callbacks import Callback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp_common  # noqa: E402
sys.path.insert(0, os.path.join(sp_common.BEN_ROOT, 'scripts', 'training'))
import gpu_epochs  # noqa: E402

N_BIDS = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--start', help='the network to start from (.keras)')
    parser.add_argument('--scratch', action='store_true', help='train a new network from random weights instead')
    parser.add_argument('--lstm-size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--keep', type=float, default=0.95)
    parser.add_argument('--decay-steps', type=int, default=12000)
    parser.add_argument('--base', required=True, help='base.npz of selfplay_build.py base')
    parser.add_argument('--sp', action='append', default=[], help='sp.npz of selfplay_build.py games; may be repeated')
    parser.add_argument('--out', required=True)
    parser.add_argument('--name', required=True, help="model name; must not contain '-E'")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--sp-share', type=float, default=0.25, help='share of self-play rows in an epoch')
    parser.add_argument('--pg', action='append', default=[], help='decision file for the policy-gradient loss; may be repeated')
    parser.add_argument('--pg-coef', type=float, default=2.0, help='weight of the expected utility in the loss')
    parser.add_argument('--pg-weight', choices=['imp', 'mp', 'mix'], default='mix', help='the utility a result counts for')
    parser.add_argument('--pg-anchor', type=float, default=0.0, help='cross-entropy weight of the call made at a policy-gradient turn')
    parser.add_argument('--pg-holdout', type=int, default=20, help='keep the turns of every N-th deal out of the training (0 = none)')
    parser.add_argument('--base-rows', type=int, default=0, help='use only this many base rows, spread evenly (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if bool(args.start) == args.scratch:
        raise SystemExit('Give --start MODEL or --scratch.')
    if not args.sp and not args.pg and not args.scratch:
        raise SystemExit('Fine-tuning needs --sp or --pg.')
    if '-E' in args.name:
        raise SystemExit("--name must not contain '-E' (the epoch is read from the file name after it)")
    if not tf.config.list_physical_devices('GPU'):
        raise SystemExit('No GPU visible to TensorFlow.')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    base = np.load(args.base)
    static, bids, cls = base['static'], base['bids'], base['cls']
    if args.base_rows and args.base_rows < len(static):
        keep = np.linspace(0, len(static) - 1, args.base_rows).astype(np.int64)
        static, bids, cls = static[keep], bids[keep], cls[keep]
    n_base = len(static)
    parts = [(static, bids, cls, np.ones(cls.shape, np.uint8))]
    for path in args.sp:
        sp = np.load(path)
        if sp['static'].shape[1] != static.shape[1]:
            raise SystemExit(f'{path}: {sp["static"].shape[1]} static features, the base set has {static.shape[1]}')
        parts.append((sp['static'], sp['bids'], sp['cls'], sp['wgt']))
    pg = held = None
    if args.pg:
        d = sp_common.load_decisions(args.pg)
        if d['static'].shape[1] != static.shape[1]:
            raise SystemExit(f'{args.pg}: {d["static"].shape[1]} static features, the base set has {static.shape[1]}')
        width = d['cand'].shape[1]
        valid = np.arange(width)[None, :] < d['n'][:, None]
        diff = np.where(valid, d['score'] - d['score'][:, :1], 0)
        d['valid'], d['u'] = valid.astype(np.float32), sp_common.utilities(diff, args.pg_weight)
        out = d['deal'] % args.pg_holdout == 0 if args.pg_holdout else np.zeros(len(valid), bool)
        keys = ('static', 'bids', 'step', 'cand', 'score', 'valid', 'u', 'deal')
        pg = {k: d[k][~out] for k in keys}
        held = {k: d[k][out] for k in keys} if out.any() else None
        pg_cls = np.zeros((len(pg['step']), bids.shape[1]), np.uint8)
        pg_wgt = np.zeros((len(pg['step']), bids.shape[1]), np.float32)
        pg_cls[np.arange(len(pg_cls)), pg['step']] = pg['cand'][:, 0]
        pg_wgt[np.arange(len(pg_cls)), pg['step']] = args.pg_anchor
        parts.append((pg['static'], pg['bids'], pg_cls, pg_wgt))
    static, bids, cls, wgt = (np.concatenate([p[k] for p in parts]) for k in range(4))
    n_sp = len(static) - n_base
    repeat = max(1, round(args.sp_share / (1 - args.sp_share) * n_base / n_sp)) if n_sp else 0
    rows = np.concatenate([np.arange(n_base), np.tile(np.arange(n_base, n_base + n_sp), repeat)]).astype(np.int32)
    n_examples = len(rows)
    n_rounds = bids.shape[1]
    steps_per_epoch = n_examples // args.batch
    print(f'{n_base} base rows, {n_sp} self-play rows repeated {repeat} times '
          f'({100 * repeat * n_sp / n_examples:.1f}% of an epoch), {steps_per_epoch} steps an epoch, '
          f'{args.epochs} epochs at learning rate {args.lr}' + (', from random weights' if args.scratch else ''), flush=True)
    if pg:
        print(f'policy gradient: {len(pg["step"])} turns among the self-play rows, coefficient {args.pg_coef}, utility '
              f'{args.pg_weight}, anchor {args.pg_anchor}; {len(held["step"]) if held else 0} turns of '
              f'{len(range(0, int(d["deals"]), args.pg_holdout)) if held else 0} deals kept out', flush=True)

    if args.scratch:
        # The network of bidding_nn_keras.py, with its seed and its learning rate schedule
        from tensorflow.keras import initializers, models
        inputs = tf.keras.Input(shape=(None, static.shape[1] + 4 * N_BIDS), dtype=tf.float16)
        x = inputs
        for _ in range(args.layers):
            x = layers.LSTM(units=args.lstm_size, return_sequences=True,
                            kernel_initializer=initializers.GlorotUniform(seed=args.seed))(x)
            x = layers.Dropout(1 - args.keep)(x)
            x = layers.BatchNormalization()(x)
        outputs = layers.TimeDistributed(layers.Dense(N_BIDS, activation='softmax'), name='bid_output')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        rate = optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr, decay_steps=args.decay_steps,
                                                     decay_rate=0.96, staircase=True)
    else:
        model = tf.keras.models.load_model(args.start)
        if len(model.outputs) != 1:
            raise SystemExit(f'{args.start} has {len(model.outputs)} outputs; this script trains the single bid output')
        rate = args.lr
    def pg_loss(y, p):
        """Cross-entropy on the named targets minus the expected utility among the tried calls; y = targets, utilities, mask."""
        p = tf.cast(p, tf.float32)
        utility, mask = y[..., N_BIDS:2 * N_BIDS], y[..., 2 * N_BIDS:]
        expected = tf.reduce_sum(mask * p * utility, axis=-1) / (tf.reduce_sum(mask * p, axis=-1) + 1e-7)
        return tf.keras.losses.categorical_crossentropy(y[..., :N_BIDS], p) - args.pg_coef * expected

    model.compile(optimizer=optimizers.Adam(learning_rate=rate), loss=pg_loss if pg else 'categorical_crossentropy')
    # As bidding_nn_keras.py: unrolled LSTM layers and an XLA-compiled step for training, the
    # layers switched back while saving so that the saved model takes auctions of any length.
    lstm_layers = [layer for layer in model.layers if isinstance(layer, layers.LSTM)]

    def set_unroll(value):
        for layer in lstm_layers:
            layer.unroll = value
            layer.supports_jit = value

    set_unroll(True)
    model.jit_compile = True

    with tf.device('/GPU:0'):
        g_rows = tf.constant(rows)
        g_static = tf.constant(static)
        g_bids = tf.constant(bids)
        g_cls = tf.constant(cls)
        g_wgt = tf.constant(wgt)
        if pg:      # a row's policy-gradient turn: 0 = none (an empty row), else 1 + its number in the decision arrays
            first = len(static) - len(pg['step'])
            turn = np.zeros(len(static), np.int32)
            turn[first:] = 1 + np.arange(len(pg['step']))
            lead = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), a])  # noqa: E731
            g_turn = tf.constant(turn)
            g_pstep = tf.constant(lead(pg['step'].astype(np.int32)))
            g_pcand = tf.constant(lead(pg['cand'].astype(np.int32)))
            g_pu = tf.constant(lead(pg['u']))
            g_pvalid = tf.constant(lead(pg['valid']))

    def batch(idx):
        r = tf.gather(g_rows, idx)
        s = tf.gather(g_static, r)
        b = tf.one_hot(tf.gather(g_bids, r), N_BIDS, dtype=tf.float16)
        x = tf.concat([tf.repeat(s[:, None, :], n_rounds, axis=1),
                       tf.reshape(b, (args.batch, n_rounds, 4 * N_BIDS))], axis=2)
        y = tf.one_hot(tf.gather(g_cls, r), N_BIDS, dtype=tf.float32)
        y = y * tf.cast(tf.gather(g_wgt, r), tf.float32)[:, :, None]
        if not pg:
            return x, y
        j = tf.gather(g_turn, r)
        calls = tf.one_hot(tf.gather(g_pcand, j), N_BIDS, dtype=tf.float32)                 # (batch, tried calls, 40)
        v = tf.gather(g_pvalid, j)
        utility = tf.reduce_sum(calls * (tf.gather(g_pu, j) * v)[:, :, None], axis=1)
        mask = tf.reduce_sum(calls * v[:, :, None], axis=1)
        at = tf.one_hot(tf.gather(g_pstep, j), n_rounds, dtype=tf.float32)
        return x, tf.concat([y, at[:, :, None] * tf.concat([utility, mask], axis=1)[:, None, :]], axis=2)

    validation = None
    if held:
        with tf.device('/GPU:0'):
            h_static = tf.constant(held['static'])
            h_bids = tf.constant(held['bids'])
            h_step = tf.constant(held['step'].astype(np.int32))
            h_cand = tf.constant(held['cand'].astype(np.int32))
        held_deals = len(range(0, int(d['deals']), args.pg_holdout))      # all deals kept out, with or without a turn
        h_valid = held['valid'] > 0

        @tf.function
        def outputs_at(r):
            n = tf.shape(r)[0]
            s = tf.gather(h_static, r)
            b = tf.one_hot(tf.gather(h_bids, r), N_BIDS, dtype=tf.float16)
            x = tf.concat([tf.repeat(s[:, None, :], n_rounds, axis=1), tf.reshape(b, (n, n_rounds, 4 * N_BIDS))], axis=2)
            p = tf.gather(tf.cast(model(x, training=False), tf.float32), tf.gather(h_step, r), axis=1, batch_dims=1)
            return tf.gather(p, tf.gather(h_cand, r), axis=1, batch_dims=1)

        def held_state():
            """On the turns kept out: (expected utility among the tried calls, first choice among them)."""
            rows = np.arange(len(h_valid), dtype=np.int32)
            p = np.concatenate([outputs_at(tf.constant(rows[i:i + 4096])).numpy() for i in range(0, len(rows), 4096)])
            p = np.where(h_valid, p, 0)
            expected = float(((p * held['u']).sum(axis=1) / (p.sum(axis=1) + 1e-7)).mean())
            return expected, np.where(h_valid, p, -1).argmax(axis=1)

        before = held_state()
        print(f'turns kept out, before: expected utility {before[0]:+.4f}', flush=True)

        def validation():
            expected, first = held_state()
            on = first != before[1]
            rows = np.arange(len(first))
            gain = sp_common.imps_array(held['score'][rows, first] - held['score'][rows, before[1]])
            logs = {'val_expected_utility': expected, 'val_expected_utility_before': before[0],
                    'val_first_choice_changed': float(on.mean()), 'val_imps_a_deal': float(gain.sum()) / held_deals,
                    'val_changes_better': float((gain > 0).sum()), 'val_changes_worse': float((gain < 0).sum())}
            print(f'turns kept out: expected utility {expected:+.4f} (before {before[0]:+.4f}); first choice among the tried '
                  f'calls changed at {100 * on.mean():.1f}% of turns, better {int((gain > 0).sum())}, worse '
                  f'{int((gain < 0).sum())}, {logs["val_imps_a_deal"]:+.4f} IMPs a deal', flush=True)
            return logs

    os.makedirs(os.path.join(args.out, 'model'), exist_ok=True)
    stem = os.path.join(args.out, 'model', f'{args.name}_{datetime.datetime.now():%Y-%m-%d}')
    history = []

    class Save(Callback):
        def on_epoch_end(self, epoch, logs=None):
            path = f'{stem}-E{epoch + 1:02d}.keras'
            set_unroll(False)
            try:
                model.save(path)
            finally:
                set_unroll(True)
            history.append({'epoch': epoch + 1, 'model': path, **(logs or {})})
            print(f'Saved {path}  {logs}', flush=True)

    t0 = time.time()
    gpu_epochs.fit_batches(model, batch, n_examples, args.batch, args.epochs, steps_per_epoch, callbacks=[Save()],
                           validation=validation)
    with open(os.path.join(args.out, 'train.json'), 'w') as f:
        json.dump({'args': vars(args), 'base_rows': n_base, 'sp_rows': n_sp, 'sp_repeat': repeat,
                   'pg_turns': len(pg['step']) if pg else 0, 'pg_turns_kept_out': len(held['step']) if held else 0,
                   'seconds': round(time.time() - t0, 1), 'epochs': history}, f, indent=1)
    print(f'Done in {time.time() - t0:.0f} s. Model: {history[-1]["model"]}')


if __name__ == '__main__':
    main()
