#!/usr/bin/env python3
"""Value network: what each tried call is worth, learnt from the explore games; and training targets from it.

    python selfplay_value.py --start INCUMBENT.keras --rl rl.npz [--rl more.npz] --out DIR

The targets of selfplay_build.py games say "the call with the best result on this deal". One
deal is one layout of the hidden cards: the call that was best on it need not be the best
call. The value network learns, from many deals, what a call is worth on average with this
hand after this auction: for every tried call of a turn, the utility of its double dummy
result against the result of the call made (sp_common.utilities: --weight imp, mp or mix; the
call made counts 0). It is the incumbent's network with its last layer replaced by one value
for each of the 40 calls, trained by squared error on the tried calls only.

A network asked about the deals it was trained on would simply remember their results. So the
deals are split into --folds parts, and each part is rated by a network trained on the other
parts. Everything reported and everything written comes from those unseen ratings:

  value.json  how good the ratings are: the squared error against always answering 0, and
              what would have happened on the real cards had the learner made the call rated
              highest, wherever that call beats the call made by more than a margin (turns
              changed, IMPs a changed turn, IMPs a deal). One change at a time: the rest of
              the auction is as it was. For comparison, the best call in hindsight.
  sp.npz      training targets in the form of selfplay_build.py games, for selfplay_train.py:
              at each turn the call rated highest, weighted by --scale times its lead over
              the next call (at most --max-weight); turns with a lead below --margin are left
              out. sp.json describes them.
  value-f<k>.keras  the networks, one a part (each has not seen part k).

Needs a GPU.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, optimizers  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sp_common  # noqa: E402
sys.path.insert(0, os.path.join(sp_common.BEN_ROOT, 'scripts', 'training'))
import gpu_epochs  # noqa: E402

N_BIDS = 40
MARGINS = [0, 0.25, 0.5, 1, 1.5, 2, 3]


def value_loss(y, q):
    """Squared error on the tried calls of the turn; y = targets and mask side by side."""
    target, mask = y[..., :N_BIDS], y[..., N_BIDS:]
    return tf.reduce_sum(mask * tf.square(tf.cast(q, tf.float32) - target), axis=-1)


def picks_report(q, u, real, valid, deal_count, margins=MARGINS):
    """What the highest-rated call would have brought. q, u, real: (rows, K) rating, utility and
    real IMPs of each tried call against the call made (column 0)."""
    lead = np.where(valid, q - q[:, :1], -np.inf)
    pick = lead.argmax(axis=1)
    lead = lead[np.arange(len(q)), pick]
    gain = real[np.arange(len(q)), pick]
    util = u[np.arange(len(q)), pick]
    rows = []
    for m in margins:
        on = (pick != 0) & (lead > m)
        n = int(on.sum())
        rows.append({'margin': m, 'turns changed': n, 'share of turns': round(n / len(q), 4),
                     'imps a changed turn': round(float(gain[on].mean()), 3) if n else 0,
                     'better': int((gain[on] > 0).sum()), 'worse': int((gain[on] < 0).sum()),
                     'imps a deal': round(float(gain[on].sum()) / deal_count, 4),
                     'utility a deal': round(float(util[on].sum()) / deal_count, 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--start', required=True, help="the incumbent's network (.keras): its layers below the last are the start")
    parser.add_argument('--rl', action='append', required=True, help='decision file of selfplay_build.py decisions; may be repeated')
    parser.add_argument('--out', required=True)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--weight', choices=['imp', 'mp', 'mix'], default='mix', help='the utility a result counts for')
    parser.add_argument('--scale', type=float, default=3.0, help='target weight = scale times the lead of the best call')
    parser.add_argument('--margin', type=float, default=0.25, help='least lead of the best call for a target')
    parser.add_argument('--max-weight', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not tf.config.list_physical_devices('GPU'):
        raise SystemExit('No GPU visible to TensorFlow.')
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    d = sp_common.load_decisions(args.rl)
    n_rows, width = d['cand'].shape
    n_rounds = d['bids'].shape[1]
    valid = np.arange(width)[None, :] < d['n'][:, None]
    diff = np.where(valid, d['score'] - d['score'][:, :1], 0)
    u = sp_common.utilities(diff, args.weight)
    real = sp_common.imps_array(diff).astype(np.float32)
    fold = d['deal'] % args.folds
    print(f'{n_rows} turns of {d["deals"]} deals, at most {width} tried calls a turn, utility {args.weight}, '
          f'{args.folds} parts, {args.epochs} epochs at learning rate {args.lr}', flush=True)

    with tf.device('/GPU:0'):
        g_static = tf.constant(d['static'])
        g_bids = tf.constant(d['bids'])
        g_step = tf.constant(d['step'].astype(np.int32))
        g_cand = tf.constant(d['cand'].astype(np.int32))
        g_u = tf.constant(u)
        g_valid = tf.constant(valid.astype(np.float32))

    def inputs(r, n):
        s = tf.gather(g_static, r)
        b = tf.one_hot(tf.gather(g_bids, r), N_BIDS, dtype=tf.float16)
        return tf.concat([tf.repeat(s[:, None, :], n_rounds, axis=1), tf.reshape(b, (n, n_rounds, 4 * N_BIDS))], axis=2)

    def targets(r):
        calls = tf.one_hot(tf.gather(g_cand, r), N_BIDS, dtype=tf.float32)        # (B, K, 40)
        v = tf.gather(g_valid, r)
        target = tf.reduce_sum(calls * (tf.gather(g_u, r) * v)[:, :, None], axis=1)
        mask = tf.reduce_sum(calls * v[:, :, None], axis=1)
        at = tf.one_hot(tf.gather(g_step, r), n_rounds, dtype=tf.float32)
        return at[:, :, None] * tf.concat([target, mask], axis=1)[:, None, :]      # (B, 8, 80)

    q_hat = np.zeros((n_rows, width), np.float32)
    curves = []
    t0 = time.time()
    for k in range(args.folds):
        tf.keras.backend.clear_session()
        base = tf.keras.models.load_model(args.start)
        if len(base.outputs) != 1:
            raise SystemExit(f'{args.start} has {len(base.outputs)} outputs; this script reads the single bid output')
        head = layers.TimeDistributed(layers.Dense(N_BIDS, kernel_initializer='zeros'), name='value_output')
        model = tf.keras.Model(base.input, head(base.layers[-1].input))
        model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), loss=value_loss)
        lstm_layers = [layer for layer in model.layers if isinstance(layer, layers.LSTM)]

        def set_unroll(value):
            for layer in lstm_layers:
                layer.unroll = value
                layer.supports_jit = value

        set_unroll(True)
        model.jit_compile = True
        train_rows = np.flatnonzero(fold != k).astype(np.int32)
        test_rows = np.flatnonzero(fold == k).astype(np.int32)
        with tf.device('/GPU:0'):
            g_rows = tf.constant(train_rows)

        def batch(idx):
            r = tf.gather(g_rows, idx)
            return inputs(r, args.batch), targets(r)

        @tf.function
        def rate(r):
            q = model(inputs(r, tf.shape(r)[0]), training=False)
            q = tf.gather(tf.cast(q, tf.float32), tf.gather(g_step, r), axis=1, batch_dims=1)      # (B, 40)
            return tf.gather(q, tf.gather(g_cand, r), axis=1, batch_dims=1)                        # (B, K)

        def rate_rows(rows):
            return np.concatenate([rate(tf.constant(rows[i:i + 4096])).numpy() for i in range(0, len(rows), 4096)])

        deals_k = len(range(k, int(d['deals']), args.folds))      # all deals of the part, with or without a turn
        curve = []

        def validation():
            q = rate_rows(test_rows)
            others = valid[test_rows] & (np.arange(width)[None, :] > 0)
            error = float(((q - u[test_rows])[others] ** 2).mean())
            zero = float((u[test_rows][others] ** 2).mean())
            at0 = picks_report(q, u[test_rows], real[test_rows], valid[test_rows], deals_k, [0, 1])
            curve.append({'error': round(error, 4), 'error of 0': round(zero, 4),
                          'imps a deal, margin 0': at0[0]['imps a deal'], 'imps a deal, margin 1': at0[1]['imps a deal']})
            print(f'part {k}, unseen turns: squared error {error:.4f} (answering 0: {zero:.4f}), '
                  f'IMPs a deal with the highest-rated call {at0[0]["imps a deal"]:+.4f}, lead above 1 only '
                  f'{at0[1]["imps a deal"]:+.4f}', flush=True)
            return {'val_error': error}

        steps = len(train_rows) // args.batch
        gpu_epochs.fit_batches(model, batch, len(train_rows), args.batch, args.epochs, steps, validation=validation)
        q_hat[test_rows] = rate_rows(test_rows)
        curves.append(curve)
        set_unroll(False)
        model.save(os.path.join(args.out, f'value-f{k}.keras'))
        print(f'part {k} done, {time.time() - t0:.0f} s', flush=True)

    # How good the unseen ratings are
    others = valid & (np.arange(width)[None, :] > 0)
    error = float(((q_hat - u)[others] ** 2).mean())
    zero = float((u[others] ** 2).mean())
    hindsight = np.where(valid, real, -np.inf).max(axis=1)
    report = {'args': vars(args), 'turns': n_rows, 'deals': int(d['deals']), 'seconds': round(time.time() - t0, 1),
              'squared error': round(error, 4), 'squared error of answering 0': round(zero, 4),
              'share of the spread explained': round(1 - error / zero, 4),
              'best in hindsight, imps a deal': round(float(hindsight.sum()) / d['deals'], 4),
              'highest-rated call': picks_report(q_hat, u, real, valid, d['deals']),
              'by part and epoch': curves}
    for name, rows in (('matchpoint boards', d['mp'] == 1), ('IMP boards', d['mp'] == 0)):
        n_deals = len(np.unique(d['deal'][rows]))
        report[f'highest-rated call, {name}'] = picks_report(q_hat[rows], u[rows], real[rows], valid[rows], max(n_deals, 1), [0, 1])

    # Targets: the call rated highest, by its lead over the next
    rated = np.where(valid, q_hat, -np.inf)
    order = np.argsort(-rated, axis=1)
    top = order[:, 0]
    lead = rated[np.arange(n_rows), top] - rated[np.arange(n_rows), order[:, 1]]
    keep = np.flatnonzero(lead >= args.margin)
    weight = np.clip(args.scale * lead[keep], 0, args.max_weight)
    cls = np.zeros((len(keep), n_rounds), np.uint8)
    wgt = np.zeros((len(keep), n_rounds), np.float16)
    cls[np.arange(len(keep)), d['step'][keep]] = d['cand'][keep, top[keep]]
    wgt[np.arange(len(keep)), d['step'][keep]] = weight
    changed = top[keep] != 0
    meta = {'method': 'value', 'target': 'the call rated highest by the value network', 'weight': args.weight,
            'scale': args.scale, 'margin': args.margin, 'decision files': args.rl, 'deals': int(d['deals']),
            'turns with candidates': n_rows, 'turns where the results differ': int((diff != 0).any(axis=1).sum()),
            'rows': len(keep), 'targets that are not the call made': int(changed.sum()),
            'targets that are the call made': int((~changed).sum()),
            'mean weight': round(float(weight.mean()), 3) if len(keep) else 0,
            'mean weight of targets that are not the call made': round(float(weight[changed].mean()), 3) if changed.any() else 0,
            'real imps a deal had the targets that are not the call made been bid':
                round(float(real[keep, top[keep]][changed].sum()) / d['deals'], 4)}
    np.savez(os.path.join(args.out, 'sp.npz'), static=d['static'][keep], bids=d['bids'][keep], cls=cls, wgt=wgt, meta=json.dumps(meta))
    report['targets'] = meta
    for name, content in (('sp.json', meta), ('value.json', report)):
        with open(os.path.join(args.out, name), 'w') as f:
            json.dump(content, f, indent=1)

    print(f'\nUnseen ratings: squared error {error:.4f}, answering 0 {zero:.4f} ({100 * (1 - error / zero):.1f}% of the spread explained)')
    print(f'Best call in hindsight: {report["best in hindsight, imps a deal"]:+.4f} IMPs a deal')
    print('Highest-rated call instead of the call made, when it leads by more than the margin:')
    print('  margin   turns  share  IMPs a turn   better    worse  IMPs a deal')
    for r in report['highest-rated call']:
        print(f'  {r["margin"]:6}  {r["turns changed"]:6}  {100 * r["share of turns"]:4.1f}%  {r["imps a changed turn"]:+11.3f}  '
              f'{r["better"]:7}  {r["worse"]:7}  {r["imps a deal"]:+11.4f}')
    print(f'Targets: {meta["rows"]} rows, {meta["targets that are not the call made"]} not the call made, '
          f'mean weight {meta["mean weight"]}')
    print(f'Done in {time.time() - t0:.0f} s.')


if __name__ == '__main__':
    main()
