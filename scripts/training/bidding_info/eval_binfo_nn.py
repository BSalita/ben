"""
Evaluate bidding-info models on boards they were not trained on.

Usage: python eval_binfo_nn.py <bin_dir> <model.keras> [<model.keras> ...] [rounds=1]

  bin_dir: x.npy, y.npy, HCP.npy and SHAPE.npy made by bidding_binary_keras.py from a held-out
           PBN (use max_occurrences=1000000 there, so no board is dropped).
  rounds=1 also prints the errors per bidding round.

The info network estimates, from one hand and the auction so far, the high card points and the
suit lengths of the three other hands; BEN deals its samples around those estimates. Errors
are given in real units (points, cards per suit), separately for

  bidding   the rounds in which the player still has a call to make: what BEN uses when it
            samples during the auction;
  after     the first round after the auction has ended (all calls seen): close to what BEN
            uses for the opening lead and the play.

"no info" is the error of dividing what the player does not hold evenly over the three other
hands, without looking at the auction; a model must be well below it to be of use.
"""
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

for d in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)

args = [a for a in sys.argv[1:] if '=' not in a]
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
if len(args) < 2:
    print(__doc__)
    sys.exit(1)
bin_dir, model_paths = args[0], args[1:]
per_round = int(opts.get('rounds', 0))

PAD_END = 1
HCP_SCALE, SHAPE_SCALE = 4.0, 1.75      # the targets are (hcp - 10) / 4 and (length - 3.25) / 1.75

X = np.load(os.path.join(bin_dir, 'x.npy'))
y = np.load(os.path.join(bin_dir, 'y.npy')).argmax(axis=-1)
HCP = np.load(os.path.join(bin_dir, 'HCP.npy')).astype(np.float32) * HCP_SCALE
SHAPE = np.load(os.path.join(bin_dir, 'SHAPE.npy')).astype(np.float32) * SHAPE_SCALE
n, n_rounds, _ = X.shape

bidding = y != PAD_END                                        # (n, rounds)
ended = ~bidding
after = ended & (np.cumsum(ended, axis=1) == 1)               # first round after the auction
print(f"{n} hands ({n // 4} boards), {int(bidding.sum())} bidding rounds, {int(after.sum())} hands with a round after the auction")

# Without the auction: what the player does not hold, divided evenly over the three others
own_hcp = X[:, :, 4].astype(np.float32) * HCP_SCALE + 10
own_shape = X[:, :, 5:9].astype(np.float32) * SHAPE_SCALE
total_hcp = float(np.mean(own_hcp[:, 0] + (HCP[:, 0] + 10).sum(axis=-1)))   # 40 unless the points are adjusted
print(f"Points in a deal: {total_hcp:.2f}")
no_info_hcp = np.repeat(((total_hcp - own_hcp) / 3)[:, :, None], 3, axis=2) - (HCP + 10)
no_info_shape = np.tile(-own_shape / 3, (1, 1, 3)) - SHAPE               # both sides are offsets from 3.25


def predict(model):
    hcp, shape = [], []
    for a in range(0, n, 8192):
        h, s = model(tf.constant(X[a:a + 8192]), training=False)
        hcp.append(h.numpy().astype(np.float32))
        shape.append(s.numpy().astype(np.float32))
    return np.concatenate(hcp) * HCP_SCALE, np.concatenate(shape) * SHAPE_SCALE


def errors(e_hcp, e_shape, mask):
    h, s = e_hcp[mask], e_shape[mask]
    return {'hcp': np.abs(h).mean(), 'hcp_lho': np.abs(h[:, 0]).mean(), 'hcp_pard': np.abs(h[:, 1]).mean(),
            'hcp_rho': np.abs(h[:, 2]).mean(), 'hcp_rmse': np.sqrt((h ** 2).mean()),
            'shape': np.abs(s).mean(), 'shape_rmse': np.sqrt((s ** 2).mean())}


def report(name, e_hcp, e_shape):
    print(f"=== {name}")
    out = []
    for label, mask in (('bidding', bidding), ('after', after)):
        e = errors(e_hcp, e_shape, mask)
        out.append(e)
        print(f"   {label:8s} hcp error {e['hcp']:.3f} (lho {e['hcp_lho']:.3f}, partner {e['hcp_pard']:.3f}, rho {e['hcp_rho']:.3f}; "
              f"rmse {e['hcp_rmse']:.3f})   suit length error {e['shape']:.4f} (rmse {e['shape_rmse']:.4f})")
    if per_round:
        for t in range(n_rounds):
            m = np.zeros_like(bidding)
            m[:, t] = bidding[:, t]
            if m.any():
                e = errors(e_hcp, e_shape, m)
                print(f"   round {t + 1}: {int(m.sum()):7d} rows   hcp {e['hcp']:.3f}   suit length {e['shape']:.4f}")
    b, a = out
    print(f"   summary: bidding hcp {b['hcp']:.3f} shape {b['shape']:.4f}; after hcp {a['hcp']:.3f} shape {a['shape']:.4f}")


report("no info", no_info_hcp, no_info_shape)
for path in model_paths:
    model = load_model(path, compile=False)
    p_hcp, p_shape = predict(model)
    report(os.path.basename(path), p_hcp - HCP, p_shape - SHAPE)
