"""
How well BEN's sampled deals fit the auction with a given bidding-info model.

Usage: python eval_binfo_sampling.py <config> <boards.pbn> <info.keras> [<info.keras> ...] [n=1000] [samples=2000] [seed=1]

BEN deals its samples around the bidding-info model's estimate of the three hidden hands and
keeps the ones whose hands would have bid as the auction went (score at least
bidding_threshold_sampling of the configuration, judged by the configuration's bidding models).
A better estimate gives more deals that pass.

For each of n boards of the PBN file (boards the models were not trained on), one decision
point of the auction is drawn, with at least three calls made before it. With every info model
the sampler then deals the same number of samples for the player to call, from the same random
seed, and the deals that pass are counted. Everything but the info model comes from <config>.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BEN_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(BEN_ROOT, 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'bidding'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import conf
from bidding.binary import DealData
from bidding_binary_keras import load_pbn
from nn.bid_info_tf2 import BidInfo
from nn.models_tf2 import Models
from sample import Sample

args = [a for a in sys.argv[1:] if '=' not in a]
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
if len(args) < 3:
    print(__doc__)
    sys.exit(1)
config_path, pbn_path, info_paths = args[0], args[1], args[2:]
n_boards = int(opts.get('n', 1000))
n_samples = int(opts.get('samples', 2000))
seed = int(opts.get('seed', 1))

configuration = conf.load(config_path)
models = Models.from_conf(configuration, BEN_ROOT)
sampler = Sample.from_conf(configuration)
same_for_opponents = getattr(models, 'opponent_binfo_model', None) is models.binfo_model

with open(pbn_path, encoding='utf-8') as f:
    lines = load_pbn(f.readlines())
rng = np.random.default_rng(seed)
positions = []
for j in rng.permutation(len(lines) // 2):
    deal = DealData.from_deal_auction_string(lines[2 * j], lines[2 * j + 1].replace('*', ''), "", models.ns, models.ew,
                                             models.n_cards_bidding, rotate=False)
    auction = deal.auction
    calls_before = np.cumsum([call != 'PAD_START' for call in auction])
    choices = [k for k in range(1, len(auction)) if calls_before[k - 1] >= 3]
    if not choices:
        continue
    k = int(rng.choice(choices))
    positions.append((auction[:k], k % 4, deal.deal_str.split()[k % 4], [deal.vuln_ns, deal.vuln_ew]))
    if len(positions) == n_boards:
        break
print(f"{len(positions)} decision points, {n_samples} samples dealt at each; a sample passes with a score of at least "
      f"{sampler.bidding_threshold_sampling}", flush=True)

print()
print(f"{'info model':52s} {'passing':>8s} {'share':>7s} {'>=15':>6s} {'none':>6s} {'quality':>8s} {'seconds':>8s}")
first = None
for path in info_paths:
    models.binfo_model = BidInfo(path)
    if same_for_opponents:
        models.opponent_binfo_model = models.binfo_model
    passing, quality = [], []
    t0 = time.time()
    for i, (auction, seat, hand_str, vuln) in enumerate(positions):
        accepted, scores, _, _, q = sampler.sample_cards_auction(auction, seat, hand_str, vuln, n_samples,
                                                                 np.random.default_rng(seed + i), models, None, None, {},
                                                                 False, False)
        passing.append(len(accepted))
        if len(accepted) > 0:
            quality.append(float(q))
    passing = np.array(passing)
    line = (f"{os.path.basename(path)[:52]:52s} {passing.mean():8.1f} {100 * passing.mean() / n_samples:6.2f}% "
            f"{100 * (passing >= 15).mean():5.1f}% {100 * (passing == 0).mean():5.1f}% {np.mean(quality):8.3f} {time.time() - t0:8.0f}")
    if first is None:
        first = passing
    else:
        # The same decision points and seeds as for the first model: the difference per point
        diff = passing - first
        line += f"   against the first: {diff.mean():+.1f} +- {diff.std(ddof=1) / np.sqrt(len(diff)):.1f}"
    print(line, flush=True)
print()
print("passing: deals that pass per decision point (mean); share: of the samples dealt;")
print(">=15 and none: decision points with at least 15 passing deals, and with none;")
print("quality: mean score of the passing deals; against the first: mean difference in passing")
print("deals per decision point to the first model, with its standard error.")
