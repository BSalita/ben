"""The belief model's forward pass in numpy, so the card play can use it without PyTorch.

The model (trained in PyTorch, src/belief-model/belief.py in the bridge repo) is a causal transformer over 128
fixed slots: vulnerability, the calls, the viewer's cards, the start of play, the lead, dummy, then the cards
played. Before each card it gives, for every unseen card, the probability that each hidden hand holds it.

This file holds the same input layout (build), the forward pass for one query, and nothing else. The weights come
from an .npz written by the belief model's export_npz.py (the torch state dict with its arguments). The two give
the same probabilities to about 1e-5 (checked in export_npz.py).
"""
import math

import numpy as np

L = 128
NC = 48
S_VUL, S_CALL, S_OWN, S_START, S_LEAD, S_DUMMY, S_PLAY = 0, 1, 49, 62, 63, 64, 77
T_PAD, T_VUL, T_CALL, T_OWN, T_START, T_PLAYED, T_DUMMY = range(7)
# slot of the query before card j (0-based): the start of play, dummy's last card, then the card before
QUERY = np.array([S_START, S_DUMMY + 12] + [S_PLAY + j - 2 for j in range(2, 52)])


def build(owner, dealer, vul, calls, decl, play, viewer_abs):
    """One game seen from viewer_abs -> token fields [L] (kind, card, seat, extra), valid [L], and the seats
    allowed at each query smask [52, 4]. owner [52] absolute seat 0..3 = N E S W of each card; calls: codes
    (0 pass, 1 double, 2 redouble, 3 + 5*(level-1) + strain C D H S N); play: card codes in the order played."""
    kind = np.zeros(L, np.int64)
    card = np.zeros(L, np.int64)
    seat = np.zeros(L, np.int64)
    extra = np.zeros(L, np.int64)
    valid = np.zeros(L, bool)
    ns, ew = vul in (1, 3), vul in (2, 3)
    we_ns = viewer_abs % 2 == 0
    we, they = (ns, ew) if we_ns else (ew, ns)
    kind[S_VUL], extra[S_VUL], valid[S_VUL] = T_VUL, int(we) + 2 * int(they), True
    for i, c in enumerate(calls[:NC]):
        kind[S_CALL + i], extra[S_CALL + i] = T_CALL, c
        seat[S_CALL + i] = (dealer + i - viewer_abs) % 4
        valid[S_CALL + i] = True
    rel = (np.asarray(owner, np.int64) - viewer_abs) % 4
    kind[S_OWN:S_OWN + 13], card[S_OWN:S_OWN + 13], valid[S_OWN:S_OWN + 13] = T_OWN, np.flatnonzero(rel == 0), True
    kind[S_START], valid[S_START] = T_START, True
    dummy_rel = (decl + 2 - viewer_abs) % 4
    if len(play):
        kind[S_LEAD], card[S_LEAD], seat[S_LEAD], valid[S_LEAD] = T_PLAYED, play[0], rel[play[0]], True
        kind[S_DUMMY:S_DUMMY + 13], card[S_DUMMY:S_DUMMY + 13] = T_DUMMY, np.flatnonzero(rel == dummy_rel)
        seat[S_DUMMY:S_DUMMY + 13], valid[S_DUMMY:S_DUMMY + 13] = dummy_rel, True
        for k, c in enumerate(play[1:]):
            kind[S_PLAY + k], card[S_PLAY + k], seat[S_PLAY + k], valid[S_PLAY + k] = T_PLAYED, c, rel[c], True
    smask = np.ones((52, 4), bool)
    smask[:, 0] = False
    smask[1:, dummy_rel] = False
    return kind, card, seat, extra, valid, smask


def _erf(x):
    # Numerical Recipes' erfc (Chebyshev fit, fractional error below 1.2e-7 everywhere), vectorised
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return 1.0 - np.where(x >= 0, ans, 2.0 - ans)


def _gelu(x):
    # exact GELU (PyTorch's default): x * Phi(x)
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + _erf(x64 / math.sqrt(2.0)))).astype(np.float32)


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class BeliefNP:
    def __init__(self, path):
        z = np.load(path)
        self.d, self.layers, self.heads = int(z["arg_d"]), int(z["arg_layers"]), int(z["arg_heads"])
        self.w = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("arg_")}

    def query(self, kind, card, seat, extra, valid, k):
        """log-probabilities [52 cards, 4 seats] (unnormalised; mask and normalise with smask) at query k"""
        w, d, H = self.w, self.d, self.heads
        q_slot = int(QUERY[k])
        n = q_slot + 1                         # causal: the query reads only slots 0..q_slot
        x = (w["kind.weight"][kind[:n]] + w["card.weight"][card[:n]] + w["seat.weight"][seat[:n]]
             + w["extra.weight"][extra[:n]] + w["pos.weight"][:n])
        dh = d // H
        allowed = np.tril(np.ones((n, n), bool)) & valid[None, :n]
        for i in range(self.layers):
            p = f"enc.layers.{i}."
            h = _ln(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
            qkv = h @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
            q, kk, v = (qkv[:, j * d:(j + 1) * d].reshape(n, H, dh).transpose(1, 0, 2) for j in range(3))
            s = q @ kk.transpose(0, 2, 1) / math.sqrt(dh)
            s = np.where(allowed[None], s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(1, 0, 2).reshape(n, d)
            x = x + a @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"]
            h = _ln(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
            f = _gelu(h @ w[p + "linear1.weight"].T + w[p + "linear1.bias"])
            x = x + f @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
        h = _ln(x[q_slot], w["norm.weight"], w["norm.bias"])
        return (h @ w["head.weight"].T + w["head.bias"]).reshape(52, 4)


def probabilities(model, owner, dealer, vul, calls, decl, play, viewer_abs):
    """[52, 4] probabilities over relative seats (0 viewer, 1 LHO, 2 partner, 3 RHO) before card len(play)"""
    kind, card, seat, extra, valid, smask = build(owner, dealer, vul, calls, decl, play, viewer_abs)
    k = len(play)
    logits = model.query(kind, card, seat, extra, valid, k).astype(np.float64)
    logits = np.where(smask[k][None, :], logits, -1e9)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)
