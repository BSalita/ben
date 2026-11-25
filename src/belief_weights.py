"""Weights for the play sampler's layouts from a belief model (src/belief-model in the bridge repo).

The belief model reads the auction, the player's own cards, dummy and the cards played so far, and gives for each
unseen card the probability that each hidden hand holds it. A layout's log-likelihood is the sum over the unseen
honours (A to 8) of the log-probability of the hand the layout puts it in, plus, suit by suit, the log-probability
of the layout's number of small cards (7 and below, which the play sampler does not tell apart) in each hidden hand.

Off unless [sampling] play_belief_model names a model file: .npz runs in numpy (belief_np.py, no PyTorch needed);
.pt runs in PyTorch, with play_belief_code = the folder holding belief.py.
"""
import os
import sys

import numpy as np

_MODEL = {}
STRAINS = "CDHSN"


def _load(path, code):
    key = (path, code)
    if key not in _MODEL:
        if path.endswith(".npz"):
            import belief_np
            _MODEL[key] = ("np", belief_np.BeliefNP(path), belief_np)
        else:
            import torch
            if code and code not in sys.path:
                sys.path.insert(0, code)
            import belief
            torch.set_num_threads(int(os.environ.get("BEN_BELIEF_THREADS", "2")))
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]
            m = belief.BeliefModel(d=a["d"], layers=a["layers"], heads=a["heads"], ff=4 * a["d"])
            m.load_state_dict(ck["state"])
            m.eval()
            _MODEL[key] = ("torch", m, (belief, torch))
    return _MODEL[key]


def _call_code(c):
    c = c.upper()
    if c in ("PASS", "P"):
        return 0
    if c in ("X", "D"):
        return 1
    if c in ("XX", "R"):
        return 2
    return 3 + 5 * (int(c[0]) - 1) + STRAINS.index(c[1])


def _winner(trick, strain_i):
    """trick: list of (player, card52) in order; strain_i: 0 = NT, 1..4 = S H D C (BEN's bidding.get_strain_i)"""
    led = trick[0][1] // 13
    trump = None if strain_i == 0 else strain_i - 1
    def power(pc):
        s = pc[1] // 13
        return (2 if s == trump else 1 if s == led else 0, -(pc[1] % 13))
    return max(trick, key=power)[0]


def card_probabilities(path, code, auction, vuln, contract, strain_i, player_to_nesw, player_i, trick_i,
                       played_cards, n_current, own_now52, public_now52):
    """Belief model's [52, 4] probabilities (relative seats: 0 viewer, 1 LHO, 2 partner, 3 RHO) at this moment,
    and the viewer's absolute seat. played_cards: the flat list the game passes to the sampler (each player's
    cards of the finished tricks, player by player, then the current trick in order); n_current: cards in the
    current trick. Players are BEN's relative ones (0 lefty, 1 dummy, 2 righty, 3 declarer)."""
    backend, model, lib = _load(path, code)
    rows = [list(played_cards[p * trick_i:(p + 1) * trick_i]) for p in range(4)]
    current = list(played_cards[4 * trick_i:4 * trick_i + n_current])
    if len(played_cards) != 4 * trick_i + n_current:
        return None, None
    order, leader = [], 0
    for t in range(trick_i):
        trick = [((leader + i) % 4, rows[(leader + i) % 4][t]) for i in range(4)]
        order += trick
        leader = _winner(trick, strain_i)
    order += [((leader + i) % 4, c) for i, c in enumerate(current)]
    decl_abs = player_to_nesw(3, contract)
    viewer_rel = 3 if player_i in (1, 3) else player_i
    viewer_abs = player_to_nesw(viewer_rel, contract)
    owner = np.full(52, -1, np.int64)
    # the viewer's and dummy's original hands: what they hold now plus what they played
    own_player, dummy_player = (3, 1) if player_i in (1, 3) else (player_i, 1)
    own_now = public_now52 if player_i == 1 else own_now52
    dummy_now = own_now52 if player_i == 1 else public_now52
    for c in np.flatnonzero(np.asarray(own_now) > 0):
        owner[c] = viewer_abs
    for c in np.flatnonzero(np.asarray(dummy_now) > 0):
        owner[c] = player_to_nesw(1, contract)
    for p, c in order:
        owner[c] = player_to_nesw(p, contract)
    # cards nobody has shown: any hidden seat (they only fill the model's target slots, which are not read)
    hidden_abs = [player_to_nesw(p, contract) for p in range(4) if p not in (own_player, dummy_player)]
    left = np.flatnonzero(owner < 0)
    owner[left[:max(0, 13 - int((owner == hidden_abs[0]).sum()))]] = hidden_abs[0]
    owner[owner < 0] = hidden_abs[1]
    if any((owner == s).sum() != 13 for s in range(4)):
        return None, None
    calls = [_call_code(c) for c in auction if c not in ("PAD_START", "PAD_END")]
    dealer = sum(1 for c in auction if c == "PAD_START") % 4
    ns, ew = bool(vuln[0]), bool(vuln[1])
    play = np.full((1, 52), -1, np.int8)
    play[0, :len(order)] = [c for _, c in order]
    g = dict(owner=owner[None].astype(np.int8), dealer=np.array([dealer]), vul=np.array([int(ns) + 2 * int(ew)]),
             calls=np.full((1, 64), -1, np.int8), ncalls=np.array([min(len(calls), 64)]), decl=np.array([decl_abs]),
             play=play, nplay=np.array([len(order)]))
    g["calls"][0, :min(len(calls), 64)] = calls[:64]
    k = len(order)
    if backend == "np":
        # numpy forward pass (belief_np.py): the same inputs, only the query that is needed
        p = lib.probabilities(model, owner, dealer, int(ns) + 2 * int(ew), calls[:64], decl_abs,
                              [c for _, c in order], viewer_abs)
    else:
        belief, torch = lib
        kind, card, seat, extra, valid, _, _, smask, _ = belief.build(g, np.array([viewer_abs]))
        # query k reads only the slots before card k (causal), so the unknown next card does not enter
        with torch.no_grad():
            t = [torch.from_numpy(x) for x in (kind, card, seat, extra, valid, smask)]
            p = belief.masked_logp(model(*t[:5]), t[5]).exp()[0, k].numpy()
    return p, viewer_abs, order


def layout_weights_log(path, code, auction, vuln, contract, strain_i, player_to_nesw, player_i, trick_i,
                       played_cards, n_current, own_now52, public_now52, hidden_1_i, hidden_2_i, hands32_1, hands32_2):
    """log-likelihood of each sampled layout (hands32_1, hands32_2: [n, 32] original hidden hands), or None."""
    p, viewer_abs, order = card_probabilities(path, code, auction, vuln, contract, strain_i, player_to_nesw, player_i,
                                              trick_i, played_cards, n_current, own_now52, public_now52)
    if p is None:
        return None
    seen = set(int(c) for c in np.flatnonzero(np.asarray(own_now52) > 0)) | set(int(c) for c in np.flatnonzero(np.asarray(public_now52) > 0))
    seen |= set(int(c) for _, c in order)
    unseen = [c for c in range(52) if c not in seen]
    played_by = [[c for q, c in order if q == h] for h in (hidden_1_i, hidden_2_i)]
    hidden_abs = [player_to_nesw(hidden_1_i, contract), player_to_nesw(hidden_2_i, contract)]
    return layout_loglik(p, viewer_abs, hidden_abs, unseen, [hands32_1, hands32_2], played_by)


def layout_loglik(p, viewer_abs, hidden_abs, unseen52, hands32, played_by_hidden52):
    """log-likelihood of each layout under the belief probabilities p [52, 4].

    hidden_abs: absolute seats of the two hidden hands; hands32: their original hands in the sampler's
    32-card code, [n, 32] each; unseen52: the cards the viewer cannot see now; played_by_hidden52: for each
    hidden hand, the cards it has played."""
    rel = [(h - viewer_abs) % 4 for h in hidden_abs]
    q1 = np.clip(p[:, rel[0]], 1e-6, 1.0)
    q2 = np.clip(p[:, rel[1]], 1e-6, 1.0)
    s = q1 + q2
    q1, q2 = q1 / s, q2 / s
    unseen = np.asarray(sorted(unseen52), dtype=np.int64)
    rank = unseen % 13
    suit = unseen // 13
    hon = unseen[rank < 7]
    h1 = hands32[0]
    n = h1.shape[0]
    ll = np.zeros(n, np.float64)
    if len(hon):
        idx32 = (hon // 13) * 8 + hon % 13
        in1 = h1[:, idx32] > 0
        ll += np.where(in1, np.log(q1[hon])[None, :], np.log(q2[hon])[None, :]).sum(axis=1)
    for st in range(4):
        small = unseen[(suit == st) & (rank >= 7)]
        if not len(small):
            continue
        # polynomial prod (q2 + q1 x): coefficient k = probability that hand 1 holds exactly k of them
        poly = np.array([1.0])
        for c in small:
            poly = np.convolve(poly, np.array([q2[c], q1[c]]))
        played_small = sum(1 for c in played_by_hidden52[0] if c // 13 == st and c % 13 >= 7)
        k1 = h1[:, st * 8 + 7].astype(np.int64) - played_small
        ok = (k1 >= 0) & (k1 < len(poly))
        ll += np.where(ok, np.log(np.maximum(poly[np.clip(k1, 0, len(poly) - 1)], 1e-12)), -50.0)
    return ll
