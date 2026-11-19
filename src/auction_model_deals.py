"""Deals for the bidding search from the auction model (src/auction-model), instead of BEN's own sampler.

Switched on by `[sampling] auction_model = <run folder holding model.pt>` (see BotBid.sample_hands_for_auction).
The model gives, for one hand and the auction so far, the suit lengths and the A, K, Q, J of the three hidden
hands as one joint distribution; a draw from it is completed to a deal by dealing the remaining cards of each
suit at random within the lengths. Every deal is legal, so there is no filtering and no "not enough samples".

The model is PyTorch. Where to find it:
  AUCTION_MODEL_CODE   folder with model.py and infer.py (default: <BEN>/../auction-model, then /code, then
                       /var/lib/ben/auction-model/code)
  AUCTION_MODEL_TORCH  folder holding the torch package if it is not importable (default /var/lib/ben/torch-pkgs)
  AUCTION_MODEL_THREADS  torch threads in this process (default 1: replays run one process per core)
"""
import os
import sys
import time

import numpy as np

_predictors = {}
SEATS = "NESW"
VUL = {(False, False): "None", (True, False): "N_S", (False, True): "E_W", (True, True): "Both"}


def _predictor(run):
    if run not in _predictors:
        try:
            import torch
        except ImportError:
            sys.path.append(os.environ.get("AUCTION_MODEL_TORCH", "/var/lib/ben/torch-pkgs"))
            import torch
        here = os.path.dirname(os.path.abspath(__file__))
        for code in (os.environ.get("AUCTION_MODEL_CODE"), os.path.join(here, "..", "..", "auction-model"), "/code",
                     "/var/lib/ben/auction-model/code"):
            if code and os.path.isfile(os.path.join(code, "infer.py")):
                break
        else:
            raise FileNotFoundError("auction model code not found; set AUCTION_MODEL_CODE")
        # appended, not inserted: BEN's own modules (scoring, ...) must keep winning
        sys.path.append(os.path.abspath(code))
        import importlib.util
        mods = {}
        for name in ("scoring", "model", "infer"):
            # load under private names: BEN has its own `scoring`
            spec = importlib.util.spec_from_file_location("auction_model_" + name, os.path.join(code, name + ".py"))
            mods[name] = importlib.util.module_from_spec(spec)
        saved = {k: sys.modules.get(k) for k in ("scoring", "model", "infer")}
        try:
            for name in ("scoring", "model", "infer"):
                sys.modules[name] = mods[name]
                sys.modules["auction_model_" + name] = mods[name]
                mods[name].__spec__.loader.exec_module(mods[name])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        pred = mods["infer"].Predictor(run, device="cpu")
        torch.set_num_threads(int(os.environ.get("AUCTION_MODEL_THREADS", "1")))
        _predictors[run] = (pred, torch)
    return _predictors[run]


_true_deals = {}


def true_deal(field_path, seat, hand_str):
    """Measurement only (`[sampling] test_true_deal_file`): the four real hands of the board on which `seat` holds
    `hand_str`, from a Brill field file (one JSON object a line with "deal": "N:... ... ... ..."). Handing the search
    the real deal bounds what any better picture of the hidden hands could be worth."""
    if field_path not in _true_deals:
        import json
        table = {}
        with open(field_path) as f:
            for line in f:
                hands = json.loads(line)["deal"][2:].split()
                for s in range(4):
                    table[(s, hands[s])] = hands
        _true_deals[field_path] = table
    return _true_deals[field_path][(seat, hand_str)]


def guard(run, hand_str, seat, dealer, vuln, auction, margin):
    """Passout seat, the other side in a partscore: the cheapest bid in the strain where the model's expected
    double-dummy score for our side beats defending by `margin` points or more, or None.
    Returns (bid, our expected score, expected score defending)."""
    pred, _ = _predictor(run)
    calls = [c for c in auction if c not in ("PAD_START", "PAD_END")]
    bids = [(i, c) for i, c in enumerate(calls) if c[0].isdigit()]
    if not bids or calls[-2:] != ["PASS", "PASS"]:
        return None
    i_last, last = bids[-1]
    by = (dealer + i_last) % 4
    level, strain = int(last[0]), last[1]
    game = level >= (3 if strain == "N" else 4 if strain in "HS" else 5)
    if by % 2 == seat % 2 or game or any(c in ("X", "XX") for c in calls[i_last:]):
        return None

    def declarer(side_seat, s):
        first = [(dealer + i) % 4 for i, c in bids if c[1] == s and (dealer + i) % 2 == side_seat % 2]
        return SEATS[first[0] if first else side_seat]

    pos = pred.position(hand_str, SEATS[seat], SEATS[dealer], VUL[(bool(vuln[0]), bool(vuln[1]))], calls)
    defend = float(pos.contract(last, declarer(by, strain))[0])
    best = None
    for s in "CDHSN":
        lvl = level if "CDHSN".index(s) > "CDHSN".index(strain) else level + 1
        if lvl > 3:
            continue
        ours = float(pos.contract(f"{lvl}{s}", declarer(seat, s))[0])
        if best is None or ours > best[1]:
            best = (f"{lvl}{s}", ours)
    if best is None or best[1] - defend < margin:
        return None
    return best[0], best[1], defend


def tactical(run, hand_str, seat, dealer, vuln, auction, rules, limit=3, matchpoint=None):
    """Calls the purpose rules of auction-model/tactical.py name at this turn: [(bid, rule, text)], at most `limit`,
    in the order of `rules` (e.g. ['double', 'sacrifice', 'interfere', 'lead']). Of an interfering suit's levels
    only the lowest and the highest are kept."""
    pred, _ = _predictor(run)                       # also puts the model's folder on sys.path
    import tactical as purpose
    calls = [c for c in auction if c not in ("PAD_START", "PAD_END")]
    pos = pred.position(hand_str, SEATS[seat], SEATS[dealer], VUL[(bool(vuln[0]), bool(vuln[1]))], calls)
    props, _ = purpose.propose(pos, hand_str, calls, dealer, seat, matchpoint=matchpoint)
    out = []
    for rule in rules:
        mine = [p for p in props if p["rule"] == rule]
        if rule == "interfere":
            ends = {}
            for p in mine:
                ends.setdefault(p["bid"][1], []).append(p)
            mine = [p for ps in ends.values() for p in ([ps[0]] if len(ps) == 1 else [ps[0], ps[-1]])]
        for p in mine:
            if len(out) < limit and all(p["bid"] != b for b, _, _ in out):
                facts = ", ".join(f"{k} {v}" for k, v in p.items() if k not in ("rule", "tag", "bid"))
                out.append((p["bid"], rule, f"{p['tag']}; {facts}"))
    return out


def deals(run, hand_str, seat, dealer, vuln, auction, n, rng, n_cards=32, seed=0):
    """n deals as BEN's bidding arrays: (hands (n, 4, n_cards) by seat N,E,S,W; mean points of LHO, partner, RHO;
    mean suit lengths of the three (12,); seconds). `auction` is BEN's list ('PAD_START', 'PASS', 'X', '1C', ...)."""
    w = n_cards // 4                    # columns a suit: the top w-1 ranks one by one, then a count of the rest
    assert w >= 5, "the bidding format must name at least A, K, Q, J"
    t0 = time.time()
    pred, torch = _predictor(run)
    torch.manual_seed(seed)
    calls = [c for c in auction if c not in ("PAD_START", "PAD_END")]
    pos = pred.position(hand_str, SEATS[seat], SEATS[dealer], VUL[(bool(vuln[0]), bool(vuln[1]))], calls)
    s = pos.samples(n)
    lens, hon = s["lens"], s["hon"]                                    # (n, suit, 3), (n, 16) seat 0 me, 1..3
    mine = [[min("AKQJT98765432".index(ch), w - 1) for ch in suit.upper()] for suit in hand_str.split(".")]
    hands = np.zeros((n, 4, n_cards), np.int32)
    for su in range(4):
        base = su * w
        for c in mine[su]:
            hands[:, seat, base + c] += 1
        # cards below the jack that I do not hold: named ranks one each, and the unnamed ones
        loose = [c for c in range(4, w - 1) if c not in mine[su]] + [w - 1] * (13 - (w - 1) - mine[su].count(w - 1))
        loose = np.array(loose, dtype=np.int64)
        for r in (1, 2, 3):
            hands[:, (seat + r) % 4, base:base + 4] = hon[:, su * 4:su * 4 + 4] == r
        room = lens[:, su, :] - np.stack([(hon[:, su * 4:su * 4 + 4] == r).sum(1) for r in (1, 2, 3)], 1)
        assert (room >= 0).all() and (room.sum(1) == len(loose)).all(), "set does not fit the cards"
        for i in range(n):
            cards = rng.permutation(loose)
            a = 0
            for r in (1, 2, 3):
                for c in cards[a:a + room[i, r - 1]]:
                    hands[i, (seat + r) % 4, base + c] += 1
                a += room[i, r - 1]
    assert (hands.sum(2) == 13).all()
    pts = np.array([4, 3, 2, 1] * 4)
    hcp = np.stack([((hon == r) * pts).sum(1) for r in (1, 2, 3)], 1).mean(0)
    shape = lens.transpose(0, 2, 1).reshape(n, 12).mean(0)
    return hands, hcp, shape, time.time() - t0
