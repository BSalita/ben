"""Shared pieces of the self-play scripts: deals, IMPs, the bidding networks as plain policies.

A "net" here is one bidding network with the few settings its input encoding needs, read
from a BEN config file. Nothing else of the config is loaded, so a worker process stays
small (the full engine with samplers, card-play and lead networks takes 1 to 3 GB).
"""
import bisect
import os
import sys
from configparser import ConfigParser

HERE = os.path.dirname(os.path.abspath(__file__))
# The engine: two folders up when these scripts sit in <BEN>/scripts/selfplay, else $BEN_ROOT
BEN_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if not os.path.isdir(os.path.join(BEN_ROOT, 'src', 'bidding')):
    BEN_ROOT = os.environ.get('BEN_ROOT', BEN_ROOT)
SRC = os.path.join(BEN_ROOT, 'src')
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)

SEATS = 'NESW'
RANKS = 'AKQJT98765432'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
VUL16 = ['None', 'NS', 'EW', 'All', 'NS', 'EW', 'All', 'None', 'EW', 'All', 'None', 'NS', 'All', 'None', 'NS', 'EW']
IMP_STEPS = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300, 1500, 1750, 2000,
             2250, 2500, 3000, 3500, 4000]
N_BIDS = 40
N_ROUNDS = 8


def imps(diff):
    """IMPs for a score difference, with its sign."""
    n = bisect.bisect_right(IMP_STEPS, abs(diff))
    return n if diff >= 0 else -n


def utilities(diff, rule):
    """What a score difference counts for, with its sign (numpy arrays): the weights of
    selfplay_build.py as a utility. imp: the IMPs, at most 12; mp: 4 for any difference, as at
    matchpoints; mix: between the two, 2 to 8. No difference counts 0."""
    import numpy as np
    diff = np.asarray(diff)
    n = np.searchsorted(IMP_STEPS, np.abs(diff), side='right')
    if rule == 'imp':
        w = np.minimum(n, 12)
    elif rule == 'mp':
        w = np.full(n.shape, 4)
    elif rule == 'mix':
        w = np.minimum(2 + (n + 1) // 2, 8)
    else:
        raise ValueError(f'unknown rule {rule}')
    return (np.sign(diff) * w).astype(np.float32)


def imps_array(diff):
    """IMPs for score differences, with their sign (numpy arrays)."""
    import numpy as np
    diff = np.asarray(diff)
    return np.sign(diff) * np.searchsorted(IMP_STEPS, np.abs(diff), side='right')


def load_decisions(paths):
    """The decision files of selfplay_build.py decisions, joined: a dict of arrays, one row a
    learner turn. Candidate arrays are padded to the widest file; deal numbers run on."""
    import numpy as np
    parts = [np.load(p) for p in paths]
    width = max(p['cand'].shape[1] for p in parts)
    out = {}
    first_deal = 0
    for p in parts:
        pad = width - p['cand'].shape[1]
        for key in ('static', 'bids', 'step', 'n', 'cand', 'score', 'p', 'deal', 'mp'):
            a = p[key]
            if key in ('cand', 'score', 'p') and pad:
                a = np.concatenate([a, np.zeros((len(a), pad), a.dtype)], axis=1)
            if key == 'deal':
                a = a + first_deal
            out.setdefault(key, []).append(a)
        first_deal += int(p['n_deals'])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['deals'] = first_deal
    return out


def to_ben(call):
    return 'PASS' if call == 'P' else call


def from_ben(call):
    return 'P' if call == 'PASS' else call


def deal_boards(n, seed, cycle=16, first=0):
    """n random deals. Dealer and vulnerability follow board numbers 1..cycle; the scoring
    (MP or IMP) changes with every pass through the cycle. The same seed gives the same deals."""
    import numpy as np
    rng = np.random.default_rng(seed)
    boards = []
    for i in range(first + n):
        cards = rng.permutation(52)
        if i < first:
            continue
        hands = []
        for h in range(4):
            mine = sorted(int(c) for c in cards[13 * h:13 * h + 13])
            hands.append('.'.join(''.join(RANKS[c % 13] for c in mine if c // 13 == s) for s in range(4)))
        number = i % cycle + 1
        boards.append({'tournament_id': f'sp-{seed}', 'board': i + 1, 'deal': 'N:' + ' '.join(hands),
                       'dealer': SEATS[(number - 1) % 4], 'vul': VUL16[number - 1],
                       'scoring': 'MP' if (i // cycle) % 2 == 0 else 'IMP'})
    return boards


def hands_nesw(deal):
    """The four hands of a PBN deal string, North first."""
    first = SEATS.index(deal[0])
    hands = deal[2:].split()
    return [hands[(i - first) % 4] for i in range(4)]


def resolve_conf(path):
    """A config path the way game.py takes it: as given, or relative to src."""
    for candidate in (path, os.path.join(SRC, path)):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(f'File not found: {path} (looked in the current folder and in {SRC})')


class NetSpec:
    """What binary.get_auction_binary needs to know about a bidding network."""

    def __init__(self, conf_path):
        conf = ConfigParser()
        with open(conf_path) as f:
            conf.read_file(f)
        self.conf_path = conf_path
        self.model_version = conf.getint('models', 'model_version', fallback=2)
        self.n_cards_bidding = conf.getint('models', 'n_cards_bidding', fallback=32)
        self.ns = int(conf['models']['ns'])
        self.ew = int(conf['models']['ew'])
        self.adjust_hcp = conf.getboolean('bidding', 'adjust_hcp', fallback=False)
        self.alert_supported = conf.getboolean('bidding', 'alert_supported', fallback=False)
        self.bidder_path = os.path.join(BEN_ROOT, conf['bidding']['bidder'])
        if self.model_version != 3:
            raise SystemExit(f'{conf_path}: model_version {self.model_version}; these scripts read version 3 networks')
        self.n_static = 2 + 2 + 1 + 4 + self.n_cards_bidding

    def hand(self, hand_str):
        import binary
        return binary.parse_hand_f(self.n_cards_bidding)(hand_str)


class Net(NetSpec):
    """A bidding network loaded for use as a policy: the legal call it rates highest."""

    def __init__(self, conf_path):
        super().__init__(conf_path)
        from nn.bidder_tf2 import Bidder
        self.bidder_model = Bidder('bidder', self.bidder_path, alert_supported=self.alert_supported)

    def probs(self, hand_bin, vuln, auction):
        """The network's output for the player to act after `auction` (BEN form, with PAD_START)."""
        import numpy as np
        import binary
        seat = len(auction) % 4
        i = seat
        while i < len(auction) and auction[i] == 'PAD_START':
            i += 4
        n_steps = 1 + (len(auction) - i) // 4
        x = binary.get_auction_binary(n_steps, auction, seat, hand_bin, vuln, self)
        bids, _ = self.bidder_model.pred_fun_seq(x)
        p = np.asarray(bids[0][-1], dtype=np.float64)
        if p.shape != (N_BIDS,):
            raise ValueError(f'network output of shape {p.shape}')
        return p


def ranked_legal(p, auction, limit=None):
    """Legal calls by falling network output: [(call, output), ...], at most `limit`."""
    import numpy as np
    from bidding import bidding
    out = []
    for k in np.argsort(-p):
        if k >= 2 and bidding.can_bid(bidding.ID2BID[k], auction):
            out.append((bidding.ID2BID[k], float(p[k])))
            if limit and len(out) >= limit:
                break
    return out


def forced_candidates(kinds, auction, hand_str):
    """Calls to try whatever the network thinks of them: [(call, kind), ...] for the player to
    act after `auction` (BEN form) holding hand_str.

      double     X or XX when legal
      compete    the other side holds the contract: the cheapest bid, and the one a level
                 higher, in every suit our side has bid or the hand has five cards of
                 (competitive bids and sacrifices)
      game       our side holds a contract below game: 3N and the game in each suit we have bid
      all:N      pass, X, XX and the next N bids above the last one, whatever they mean
    """
    from bidding import bidding
    seat = len(auction) % 4
    contract = bidding.get_contract(auction)
    ours = set()
    for i, call in enumerate(auction):
        if bidding.is_contract(call) and i % 2 == seat % 2:
            ours.add(call[1])
    theirs_now = contract is not None and bidding.get_decl_i(contract) % 2 != seat % 2
    out = []

    def add(call, kind):
        if bidding.can_bid(call, auction) and all(call != c for c, _ in out):
            out.append((call, kind))

    for kind in kinds:
        if kind == 'double':
            add('X', kind)
            add('XX', kind)
        elif kind == 'compete' and theirs_now:
            long_suits = {s for s, cards in zip('SHDC', hand_str.split('.')) if len(cards) >= 5}
            for strain in sorted(ours | long_suits):
                level = next((n for n in range(1, 8) if bidding.can_bid(f'{n}{strain}', auction)), None)
                if level:
                    add(f'{level}{strain}', kind)
                    if level < 7:
                        add(f'{level + 1}{strain}', kind)
        elif kind == 'game' and contract is not None and not theirs_now:
            add('3N', kind)
            for strain in sorted(ours - {'N'}):
                add(('4' if strain in 'HS' else '5') + strain, kind)
        elif kind.startswith('all:'):
            add('PASS', kind)
            add('X', kind)
            add('XX', kind)
            n = 0
            for k in range(5, N_BIDS):
                if bidding.can_bid(bidding.ID2BID[k], auction):
                    add(bidding.ID2BID[k], kind)
                    n += 1
                    if n >= int(kind[4:]):
                        break
    return out


def result_of(auction, dd, vuln):
    """(contract, North-South score) of a finished auction on a deal with double dummy table
    dd (rows spades, hearts, diamonds, clubs, notrump; columns N, E, S, W)."""
    import scoring
    from bidding import bidding
    contract = bidding.get_contract(auction)
    if contract is None:
        return None, 0
    decl = bidding.get_decl_i(contract)
    tricks = dd[(bidding.get_strain_i(contract) - 1) % 5][decl]
    score = scoring.score(contract, vuln[decl % 2], tricks)
    return contract, int(score if decl % 2 == 0 else -score)
