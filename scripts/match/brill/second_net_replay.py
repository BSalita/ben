#!/usr/bin/env python3
"""brill_replay.py with a second network's first choice added to South's candidates.

A measurement aid, not a setting of BEN: the bidder class is patched in the
workers. At each South turn the second network (the bidder of --second CONF)
names its most likely legal call; when its probability is at least
--second-min-p and the call is not a candidate already, it joins the candidates
with the probability South's own network gives it, and the search decides.

    python second_net_replay.py --second CONF [--second-min-p 0.5] <brill_replay.py arguments>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brill_replay  # noqa: E402

_real_init = brill_replay.worker_init


def install(south_models, second_models, min_p):
    import numpy as np
    from botbidder import BotBid
    from bidding import bidding
    from objects import CandidateBid

    real = BotBid.get_bid_candidates

    def get_bid_candidates(self, auction):
        cands, passout = real(self, auction)
        if self.models is not south_models:
            return cands, passout
        other = BotBid(self.vuln, self.hand_str, second_models, self.sampler, self.seat, self.dealer, self.ddsolver, False, False)
        soft = np.array(other.next_bid_np(auction)[0], dtype=float)
        for i in np.argsort(-soft):
            if bidding.can_bid(bidding.ID2BID[i], auction):
                break
        call = bidding.ID2BID[i]
        if soft[i] >= min_p and all(c.bid != call for c in cands):
            own = np.array(self.next_bid_np(auction)[0], dtype=float)
            cands.append(CandidateBid(bid=call, insta_score=float(own[bidding.BID2ID[call]]), alert=None))
        return cands, passout

    BotBid.get_bid_candidates = get_bid_candidates


def worker_init(cfg):
    _real_init(cfg)
    import ab_match
    from nn.models_tf2 import Models
    second = Models.from_conf(ab_match.load_conf(os.environ['SECOND_NET_CONF'], None), ab_match.BEN_ROOT)
    ab_match.platform_fixups(second)
    install(brill_replay.W['models'], second, float(os.environ.get('SECOND_NET_MIN_P', '0')))
    print(f"worker {os.getpid()}: second network {os.environ['SECOND_NET_CONF']}, min p {os.environ.get('SECOND_NET_MIN_P', '0')}")


def main():
    argv = sys.argv[1:]
    for flag, env in (('--second', 'SECOND_NET_CONF'), ('--second-min-p', 'SECOND_NET_MIN_P')):
        if flag in argv:
            i = argv.index(flag)
            os.environ[env] = argv[i + 1]
            del argv[i:i + 2]
    if 'SECOND_NET_CONF' not in os.environ:
        raise SystemExit('--second CONF is required')
    os.environ['SECOND_NET_CONF'] = brill_replay.ab_match.resolve(os.environ['SECOND_NET_CONF'])
    sys.argv = [sys.argv[0]] + argv
    brill_replay.worker_init = worker_init
    return brill_replay.main()


if __name__ == '__main__':
    sys.exit(main())
