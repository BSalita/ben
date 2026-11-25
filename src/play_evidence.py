"""The evidence BEN used to choose a card, in absolute seats, for explanations.

botcardplayer.play_card keeps, on the CardResp it returns, what its DD evaluation solved: every sampled
layout of the remaining cards, the double-dummy tricks of every legal card on every layout, the weights the
layouts carried in the averages (bidding likelihood, belief model), and PIMC's own scores where PIMC ran.
gameapi /play returns it as "evidence" only with details=true; nothing here changes the card played.

BEN works in seats relative to declarer (0 lefty, 1 dummy, 2 righty, 3 declarer). This module turns them
into absolute seats: layouts as PBN "N:<North> <East> <South> <West>" of the cards still to be played
(the current trick's cards are not in them), cards as "SA", "HT", ...
"""

SEATS = 'NESW'
SUITS = 'SHDC'
RANKS = 'AKQJT98765432'


def card_name(card52):
    card52 = int(card52)
    return SUITS[card52 // 13] + RANKS[card52 % 13]


def _abs_seat(rel_i, decl_i):
    return SEATS[(decl_i + 1 + rel_i) % 4]


def _pbn_absolute(pbn, decl_i):
    hands = pbn[2:].split(' ')
    by_seat = {_abs_seat(j, decl_i): hands[j] for j in range(4)}
    return 'N:' + ' '.join(by_seat[s] for s in SEATS)


def evidence_dict(ev, decl_i):
    """ev: CardResp.evidence (relative seats); decl_i: declarer's absolute seat (0 = North)."""
    if not ev:
        return None
    weights = ev.get('weights')
    if weights is not None:
        weights = [float(w) for w in weights]
        total = sum(weights)
        if total > 0:
            weights = [round(w / total, 6) for w in weights]
    leader = _abs_seat(ev['leader_i'], decl_i)
    out = dict(
        seat=_abs_seat(ev['player_i'], decl_i),
        declarer=SEATS[decl_i],
        contract=ev.get('contract'),
        leader=leader,
        current_trick=[card_name(c) for c in ev['current_trick52']],
        tricks_needed=int(ev['tricks_needed']),
        tricks_taken=int(ev['n_tricks_taken']),
        score_by_tricks=[int(x) for x in ev['score_by_tricks']] if ev.get('score_by_tricks') is not None else None,
        weighted=weights is not None,
        weights=weights,
        quality=None if ev.get('quality') is None else round(float(ev['quality']), 4),
        layouts=[_pbn_absolute(p, decl_i) for p in ev['layouts']],
        dd={card_name(k): [int(x) for x in v] for k, v in ev['dd'].items()},
    )
    pimc = ev.get('pimc')
    if pimc:
        out['pimc'] = dict(engine=pimc['engine'], weight=round(float(pimc['weight']), 3),
                           cards={card_name(k): [None if x is None else round(float(x), 3) for x in v]
                                  for k, v in pimc['cards'].items()})
    return out
