"""The evidence BEN used to choose a card, in absolute seats, for explanations.

botcardplayer.play_card keeps, on the CardResp it returns, what its DD evaluation solved: every sampled
layout of the remaining cards, the double-dummy tricks of every legal card on every layout, the weights the
layouts carried in the averages (bidding likelihood, belief model), and PIMC's own scores where PIMC ran.
gameapi /play returns it as "evidence" only with details=true; nothing here changes the card played.

BEN works in seats relative to declarer (0 lefty, 1 dummy, 2 righty, 3 declarer). This module turns them
into absolute seats: layouts as PBN "N:<North> <East> <South> <West>" of the cards still to be played
(the current trick's cards are not in them), cards as "SA", "HT", ...

lead_evidence_dict does the same for the opening lead (botopeninglead.BotLead.find_opening_lead): the layouts
are whole deals before the lead, `dd` gives the defence's tricks after each candidate lead, and `lead` holds
what decides between the candidates besides the table (the network's scores, the suit adjustments, and the
rule that leads the network's first choice outright).
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


def _restore_lead(pbn, lead52):
    """BEN solves each lead with the card out of the leader's hand (the first hand): put it back."""
    hands = pbn[2:].split(' ')
    suits = hands[0].split('.')
    s, r = lead52 // 13, RANKS[lead52 % 13]
    suits[s] = ''.join(sorted(suits[s] + r, key=RANKS.index))
    hands[0] = '.'.join(suits)
    return pbn[:2] + ' '.join(hands)


def lead_evidence_dict(ev, decl_i):
    """ev: CardResp.evidence of an opening lead (relative seats); decl_i: declarer's absolute seat (0 = North)."""
    if not ev or not ev.get('solved'):
        return None
    tricks = ev['tricks']                          # declarer's tricks, layout x candidate
    contract = ev['contract']
    level = int(contract[0])
    by_card, dd, x_name = {}, {}, {}
    for j, (card32, lead52, layouts) in enumerate(ev['solved']):
        name = card_name(lead52)
        by_card[name] = [_restore_lead(p, lead52) for p in layouts]
        dd[name] = [13 - int(t) for t in tricks[:, j, 0]]
        x_name[name] = SUITS[card32 // 8] + 'AKQJT98x'[card32 % 8]
    first = next(iter(by_card.values()))
    same = all(v == first for v in by_card.values())
    rule = ev['rule']
    scoring = 'MP' if rule.get('matchpoint') else 'IMP'
    out = dict(
        seat=_abs_seat(0, decl_i),
        declarer=SEATS[decl_i],
        contract=contract,
        leader=_abs_seat(0, decl_i),
        current_trick=[],
        tricks_needed=13 - (level + 6) + 1,
        tricks_taken=0,
        score_by_tricks=[int(x) for x in ev['scores_by_trick']],
        weighted=False,
        weights=None,
        quality=None if ev.get('quality') is None else round(float(ev['quality']), 4),
        layouts=[_pbn_absolute(p, decl_i) for p in first],
        dd=dd,
        lead=dict(
            # the candidate each solved card stands for ("Hx" = a small heart) and the rules that rank them
            candidate=x_name,
            # the lead network's score of each candidate at full precision (the candidates list rounds it to 3
            # decimals; BEN compares it with accept_nn and breaks ties on it rounded to 2)
            nn={name: round(float(ev['nn'][0][c32]), 6) for name, c32 in
                ((card_name(l52), c) for c, l52, _ in ev['solved'])},
            suit_adjust=dict(zip(SUITS, [round(float(a), 4) for a in ev['suit_adjust']])),
            accept_nn=float(rule['lead_accept_nn']),
            nn_on_bad_quality=bool(rule['use_biddingquality_in_eval']),
            scored=bool(rule['use_real_imp_or_mp']),
            scoring=scoring,
            exact_ranks=RANKS[:int(ev['n_cards_bidding']) // 4 - 1],
            # each layout's bidding score; layouts were drawn with these as probabilities, not weighted by them
            sample_scores=[round(float(x), 5) for x in ev['sample_scores'][:len(first)]],
        ),
    )
    if not same:     # never seen; kept so the evidence stays exact if a candidate's pips were dealt differently
        out['layouts_by_card'] = {k: [_pbn_absolute(p, decl_i) for p in v] for k, v in by_card.items()}
    return out
