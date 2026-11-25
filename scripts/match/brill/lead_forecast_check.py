"""Does double dummy after the anticipated opening lead forecast real results better than plain double dummy?

For every table of the Brill field file that has a card-play record, with the real deal:
  dd          declarer's double-dummy tricks (the defence's best lead)
  lead_top    double dummy after the lead network's first choice for the leader's hand and the auction
  lead_w      the same over every lead the leader can make, by the network's share
  dd_real     double dummy after the lead that was really made
and the tricks really taken. Uses the engine's own pieces (binary.get_auction_binary_for_lead, the lead
networks, DDSolver.solve_leads), so the forecasts are the ones [eval] dd_after_lead gives the bidding search
when a sampled deal is the real one.

  python lead_forecast_check.py --conf CONF --field field.jsonl --play play.jsonl --out DIR [--limit N]

Writes DIR/tables.jsonl (one row a table) and DIR/report.txt.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ab_match  # noqa: E402  (puts src first on sys.path)

SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
RANKS = 'AKQJT98765432'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', required=True)
    ap.add_argument('--field', required=True)
    ap.add_argument('--play', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='boards')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.chdir(ab_match.SRC)

    import numpy as np
    import binary
    import scoring
    from bidding import bidding
    from ddsolver.ddsolver import DDSolver
    from nn.models_tf2 import Models
    from util import follow_suit

    models = Models.from_conf(ab_match.load_conf(a.conf, None), ab_match.BEN_ROOT)
    ab_match.platform_fixups(models)
    dds = DDSolver()
    n_play, n_bid = models.n_cards_play, models.n_cards_bidding
    per_suit = n_play // 4
    parse_play, parse_bid = binary.parse_hand_f(n_play), binary.parse_hand_f(n_bid)

    play = {}
    with open(a.play, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            play[(r['tournament_id'], r['board'], r['player_id'])] = r['play']

    boards = [json.loads(line) for line in open(a.field, encoding='utf-8')]
    if a.limit:
        boards = boards[: a.limit]

    # one case = deal + auction (many tables share one); its tables differ only in the lead and the result
    cases = {}
    rows = []
    for b in boards:
        hands = b['deal'][2:].split()
        dealer_i = SEATS.index(b['dealer'])
        for t in b['tables']:
            cards = play.get((b['tournament_id'], b['board'], t['player_id']))
            if not cards or t.get('declarer') not in SEATS or not t.get('auction'):
                continue
            auction = ['PAD_START'] * dealer_i + ['PASS' if c == 'P' else c for c in t['auction']]
            contract = bidding.get_contract(auction)
            if contract is None or SEATS.index(contract[-1]) != SEATS.index(t['declarer']):
                continue
            leader = (SEATS.index(contract[-1]) + 1) % 4
            suits = hands[leader].split('.')
            # the lead: the first card of the record, or (records in fixed seat order) the leader's card of trick 1
            lead = next((c for c in cards[:4] if c[1] in suits['SHDC'.index(c[0])]), None)
            if lead is None:
                continue
            key = (b['tournament_id'], b['board'], tuple(t['auction']))
            if key not in cases:
                cases[key] = {'board': b, 'auction': auction, 'contract': contract, 'leader': leader}
            rows.append({'key': key, 'tournament_id': b['tournament_id'], 'board': b['board'], 'player_id': t['player_id'],
                         'type': t.get('type'), 'scoring': b['scoring'], 'contract': contract, 'tricks': t['tricks'], 'lead': lead})
    print(f'{len(boards):,} boards, {len(rows):,} tables with a lead, {len(cases):,} deal + auction cases', flush=True)

    keys = list(cases)
    n = len(keys)
    X = np.zeros((n, 10 + n_play))
    B = np.zeros((n, 15))
    hand_play = np.zeros((n, n_play))
    strains = np.zeros(n, dtype=np.int32)
    for i, k in enumerate(keys):
        c = cases[k]
        b = c['board']
        hand = b['deal'][2:].split()[c['leader']]
        hand_play[i] = parse_play(hand)
        X[i, :], B[i, :] = binary.get_auction_binary_for_lead(c['auction'], parse_bid(hand), hand_play[i:i + 1],
                                                             VULN[b['vul']], SEATS.index(b['dealer']), models)
        strains[i] = 'NSHDC'.index(c['contract'][1])
        if (i + 1) % 2000 == 0:
            print(f'  features {i + 1:,} of {n:,}', flush=True)
    shares = np.zeros((n, n_play))
    for sel, model in ((strains == 0, models.lead_nt_model), (strains > 0, models.lead_suit_model)):
        if np.any(sel):
            shares[sel] = model.pred_fun(X[sel], B[sel])[:, :n_play]
    shares = follow_suit(shares, hand_play, np.zeros((n, 4)), n_play)

    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[(int(strains[i]), cases[k]['leader'])].append(i)
    for (strain, leader), idx in groups.items():
        solved = dds.solve_leads(strain, leader, [cases[keys[i]]['board']['deal'] for i in idx], purpose='check')
        for i, cards in zip(idx, solved):
            cases[keys[i]]['leads'] = cards       # card code -> the defence's tricks
            cases[keys[i]]['shares'] = shares[i]
    print('solved', flush=True)

    def after(cards, c32):
        suit, rank = divmod(int(c32), per_suit)
        if rank < per_suit - 1:
            return cards.get(suit * 13 + rank)
        small = [v for card, v in cards.items() if card // 13 == suit and card % 13 >= per_suit - 1]
        return max(small) if small else None

    out = []
    for r in rows:
        c = cases[r.pop('key')]
        cards, sh = c['leads'], c['shares']
        b = c['board']
        dist = np.zeros(14)
        for c32 in np.nonzero(sh > 0)[0]:
            d = after(cards, c32)
            if d is not None:
                dist[13 - d] += sh[c32]
        dist /= dist.sum()
        top = int(np.argmax(sh))
        real = 'SHDC'.index(r['lead'][0]) * 13 + RANKS.index(r['lead'][1])
        real32 = (real // 13) * per_suit + min(real % 13, per_suit - 1)
        by_trick = scoring.contract_scores_by_trick(c['contract'], tuple(VULN[b['vul']]))
        level = int(c['contract'][0])
        r.update({
            'leader': SEATS[c['leader']], 'need': level + 6,
            'dd': 13 - max(cards.values()), 'lead_top': 13 - after(cards, top), 'lead_w': float(np.dot(dist, np.arange(14))),
            'dd_real': 13 - cards[real], 'p_make_w': float(dist[level + 6:].sum()),
            'top_is_real': top == real32, 'top_suit_is_real': top // per_suit == real32 // per_suit, 'share_real': float(sh[real32]),
            'score': float(by_trick[r['tricks']]), 'score_dd': float(by_trick[13 - max(cards.values())]),
            'score_top': float(by_trick[13 - after(cards, top)]), 'score_w': float(np.dot(dist, by_trick)),
        })
        out.append(r)
    with open(os.path.join(a.out, 'tables.jsonl'), 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')

    def line(name, sel):
        if not sel:
            return f'{name:<34} none'
        m = len(sel)
        text = f'{name:<34} {m:>6,}'
        for f_ in ('dd', 'lead_top', 'lead_w', 'dd_real'):
            err = [r['tricks'] - r[f_] for r in sel]
            text += f' | {sum(err) / m:+.3f} {math.sqrt(sum(e * e for e in err) / m):.3f}'
        return text

    def score_line(name, sel):
        m = len(sel)
        text = f'{name:<34} {m:>6,}'
        for f_ in ('score_dd', 'score_top', 'score_w'):
            err = [r['score'] - r[f_] for r in sel]
            text += f' | {sum(err) / m:+7.1f} {math.sqrt(sum(e * e for e in err) / m):6.1f}'
        return text

    def make_line(name, sel):
        m = len(sel)
        made = [r['tricks'] >= r['need'] for r in sel]
        said_dd = [r['dd'] >= r['need'] for r in sel]
        said_top = [r['lead_top'] >= r['need'] for r in sel]
        brier = lambda p: sum((pi - mi) ** 2 for pi, mi in zip(p, made)) / m
        return (f'{name:<34} {m:>6,} | made {sum(made) / m:.1%} | dd says {sum(said_dd) / m:.1%}, right {sum(s == k for s, k in zip(said_dd, made)) / m:.1%}'
                f' | top lead says {sum(said_top) / m:.1%}, right {sum(s == k for s, k in zip(said_top, made)) / m:.1%}'
                f' | weighted says {sum(r["p_make_w"] for r in sel) / m:.1%}; Brier dd {brier([float(s) for s in said_dd]):.4f}, top {brier([float(s) for s in said_top]):.4f}, weighted {brier([r["p_make_w"] for r in sel]):.4f}')

    splits = [('all tables', out),
              ('leader East or West (robot)', [r for r in out if r['leader'] in 'EW']),
              ('leader North or South', [r for r in out if r['leader'] in 'NS']),
              ('notrump', [r for r in out if r['contract'][1] == 'N']),
              ('suit', [r for r in out if r['contract'][1] != 'N']),
              ('partscore', [r for r in out if (int(r['contract'][0]), r['contract'][1]) < min_game(r['contract'][1])]),
              ('game', [r for r in out if min_game(r['contract'][1]) <= (int(r['contract'][0]), r['contract'][1]) and int(r['contract'][0]) < 6]),
              ('slam', [r for r in out if int(r['contract'][0]) >= 6]),
              ('doubled', [r for r in out if 'X' in r['contract']]),
              ('robot tables', [r for r in out if r['type'] == 'Robot']),
              ('human tables', [r for r in out if r['type'] == 'Human'])]
    m = len(out)
    report = [f'{len(boards):,} boards, {m:,} tables with a lead',
              '',
              "The lead network against the real lead: first choice is the real card "
              f"{sum(r['top_is_real'] for r in out) / m:.1%}, the real suit {sum(r['top_suit_is_real'] for r in out) / m:.1%}; "
              f"mean share given to the real card {sum(r['share_real'] for r in out) / m:.1%}",
              f"The real lead gives up {sum(r['dd_real'] - r['dd'] for r in out) / m:.3f} tricks a table against double dummy "
              f"(costs at least one trick on {sum(r['dd_real'] > r['dd'] for r in out) / m:.1%}); "
              f"the network's first choice {sum(r['lead_top'] - r['dd'] for r in out) / m:.3f} ({sum(r['lead_top'] > r['dd'] for r in out) / m:.1%})",
              '',
              'Tricks really taken minus the forecast: mean, then root mean square',
              f"{'':<34} {'tables':>6} | {'dd':^12} | {'top lead':^12} | {'weighted':^12} | {'real lead':^12}"]
    report += [line(name, sel) for name, sel in splits]
    report += ['', "Declarer's score really made minus the forecast: mean, then root mean square",
               f"{'':<34} {'tables':>6} | {'dd':^14} | {'top lead':^14} | {'weighted':^14}"]
    report += [score_line(name, sel) for name, sel in splits if sel]
    report += ['', 'Does the contract make?']
    report += [make_line(name, sel) for name, sel in splits if sel]
    text = '\n'.join(report)
    with open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


def min_game(strain):
    return {'N': (3, 'N'), 'S': (4, 'S'), 'H': (4, 'H'), 'D': (5, 'D'), 'C': (5, 'C')}[strain]


if __name__ == '__main__':
    main()
