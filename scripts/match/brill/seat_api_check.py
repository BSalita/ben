"""Run the conformance checklist of Brill's Seat Robot API (seat-api.txt, section 9) against a server.

    python seat_api_check.py BASE_URL [--out LOG.json] [--pause 0.7] [--skip-names]

BASE_URL is what would be handed to Brill, for example https://morty.7nt.info/gameapi/8085 .
The script is the table: it deals two boards, asks the server for every call and
every card of all four seats, and checks each answer against the rules of the
game (legal call, card held, suit followed). Then it sends the other requests
the checklist names (meanings, time, matchtype, every advertised system and
carding name, board over, malformed requests) and prints one line per item.

Requests go one at a time, `--pause` seconds apart (the server may limit a
client's rate). Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

ORIGIN = 'https://brill.aalborgdata.dk'
SEATS = 'NESW'
SUITS = 'SHDC'
RANKS = 'AKQJT98765432'
STRAINS = 'CDHSN'

BOARDS = [
    # the worked example of the specification: dealer North, North-South vulnerable, IMPs
    {'board': '7', 'dealer': 'N', 'vul': 'NS', 'matchtype': 'IMP',
     'hands': {'N': 'AKQ82.AJ4.KT9.76', 'E': 'J9.KQ98.876.J432', 'S': 'T73.762.A54.Q985', 'W': '654.T53.QJ32.AKT'}},
    # a strong South hand: South or North declares a high contract, matchpoints
    {'board': '12', 'dealer': 'S', 'vul': 'None', 'matchtype': 'MP',
     'hands': {'N': 'Q95.A84.KJ73.Q62', 'E': '64.762.Q954.T873', 'S': 'AKJ73.KQ5.A82.K4', 'W': 'T82.JT93.T6.AJ95'}},
]
TIME = {'boardTimeLimitMs': 120000, 'boardTimeUsedMs': 15000, 'sessionTimeLimitMs': 960000,
        'sessionTimeUsedMs': 135000, 'sessionBoardsLeft': 6}


class Client:
    def __init__(self, base, pause):
        self.base, self.pause, self.log, self.last = base.rstrip('/'), pause, [], 0.0

    def get(self, path, params=None, note=''):
        wait = self.pause - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        url = f'{self.base}{path}' + ('?' + urllib.parse.urlencode(params) if params else '')
        request = urllib.request.Request(url, headers={'Origin': ORIGIN, 'User-Agent': 'seat-api-check'})
        t0 = time.time()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status, headers, text = response.status, response.headers, response.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as error:
            status, headers, text = error.code, error.headers, error.read().decode('utf-8', 'replace')
        self.last = time.time()
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        entry = {'path': path, 'params': params or {}, 'status': status, 'seconds': round(self.last - t0, 2),
                 'cors': headers.get('Access-Control-Allow-Origin'), 'body': body if body is not None else text[:300],
                 'note': note}
        self.log.append(entry)
        return entry


def cards_of(hand):
    return {s + r for s, part in zip(SUITS, hand.split('.')) for r in part}


def check_deal(hands):
    deck = [c for seat in SEATS for c in cards_of(hands[seat])]
    assert len(deck) == 52 and len(set(deck)) == 52 and all(len(cards_of(hands[s])) == 13 for s in SEATS), hands


def token(call):
    """A server's call in the two-character vocabulary of ctx, or None."""
    c = str(call).strip().upper().replace('NT', 'N')
    if c in ('--', 'P', 'PASS'):
        return '--'
    if c in ('DB', 'X', 'DBL', 'DOUBLE'):
        return 'Db'
    if c in ('RD', 'XX', 'RDBL', 'REDOUBLE'):
        return 'Rd'
    return c if len(c) == 2 and c[0] in '1234567' and c[1] in STRAINS else None


def legal_call(calls, tok):
    """Is `tok` legal after `calls` (tokens from the dealer)?"""
    if tok == '--':
        return True
    last = next(((len(calls) - 1 - i, c) for i, c in enumerate(reversed(calls)) if c != '--'), None)
    if tok == 'Db':
        return last is not None and last[1] not in ('Db', 'Rd') and (len(calls) - last[0]) % 2 == 1
    if tok == 'Rd':
        return last is not None and last[1] == 'Db' and (len(calls) - last[0]) % 2 == 1
    bids = [c for c in calls if c not in ('--', 'Db', 'Rd')]
    rank = lambda b: (int(b[0]), STRAINS.index(b[1]))
    return not bids or rank(tok) > rank(bids[-1])


def auction_over(calls):
    return len(calls) >= 4 and calls[-3:] == ['--'] * 3 and (len(calls) > 3 and (any(c != '--' for c in calls) or len(calls) == 4))


def contract_of(calls, dealer):
    """(level, strain, declarer) or None for a passed-out auction."""
    bids = [(i, c) for i, c in enumerate(calls) if c not in ('--', 'Db', 'Rd')]
    if not bids:
        return None
    i, last = bids[-1]
    side = (SEATS.index(dealer) + i) % 2
    first = next(j for j, c in bids if c[1] == last[1] and (SEATS.index(dealer) + j) % 2 == side)
    return int(last[0]), last[1], SEATS[(SEATS.index(dealer) + first) % 4]


def meanings_for(seat, dealer, calls, texts):
    """The meanings array a seat is sent (its opponents' explanations only), or None when all blank."""
    me = SEATS.index(seat) % 2
    out = [texts[i] if (SEATS.index(dealer) + i) % 2 != me else '' for i in range(len(calls))]
    return json.dumps(out) if any(out) else None


def play_board(client, b, results):
    """Bid and play one board through the server. Returns the common parameters, or None if it stopped."""
    hands, dealer = b['hands'], b['dealer']
    check_deal(hands)
    common = {'dealer': dealer, 'vul': b['vul'], 'matchtype': b['matchtype'], 'board': b['board']}
    calls, texts, problems, explained = [], [], [], []
    while not auction_over(calls) and len(calls) < 60:
        seat = SEATS[(SEATS.index(dealer) + len(calls)) % 4]
        params = dict(common, hand=hands[seat], seat=seat, ctx=''.join(calls))
        m = meanings_for(seat, dealer, calls, texts)
        if m:
            params['meanings'] = m
        r = client.get('/bid', params, f"board {b['board']} bid {seat}")
        tok = token(r['body'].get('bid')) if isinstance(r['body'], dict) else None
        if r['status'] != 200 or tok is None or not legal_call(calls, tok):
            problems.append(f"{seat} after '{''.join(calls)}': status {r['status']}, answer {r['body']}")
            break
        text = (r['body'].get('explanation') or '').strip()
        explained.append((seat, tok, text, r['body'].get('alert')))
        calls.append(tok)
        texts.append(text)
    results['auctions'].append({'board': b['board'], 'calls': ' '.join(calls), 'explained': explained})
    results['bid_problems'] += problems
    if problems or not auction_over(calls):
        return None
    contract = contract_of(calls, dealer)
    ctx = ''.join(calls)
    common['ctx'] = ctx
    if contract is None:
        return common
    level, strain, declarer = contract
    dummy = SEATS[(SEATS.index(declarer) + 2) % 4]
    leader = SEATS[(SEATS.index(declarer) + 1) % 4]
    remaining = {s: cards_of(hands[s]) for s in SEATS}
    played, trick, to_play = [], [], leader

    def with_meanings(seat, params):
        m = meanings_for(seat, dealer, calls, texts)
        if m:
            params['meanings'] = m
        return params

    for n in range(52):
        actor = to_play
        if n == 0:
            r = client.get('/lead', with_meanings(actor, dict(common, hand=hands[actor], seat=actor)), f"board {b['board']} lead {actor}")
        else:
            decider = declarer if actor == dummy else actor
            params = dict(common, hand=hands[decider], seat=decider, dummy=hands[dummy], played=''.join(played))
            r = client.get('/play', with_meanings(decider, params), f"board {b['board']} card {n + 1} {actor}" + (' (dummy, asked of declarer)' if actor == dummy else ''))
        card = str(r['body'].get('card', '')).upper() if isinstance(r['body'], dict) else ''
        where = f"board {b['board']} card {n + 1} ({actor}{', dummy' if actor == dummy else ''})"
        if r['status'] != 200 or card not in remaining[actor]:
            results['card_problems'].append(f'{where}: status {r["status"]}, answer {str(r["body"])[:160]}')
            return None
        if trick and card[0] != trick[0][1][0] and any(c[0] == trick[0][1][0] for c in remaining[actor]):
            results['card_problems'].append(f'{where}: {card} does not follow {trick[0][1]}')
            return None
        if actor == dummy:
            results['dummy_cards'] += 1
        remaining[actor].discard(card)
        played.append(card)
        trick.append((actor, card))
        if len(trick) == 4:
            led = trick[0][1][0]
            trumps = [t for t in trick if t[1][0] == strain]
            pool = trumps or [t for t in trick if t[1][0] == led]
            to_play = min(pool, key=lambda t: RANKS.index(t[1][1]))[0]
            trick = []
        else:
            to_play = SEATS[(SEATS.index(actor) + 1) % 4]
    results['boards_played'].append({'board': b['board'], 'contract': f'{level}{strain} by {declarer}', 'cards': len(played)})
    common.update(declarer=declarer, dummy_hand=hands[dummy], declarer_hand=hands[declarer], played=''.join(played))
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('--out')
    parser.add_argument('--pause', type=float, default=0.7)
    parser.add_argument('--skip-names', action='store_true', help='do not try every advertised system and carding name')
    args = parser.parse_args()
    client = Client(args.base, args.pause)
    started = datetime.now()
    print(f'start {started:%Y-%m-%d %H:%M:%S}  {client.base}', flush=True)
    lines = []

    def item(ok, text, detail=''):
        mark = {True: 'PASS', False: 'FAIL', None: 'NOTE'}[ok]
        lines.append((mark, text, detail))
        print(f'{mark}  {text}' + (f'\n      {detail}' if detail else ''), flush=True)

    def bid_of(r):
        return token(r['body'].get('bid')) if r['status'] == 200 and isinstance(r['body'], dict) else None

    # GET /
    root = client.get('/', note='root')
    caps = root['body'] if isinstance(root['body'], dict) else {}
    item(root['status'] == 200, 'GET / returns 200', f"body keys: {sorted(caps)}" if caps else str(root['body'])[:200])

    # an opening call
    b0 = BOARDS[0]
    base_bid = {'hand': b0['hands']['N'], 'seat': 'N', 'dealer': 'N', 'vul': b0['vul'], 'ctx': '', 'board': 'check'}
    r = client.get('/bid', dict(base_bid, matchtype='IMP'), 'opening call')
    opening = bid_of(r)
    item(opening is not None and legal_call([], opening), '/bid with ctx= (empty) answers a legal opening call', f"answer: {r['body']}")

    # whole boards
    results = {'auctions': [], 'bid_problems': [], 'card_problems': [], 'boards_played': [], 'dummy_cards': 0}
    finished = [play_board(client, b, results) for b in BOARDS]
    for a in results['auctions']:
        print(f"      board {a['board']} auction: {a['calls']}", flush=True)
    item(not results['bid_problems'] and all(f is not None or results['card_problems'] for f in finished),
         '/bid answers every seat of a full auction and reaches three passes',
         '; '.join(results['bid_problems']) or '; '.join(f"board {a['board']}: {a['calls']}" for a in results['auctions']))
    called = [e for a in results['auctions'] for e in a['explained'] if e[1] != '--']
    with_text = [e for e in called if e[2]]
    item(len(with_text) == len(called) and bool(called), '/bid returns explanation on conventional calls',
         f"{len(with_text)} of {len(called)} non-pass calls came with an explanation; e.g. " +
         '; '.join(f'{s} {t}: "{x}"' for s, t, x, _ in with_text[:4]))
    alerts = sorted({type(e[3]).__name__ for a in results['auctions'] for e in a['explained']})
    item(alerts == ['bool'], 'alert is a boolean', f'type(s) returned: {alerts}')
    named = {k: caps.get(k) for k in ('engine', 'version', 'build')}
    item(None if not all(named.values()) else True, 'GET / says which build is answering (optional)', str(named))

    # meanings, partnerMeanings, matchtype, time: the same request with and without
    resp = {'hand': b0['hands']['E'], 'seat': 'E', 'dealer': 'N', 'vul': b0['vul'], 'ctx': '1S', 'board': 'check'}
    plain = client.get('/bid', resp, 'no meanings')
    meant = client.get('/bid', dict(resp, meanings=json.dumps(['5+ spades, 11-21 HCP'])), 'with meanings')
    item(bid_of(plain) is not None and bid_of(meant) is not None,
         'a request with meanings is answered, and one without it too', f"without: {bid_of(plain)}, with: {bid_of(meant)}")
    pm = client.get('/bid', dict(resp, ctx='1S--2S', seat='W', hand=b0['hands']['W'], partnerMeanings='true',
                                 meanings=json.dumps(['5+ spades', 'weak', '3-card support, 6-9 HCP'])), 'partnerMeanings')
    item(bid_of(pm) is not None, 'a request carrying partnerMeanings=true is answered', f"answer: {bid_of(pm)}")
    mt = {m: bid_of(client.get('/bid', dict(base_bid, **({'matchtype': m} if m else {})), f'matchtype {m}')) for m in ('MP', 'IMP', '')}
    item(all(mt.values()), 'matchtype=MP and matchtype=IMP are accepted, and an absent one',
         f"answers: {mt} (that an absent one is read as IMP cannot be seen from outside)")
    timed = client.get('/bid', dict(base_bid, **TIME), 'time parameters')
    item(bid_of(timed) is not None, 'a request with the five time parameters is answered, and one without', f"answer: {bid_of(timed)}")

    # lead, play, dummy, original hands: from the whole boards
    done = [f for f in finished if f and 'played' in f]
    leads = [e for e in client.log if e['path'] == '/lead']
    item(bool(leads) and not any('card 1 ' in p for p in results['card_problems']), '/lead answers with a card from the hand it was given',
         '; '.join(f"{e['note']}: {e['body'].get('card') if isinstance(e['body'], dict) else e['body']}" for e in leads))
    item(len(done) == len(BOARDS) and not results['card_problems'], '/play answers with a legal card for all 51 remaining cards of a board',
         '; '.join(results['card_problems']) or '; '.join(f"board {p['board']}: {p['contract']}, {p['cards']} cards" for p in results['boards_played']))
    item(not any('does not follow' in p for p in results['card_problems']) and bool(done), '/play follows suit whenever the deciding hand holds the suit led')
    item(results['dummy_cards'] >= 13 * len(done) and bool(done), "dummy's cards are answered from a request carrying seat/hand = declarer",
         f"{results['dummy_cards']} dummy cards asked of declarer")
    item(bool(done) and not results['card_problems'], 'hands are read as the original 13 cards throughout',
         'every /play carried the original 13-card hand and dummy')

    # board over, passed-out board, next board
    if done:
        f = done[0]
        over = client.get('/play', {'hand': f['declarer_hand'], 'seat': f['declarer'], 'dummy': f['dummy_hand'], 'played': f['played'],
                                    'dealer': f['dealer'], 'ctx': f['ctx'], 'vul': f['vul'], 'board': f['board']}, 'board over')
        passed = client.get('/play', {'hand': b0['hands']['N'], 'seat': 'N', 'played': '', 'dealer': 'N', 'ctx': '--------',
                                      'vul': b0['vul'], 'board': 'passed-out'}, 'passed out')
        after = client.get('/bid', dict(base_bid, matchtype='IMP'), 'next board after board over')
        ok_over = 200 <= over['status'] < 300 and not (isinstance(over['body'], dict) and over['body'].get('card'))
        ok_passed = 200 <= passed['status'] < 300 and not (isinstance(passed['body'], dict) and passed['body'].get('card'))
        item(ok_over and ok_passed and bid_of(after) is not None, 'a /play with all 52 cards (or a passed-out ctx) is answered 2xx without a card, and the next board normally',
             f"52 cards: {over['status']} {str(over['body'])[:120]}; passed out: {passed['status']} {str(passed['body'])[:120]}; next: {bid_of(after)}")

    # the same request twice, with another seat's request in between
    twice_a = client.get('/bid', dict(base_bid, matchtype='IMP'), 'repeat 1')
    client.get('/bid', {'hand': BOARDS[1]['hands']['S'], 'seat': 'S', 'dealer': 'S', 'vul': 'None', 'ctx': '', 'board': 'other'}, 'another table')
    twice_b = client.get('/bid', dict(base_bid, matchtype='IMP'), 'repeat 2')
    item(bid_of(twice_a) is not None and bid_of(twice_a) == bid_of(twice_b), 'the same request sent twice returns the same answer',
         f"{bid_of(twice_a)} and {bid_of(twice_b)}")
    if done:
        f = done[0]
        cut = f['played'][:2 * 22]      # a position in the middle of the play
        params = {'hand': f['declarer_hand'], 'seat': f['declarer'], 'dummy': f['dummy_hand'], 'played': cut, 'dealer': f['dealer'],
                  'ctx': f['ctx'], 'vul': f['vul'], 'board': f['board'], 'matchtype': 'IMP'}
        # only meaningful when declarer or dummy is to play after 22 cards; the answer must still be a held card or an error
        first = client.get('/play', params, 'play repeat 1')
        client.get('/bid', dict(resp), 'another table')
        second = client.get('/play', params, 'play repeat 2')
        item(None, 'four seats served by one process do not influence one another',
             'tested as: a request, another table\'s request, the first again. /bid: same answer. ' +
             f"/play after 22 cards: {first['status']} {str(first['body'])[:60]} and {second['status']} {str(second['body'])[:60]}")

    # malformed requests
    bad = [('no hand', '/bid', {k: v for k, v in base_bid.items() if k != 'hand'}),
           ('12 cards', '/bid', dict(base_bid, hand='AKQ82.AJ4.KT9.7')),
           ('illegal ctx', '/bid', dict(base_bid, ctx='1S1C')),
           ('no ctx on /lead', '/lead', {'hand': b0['hands']['E'], 'seat': 'E', 'dealer': 'N', 'vul': 'NS'}),
           ('wrong seat on /lead', '/lead', {'hand': b0['hands']['W'], 'seat': 'W', 'dealer': 'N', 'vul': 'NS', 'ctx': '1S--2S--4S------'}),
           ('unknown vul', '/bid', dict(base_bid, vul='Purple')),
           ('/play asked of dummy', '/play', {'hand': b0['hands']['S'], 'seat': 'S', 'dummy': b0['hands']['N'], 'played': 'D6',
                                             'dealer': 'N', 'ctx': '1S--2S--4S------', 'vul': 'NS'})]
    if done:
        f = done[0]
        ok_play = {'hand': f['declarer_hand'], 'seat': f['declarer'], 'dummy': f['dummy_hand'], 'played': f['played'][:2],
                   'dealer': f['dealer'], 'ctx': f['ctx'], 'vul': f['vul']}
        bad += [('/play with a 12-card hand', '/play', dict(ok_play, hand=ok_play['hand'][:-1])),
                ('/play without dummy', '/play', {k: v for k, v in ok_play.items() if k != 'dummy'}),
                ('/play with a card that does not exist', '/play', dict(ok_play, played='ZZ')),
                ('/play with the same card twice', '/play', dict(ok_play, played=f['played'][:2] * 2)),
                ('/play with an unknown vul', '/play', dict(ok_play, vul='Purple'))]
    # Hands that are not 13 different cards, on /lead. Sent only to a server that refused the
    # 12-card hand on /bid: BEN without that check never answers a 12-card /lead (its sampler
    # loops for ever) and answers nothing else until it is restarted. Seen on prod, 2026-11-19.
    lead_ok = {'seat': 'E', 'dealer': 'N', 'vul': 'NS', 'ctx': '1S--2S--4S------'}
    bad_hands = [('/lead with a 12-card hand', '/lead', dict(lead_ok, hand='J9.KQ98.876.J43')),
                 ('/lead with a card twice', '/lead', dict(lead_ok, hand='J9.KQ98.8766.J43')),
                 ('/lead with an unknown rank', '/lead', dict(lead_ok, hand='J9.KQ98.876.J43Z')),
                 ('/bid with a card twice', '/bid', dict(base_bid, hand='AKQ82.AJ4.KT99.7'))]
    outcomes = []
    for name, path, params in bad + bad_hands:
        if (name, path, params) in bad_hands and not all(g for n, g, _, _ in outcomes if n == '12 cards'):
            outcomes.append((name, False, 'not sent', 'the 12-card hand on /bid was not refused'))
            continue
        r = client.get(path, params, f'malformed: {name}')
        good = not 200 <= r['status'] < 300 and isinstance(r['body'], dict) and 'error' in r['body']
        outcomes.append((name, good, r['status'], str(r['body'])[:90]))
    item(all(g for _, g, _, _ in outcomes), 'a malformed request returns non-2xx with an error field',
         '; '.join(f"{n}: {s} {body}" for n, g, s, body in outcomes))

    # advertised names
    if caps.get('playsSystems') and not args.skip_names:
        mid = {'/bid': dict(base_bid, matchtype='IMP'),
               '/lead': {'hand': b0['hands']['E'], 'seat': 'E', 'dealer': 'N', 'vul': 'NS', 'ctx': '1S--2S--4S------'}}
        if done:
            f = done[0]
            mid['/play'] = {'hand': f['declarer_hand'], 'seat': f['declarer'], 'dummy': f['dummy_hand'], 'played': f['played'][:2],
                            'dealer': f['dealer'], 'ctx': f['ctx'], 'vul': f['vul']}
        tries = ([('ourSystem', n) for n in caps.get('playsSystems', [])] + [('theirSystem', n) for n in caps.get('readsSystems', [])] +
                 [(p, n) for p in ('ourCarding', 'theirCarding') for n in caps.get('cardings', [])])
        failed, count = [], 0
        for path, params in mid.items():
            for key, name in tries:
                r = client.get(path, dict(params, **{key: name}), f'{key}={name}')
                count += 1
                answer = isinstance(r['body'], dict) and (r['body'].get('bid') or r['body'].get('card'))
                if r['status'] != 200 or not answer:
                    failed.append(f"{path} {key}={name}: {r['status']} {str(r['body'])[:80]}")
        unknown = [client.get(path, dict(params, ourSystem='NO-SUCH-SYSTEM'), 'unknown system') for path, params in mid.items()]
        unknown_ok = all(u['status'] == 400 and isinstance(u['body'], dict) and 'error' in u['body'] for u in unknown)
        item(not failed and unknown_ok, 'every advertised system and carding name is accepted on all three endpoints, and an unlisted name returns 400 with an error',
             f"{count} named requests, {len(failed)} refused" + (': ' + '; '.join(failed[:6]) if failed else '') +
             f"; unlisted name: {[u['status'] for u in unknown]} {str(unknown[0]['body'])[:100]}")
        item(None, 'defaults', f"GET / defaults: {caps.get('defaults')}")

    # token, https, CORS
    item(None, 'bearer token', 'the server checks none; ' + ('address is https with a certificate Python accepted' if client.base.startswith('https://') else 'address is not https'))
    watched = [e for e in client.log if e['path'] in ('/', '/bid', '/lead', '/play')]
    missing = [e for e in watched if not e['cors']]
    item(not missing, 'every response on /, /bid, /lead and /play carries Access-Control-Allow-Origin',
         f"{len(watched)} responses, {len(missing)} without the header" +
         (': ' + '; '.join(sorted({f"{e['path']} {e['status']} ({e['note']})" for e in missing})[:6]) if missing else ''))

    ended = datetime.now()
    slow = sorted(client.log, key=lambda e: -e['seconds'])[:3]
    by_path = {p: [e['seconds'] for e in client.log if e['path'] == p and e['status'] == 200] for p in ('/bid', '/lead', '/play')}
    print('timing: ' + '; '.join(f"{p} {len(v)} answers, median {sorted(v)[len(v) // 2]:.1f} s, longest {max(v):.1f} s" for p, v in by_path.items() if v))
    print('slowest: ' + '; '.join(f"{e['seconds']} s {e['path']} ({e['note']})" for e in slow))
    counts = {m: sum(1 for l in lines if l[0] == m) for m in ('PASS', 'FAIL', 'NOTE')}
    print(f"end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}  {len(client.log)} requests  {counts}")
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'base': client.base, 'started': f'{started:%Y-%m-%d %H:%M:%S}', 'items': lines, 'results': results,
                       'requests': client.log}, f, ensure_ascii=False, indent=1)
        print(f'log: {args.out}')
    sys.exit(1 if counts['FAIL'] else 0)


if __name__ == '__main__':
    main()
