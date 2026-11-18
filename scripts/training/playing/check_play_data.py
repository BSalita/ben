"""
Check a play file (4 lines per table: deal "W:...", "dealer vuln contract.result.declarer",
auction, play) for records where the cards played do not come from the hands that the
declarer letter says played them.

Usage: python check_play_data.py <play.txt> [<play.txt> ...] [max=0] [show=3]

For every record it finds who holds the card led to trick 1 (the real opening leader) and
compares that seat with the left-hand opponent of the declarer given in the record. It also
replays the whole record from the real leader and counts the cards that are not in the hand
whose turn it is.
"""
import sys
from collections import Counter

SEATS = 'WNES'            # order of the hands in the deal string

args = [a for a in sys.argv[1:] if '=' not in a]
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
max_records = int(opts.get('max', 0))
show = int(opts.get('show', 3))


def records(path):
    with open(path) as f:
        while True:
            lines = [f.readline().strip() for _ in range(4)]
            if not lines[0]:
                return
            yield lines


def winner(trick, trump):
    led = trick[0][0]
    best, best_value = 0, -1
    for i, card in enumerate(trick):
        value = 14 - 'AKQJT98765432'.index(card[1])
        if card[0] == trump:
            value += 13
        elif card[0] != led:
            value = 0
        if value > best_value:
            best, best_value = i, value
    return best


n = 0
offsets = Counter()          # (real leader - leader from the declarer letter) % 4
auction_offsets = Counter()  # the same against the leader that the dealer letter and the auction give
by_declarer = Counter()
bad_cards = Counter()        # records by number of cards not in the hand on turn, replayed from the real leader
no_holder = 0
shown = 0
for path in args:
    for deal, meta, auction, play in records(path):
        if max_records and n >= max_records:
            break
        n += 1
        hands = [set(suit + c for suit, cards in zip('SHDC', hand.split('.')) for c in cards) for hand in deal[2:].split()]
        dealer, vuln, outcome = meta.split()
        contract, _, declarer = outcome.split('.')
        cards = [play[i:i + 2] for i in range(0, len(play), 2)]
        holder = next((i for i, h in enumerate(hands) if cards[0] in h), None)
        if holder is None:
            no_holder += 1
            continue
        stated = (SEATS.index(declarer) + 1) % 4
        off = (holder - stated) % 4
        offsets[off] += 1
        # The declarer that the dealer letter and the auction give
        calls = auction.split()
        bids = [j for j, call in enumerate(calls) if call not in ('PP', 'DD', 'RR')]
        if bids:
            last = bids[-1]
            first = next(j for j in bids if j % 2 == last % 2 and calls[j][1] == calls[last][1])
            auction_offsets[(holder - (SEATS.index(dealer) + first + 1)) % 4] += 1
        by_declarer[(declarer, off)] += 1
        # Replay from the real leader
        on_lead, missing = holder, 0
        for t in range(0, len(cards) - 3, 4):
            trick = cards[t:t + 4]
            for i, card in enumerate(trick):
                if card not in hands[(on_lead + i) % 4]:
                    missing += 1
            on_lead = (on_lead + winner(trick, contract[1])) % 4
        bad_cards[min(missing, 5)] += 1
        if off and shown < show:
            shown += 1
            print("Declarer letter does not match the play:")
            print("  ", deal)
            print("  ", meta)
            print("  ", auction)
            print("  ", play)
            print(f"   {cards[0]} is held by {SEATS[holder]}, so declarer is {SEATS[(holder - 1) % 4]}; the record says {declarer}")

print()
print(f"{n} records, {no_holder} where nobody holds the card led")
print("Real opening leader against the leader the declarer letter gives (seats clockwise):")
for off in range(4):
    print(f"   {off} seats on: {offsets[off]:8d}  {100 * offsets[off] / max(1, sum(offsets.values())):5.1f}%")
print("By declarer letter (letter, seats on): ", dict(sorted(by_declarer.items())))
print("Real opening leader against the leader the dealer letter and the auction give:")
for off in range(4):
    print(f"   {off} seats on: {auction_offsets[off]:8d}  {100 * auction_offsets[off] / max(1, sum(auction_offsets.values())):5.1f}%")
print("Cards not in the hand on turn when replayed from the real leader (5 = five or more):")
for k in sorted(bad_cards):
    print(f"   {k}: {bad_cards[k]:8d}  {100 * bad_cards[k] / max(1, sum(bad_cards.values())):5.1f}%")
