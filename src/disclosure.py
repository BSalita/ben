"""The explanation sent with a call, with its numbers widened to what our calls show in practice.

EPBot writes the explanation from the convention card (bba/BBA.py, explain_last_bid):

    Bidable suit -- 5+!S; 6-!H; 9-11 HCP; Forcing

Our calls come from a network and a search, not from EPBot, and even EPBot's own calls
stray from its numbers (its point count is not plain high-card points). A calibration
file says, for each kind of explanation, by how much the hands we really hold fall outside
the stated numbers: cards below a stated least, cards above a stated most, points below,
points above. calibrate_text() widens the numbers of a text by those amounts. The
widening depends on the call and its meaning only, never on the hand held, so the text
tells nothing about the hand that the method does not show anyway.

The file is fitted and checked with the disclosure-audit tool (`disclosure-audit
calibrate`); [sampling] disclosure_calibration = <file> switches it on for the REST API.
"""
import json
import re

_tables = {}

_LENGTH = re.compile(r'^(?:(\d+)-(\d+)|(\d+)=|(\d+)\+|(\d+)-)(!?[CDHS])$')
_POINTS = re.compile(r'^(?:(\d+)-(\d+)|(\d+)\+|(\d+)-) HCP$')


def load(path):
    """The calibration table of a file (cached): {'heads': {'bid|Bidable suit': [l-, l+, p-, p+]}, 'fallback': {kind: [...]}}"""
    if path not in _tables:
        with open(path, encoding='utf-8') as f:
            _tables[path] = json.load(f)
    return _tables[path]


def kind_of(call):
    return 'pass' if call in ('PASS', 'P') else 'double' if call in ('X', 'XX') else 'bid'


def slack_for(table, kind, head):
    return table.get('heads', {}).get(f'{kind}|{head}') or table.get('fallback', {}).get(kind) or [0, 0, 0, 0]


def _length_text(lo, hi, suit):
    # as BBABotBid.extract_lengths
    if lo <= 0:
        return f'{hi}-{suit}' if hi < 13 else None
    if hi >= 13:
        return f'{lo}+{suit}'
    return f'{lo}={suit}' if lo == hi else f'{lo}-{hi}{suit}'


def _points_text(lo, hi):
    # as BBABotBid.extract_hcp
    if lo > 0:
        return f'{lo}-{hi} HCP' if hi < 37 else f'{lo}+ HCP'
    return f'{hi}- HCP' if hi < 37 else None


def calibrate_text(text, call, table):
    """`text` of explain_last_bid for `call`, with its suit lengths and point range widened by the table."""
    if not text or ' -- ' not in text and not text.lstrip().startswith('--'):
        return text
    words, _, numbers = text.partition('--')
    head = words.strip() or '(numbers only)'
    less, more, lighter, heavier = slack_for(table, kind_of(call), head)
    if not (less or more or lighter or heavier):
        return text
    out = []
    for part in numbers.split(';'):
        token = part.strip()
        m = _LENGTH.match(token)
        if m:
            a, b, exact, least, most, suit = m.groups()
            lo, hi = (int(a), int(b)) if a else (int(exact), int(exact)) if exact else (int(least), 13) if least else (0, int(most))
            token = _length_text(max(lo - less, 0), min(hi + more, 13), suit)
        else:
            m = _POINTS.match(token)
            if m:
                a, b, least, most = m.groups()
                lo, hi = (int(a), int(b)) if a else (int(least), 37) if least else (0, int(most))
                token = _points_text(max(lo - lighter, 0), min(hi + heavier, 37))
        if token is not None:        # a number widened away is left out; an empty part stays, as explain_last_bid writes it
            out.append(token)
    # explain_last_bid writes "<words> -- <lengths>; <points>; <flags>" with a space after "--" and after each ";"
    return words + '-- ' + '; '.join(out)


def outside(text, hand):
    """What of `hand` ('S.H.D.C') is outside the numbers of an explanation text: ['2 S, stated 3+', ...]."""
    if not text or '--' not in text:
        return []
    held = {s: len(cards) for s, cards in zip('SHDC', hand.split('.'))}
    hcp = sum({'A': 4, 'K': 3, 'Q': 2, 'J': 1}.get(c, 0) for c in hand)
    wrong = []
    for part in text.partition('--')[2].split(';'):
        token = part.strip()
        m = _LENGTH.match(token)
        if m:
            a, b, exact, least, most, suit = m.groups()
            lo, hi = (int(a), int(b)) if a else (int(exact), int(exact)) if exact else (int(least), 13) if least else (0, int(most))
            n = held[suit[-1]]
            if n < lo or n > hi:
                wrong.append(f'{n} {suit[-1]}, stated {token[:-len(suit)]}')
            continue
        m = _POINTS.match(token)
        if m:
            a, b, least, most = m.groups()
            lo, hi = (int(a), int(b)) if a else (int(least), 37) if least else (0, int(most))
            if hcp < lo or hcp > hi:
                wrong.append(f'{hcp} HCP, stated {token[:-4]}')
    return wrong
