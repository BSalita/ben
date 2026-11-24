#!/usr/bin/env python3
"""brill_replay.py with the morty-book bidding South instead of a BEN configuration.

A measurement aid: at each South turn the compiled book (compile_book.CompiledBook, every
table of coverage.py, first row that fires) makes the call. When no row fires or the row's call
is illegal, South passes and the turn is counted as a fallback (as morty-book's book_match.py
does). North, East and West are the real robots while the auction stays on record, then
--robots, as in brill_replay.py. --conf is still loaded by the workers but not used for South.

    python book_south_replay.py --book CODE/book/chapters <brill_replay.py arguments>

CODE is a morty-book folder holding compile_book.py, coverage.py and features.py.
Each line of replay.jsonl gets 'book_fallbacks' (South turns on the way where the book passed
for lack of a legal call) and 'book_tables' (the table of each of South's calls on the way).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brill_replay  # noqa: E402

_real_init = brill_replay.worker_init
_real_replay = brill_replay.replay


def worker_init(cfg):
    _real_init(cfg)
    chapters = os.environ['BOOK_CHAPTERS']
    code = os.path.dirname(os.path.dirname(chapters))
    sys.path.insert(0, code)
    import compile_book
    import coverage
    from bidding import bidding
    compile_book.TABLES = [(flag, name) for flag, name, _ in coverage.TABLES]
    book = compile_book.CompiledBook(chapters)
    W = brill_replay.W
    W['book_seen'] = {}

    def south_call(board, hand, calls):
        dealer_i = brill_replay.SEATS.index(board['dealer'])
        call, table, rule = book.bid(hand, list(calls), board['vul'], board['scoring'])
        auction = ['PAD_START'] * dealer_i + [brill_replay.to_ben(c) for c in calls]
        fallback = call is None or not bidding.can_bid(brill_replay.to_ben(call), auction)
        if fallback:
            call = 'P'
        W['book_seen']['-'.join(calls)] = (fallback, table)
        return call

    W['south_call'] = south_call
    brill_replay.replay = replay     # spawned workers import brill_replay afresh
    print(f'worker {os.getpid()}: South is the book {chapters}')


def replay(board):
    W = brill_replay.W
    W['book_seen'] = {}
    out = _real_replay(board)
    walk = out['walk']
    dealer_i = brill_replay.SEATS.index(board['dealer'])
    fallbacks, tables = 0, []
    for i in range(len(walk)):
        if (dealer_i + i) % 4 == 2:
            fb, table = W['book_seen'].get('-'.join(walk[:i]), (False, None))
            fallbacks += fb
            tables.append(table)
    out['book_fallbacks'] = fallbacks
    out['book_tables'] = tables
    return out


def main():
    argv = sys.argv[1:]
    if '--book' not in argv:
        raise SystemExit('--book CHAPTERS is required')
    i = argv.index('--book')
    os.environ['BOOK_CHAPTERS'] = os.path.abspath(argv[i + 1])
    del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    brill_replay.worker_init = worker_init
    brill_replay.replay = replay
    return brill_replay.main()


if __name__ == '__main__':
    sys.exit(main())
