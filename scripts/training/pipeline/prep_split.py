"""Split an OK_boards PBN into train / held-out test / training sample, and add
double-dummy tables to the two evaluation files (eval_bidding_nn.py needs them).

Usage: python prep_split.py boards.pbn outdir [n_eval=20000] [processes]

test.pbn    every k-th board, never trained on
train.pbn   the rest, minus boards with the same four hands as a test board
            (the same deal with another dealer or turned one seat)
sample.pbn  n_eval boards spread evenly over train.pbn

A file that already exists in outdir is kept (the split is the same every time).
The tables are solved by single-threaded worker processes: DDS with all its
threads in one process was 10 times slower here and loaded every core.
processes defaults to half the logical CPUs, at most 64. On the 64-core P620, 4,096 boards
took 90 s with 16 processes, 43 s with 64 and 52 s with 120 (2026-11-17).
"""
import sys, os, re, time
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src"))

HEADER = "% PBN 2.1\n% EXPORT\n\n"
DEAL = re.compile(r"\[Deal \"([NESW]):([^\"]+)\"\]")
_dds3 = None


def hands_key(board):
    return frozenset(DEAL.search(board).group(2).split())


def solve_chunk(deals):
    global _dds3
    if _dds3 is None:
        from ddsolver.ddsolver import dds3
        _dds3 = dds3
    return [t["res_table"] for t in _dds3.calc_all_tables_pbn(deals, mode=-1, max_threads=1)["tables"]]


def add_tables(part, processes):
    deals = [":".join(DEAL.search(b).groups()) for b in part]
    chunks = [deals[i:i + 8] for i in range(0, len(deals), 8)]
    with mp.get_context("spawn").Pool(processes) as pool:
        tables = [t for chunk in pool.imap(solve_chunk, chunks) for t in chunk]
    result = []
    for b, rows in zip(part, tables):          # rows: strain S H D C NT x hand N E S W
        lines = ["[OptimumResultTable \"Declarer;Denomination\2R;Result\2R\"]"]
        for h, hand in enumerate("NESW"):
            for s, strain in ((4, "NT"), (0, "S"), (1, "H"), (2, "D"), (3, "C")):
                lines.append(f"{hand} {strain:>2} {rows[s][h]:>2}")
        result.append(b + "\n" + "\n".join(lines))
    return result


def write(path, part):
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n\n".join(part) + "\n\n")


def main():
    src, out = sys.argv[1], sys.argv[2]
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    processes = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, min(64, (os.cpu_count() or 2) // 2))

    text = open(src, encoding="utf-8").read()
    boards = ["\n".join(l for l in b.split("\n") if not l.startswith("%")).strip("\n")
              for b in text.split("\n\n") if "[Deal " in b]
    del text

    step = max(2, len(boards) // n_eval)
    test_idx = set(range(step // 2, len(boards), step))
    test = [boards[i] for i in sorted(test_idx)]
    test_keys = {hands_key(b) for b in test}
    train, same_deal = [], 0
    for i, b in enumerate(boards):
        if i in test_idx:
            continue
        if hands_key(b) in test_keys:
            same_deal += 1
            continue
        train.append(b)
    sample = train[::max(1, len(train) // n_eval)][:n_eval]
    print(f"{len(boards)} boards: {len(train)} train, {len(test)} test, "
          f"{same_deal} left out (same deal as a test board), {len(sample)} sample", flush=True)

    path = os.path.join(out, "train.pbn")
    if not os.path.exists(path):
        write(path, train)
    for name, part in (("test.pbn", test), ("sample.pbn", sample)):
        path = os.path.join(out, name)
        if os.path.exists(path):
            print(f"{name}: kept", flush=True)
            continue
        t0 = time.time()
        write(path + ".tmp", add_tables(part, processes))
        os.rename(path + ".tmp", path)
        print(f"{name}: double-dummy tables for {len(part)} boards in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
