import ctypes
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import Counter
from objects import Card
from colorama import Fore, Back, Style, init
from nn.timing import ModelTimer
from ddsolver.ddsrecorder import DDSRecorder

init()


def _load_dds3():
    """Import the dds3 extension (DDS 3.x Python interface).

    Tries a normal import first (dds3 installed as a wheel). If that fails,
    falls back to a copy vendored under BEN's bin/ directory.
    """
    # Keep the real failure around. A vendored _dds3.so that exists but won't
    # load (e.g. built for a different Python version, or missing a dependent
    # shared library) raises ImportError here; swallowing it hides the cause.
    last_error = None
    try:
        import dds3
        return dds3
    except ImportError as ex:
        last_error = ex

    here = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        plat = "win"
    elif sys.platform == "darwin":
        plat = "darwin"
    else:
        plat = "linux"

    # Look for a vendored dds3 under bin/, covering the layouts BEN runs in:
    #   repo:              src/ddsolver/ddsolver.py -> <repo>/bin       (here/../..)
    #   flattened (Docker): /app/ddsolver/ddsolver.py -> /app/bin       (here/..)
    #   plus BEN_HOME/bin and <cwd>/bin as fallbacks.
    bin_roots = [
        os.path.join(os.path.abspath(os.path.join(here, "..", "..")), "bin"),
        os.path.join(os.path.abspath(os.path.join(here, "..")), "bin"),
    ]
    if os.getenv("BEN_HOME"):
        bin_roots.append(os.path.join(os.getenv("BEN_HOME"), "bin"))
    bin_roots.append(os.path.join(os.getcwd(), "bin"))

    seen = set()
    for root in bin_roots:
        for cand in (os.path.join(root, "dds3-" + plat),
                     os.path.join(root, "dds3"),
                     root):
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                import dds3
                return dds3
            except ImportError as ex:
                last_error = ex
                continue

    raise ImportError(
        "Could not import the 'dds3' extension (DDS 3.x Python interface).\n"
        "Build it from the DDS repository and install or vendor it:\n"
        "  bazel build -c opt //python:dds3_wheel_dist   # produces a wheel in dist/\n"
        "  pip install dist/dds3-*.whl\n"
        f"or place the built dds3 package in BEN's bin/dds3-{plat}/ directory.\n"
        f"A _dds3.so built for a different Python than {sys.version_info.major}."
        f"{sys.version_info.minor} will fail to load here.\n"
        "See docs/python_interface.md in the DDS repository.\n"
        f"\nUnderlying import error: {last_error!r}"
    ) from last_error


dds3 = _load_dds3()


def _gevent_patched_threading():
    try:
        from gevent import monkey
        return monkey.is_module_patched('threading')
    except ImportError:
        return False


# gameapi.py runs gevent's monkey.patch_all(), which turns the threads of a
# concurrent.futures pool into greenlets sharing one OS thread, so the solves ran
# one after the other (200 deals: 13 s instead of 2 s on 8 cores). Under gevent
# the pool is gevent's own executor, which always uses native threads. Found by
# grand-slam-bridge (PR #198).
_REAL_THREADS = _gevent_patched_threading()
if _REAL_THREADS:
    from gevent.monkey import get_original
    from gevent.threadpool import ThreadPoolExecutor as _NativeThreadPoolExecutor

# DDS 3.0.0 removed internal multi-threading from the legacy batch API
# (SolveAllBoards now solves sequentially). The modern model is one
# SolverContext per worker thread; DDSolver parallelises with a thread pool,
# and solve_board_pbn releases the GIL during the solve, so the threads run
# concurrently. Each pool thread keeps its own SolverContext - per OS thread,
# so under gevent the original thread-local, not the patched per-greenlet one.
_ctx_local = get_original('_thread', '_local')() if _REAL_THREADS else threading.local()


def _thread_context():
    ctx = getattr(_ctx_local, "ctx", None)
    if ctx is None:
        ctx = dds3.SolverContext()
        _ctx_local.ctx = ctx
    return ctx


class _CLibrarySolver:
    """Card play's solves through a shared library with DDS's C API (SolveBoardPBN), for example
    bridge-solver-cpp's libdds.so / dds.dll. Chosen with the environment variable BEN_DDS_LIBRARY
    (path of the library); without it BEN uses dds3. Only solve_board_pbn goes through it: par and
    DD tables stay on dds3. ctypes releases the GIL during the call, so the thread pool runs the
    solves in parallel as with dds3; the library must be thread-safe (DDS's API allows calls from
    several threads at once)."""

    class _DealPBN(ctypes.Structure):
        _fields_ = [("trump", ctypes.c_int), ("first", ctypes.c_int),
                    ("currentTrickSuit", ctypes.c_int * 3), ("currentTrickRank", ctypes.c_int * 3),
                    ("remainCards", ctypes.c_char * 80)]

    class _FutureTricks(ctypes.Structure):
        _fields_ = [("nodes", ctypes.c_int), ("cards", ctypes.c_int), ("suit", ctypes.c_int * 13),
                    ("rank", ctypes.c_int * 13), ("equals", ctypes.c_int * 13), ("score", ctypes.c_int * 13)]

    def __init__(self, path):
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        self.path = path
        self.lib = loader(path)
        self.lib.SolveBoardPBN.argtypes = [self._DealPBN, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(self._FutureTricks), ctypes.c_int]
        self.lib.SolveBoardPBN.restype = ctypes.c_int
        if hasattr(self.lib, "SetMaxThreads"):
            self.lib.SetMaxThreads(0)
        info = ""
        if hasattr(self.lib, "GetDDSInfo"):
            # DDSInfo: ints then char systemString[1024]; only the text is read
            buf = ctypes.create_string_buffer(4096)
            try:
                self.lib.GetDDSInfo(buf)
                info = buf.raw.decode("latin-1", "replace")
            except Exception:
                info = ""
        self.info = " ".join(info.replace("\x00", " ").split())[-200:]

    def solve_board_pbn(self, pbn, trump, first, current_trick_suit, current_trick_rank, target,
                        solutions, mode, context=None):
        dl = self._DealPBN()
        dl.trump = trump
        dl.first = first
        for i in range(3):
            dl.currentTrickSuit[i] = current_trick_suit[i]
            dl.currentTrickRank[i] = current_trick_rank[i]
        dl.remainCards = pbn.encode("ascii")
        fut = self._FutureTricks()
        ret = self.lib.SolveBoardPBN(dl, target, solutions, mode, ctypes.byref(fut), 0)
        if ret != 1:
            raise RuntimeError(f"SolveBoardPBN returned {ret} ({self.path})")
        n = fut.cards
        return {"cards": n, "suit": list(fut.suit[:n]), "rank": list(fut.rank[:n]),
                "equals": list(fut.equals[:n]), "score": list(fut.score[:n])}


_DDS_LIBRARY = os.getenv("BEN_DDS_LIBRARY")
_clib = _CLibrarySolver(_DDS_LIBRARY) if _DDS_LIBRARY else None

# Card play chooses among cards that score exactly the same by their place in the solver's list (the
# last key of the candidate sorts in botcardplayer.py), and solvers list equal cards in different
# orders. BEN_EQUAL_CARDS fixes that order, so the choice no longer depends on the solver:
#   lowest  - in each suit the lowest card first (suits in the order spades, hearts, diamonds, clubs),
#             so the lowest of equal cards is played
#   highest - the highest card first
# Unset: the solver's own order (BEN's behaviour until now).
_EQUAL_CARDS = (os.getenv("BEN_EQUAL_CARDS") or "").lower() or None
if _EQUAL_CARDS not in (None, "lowest", "highest"):
    raise ValueError(f"BEN_EQUAL_CARDS must be lowest or highest, not {_EQUAL_CARDS!r}")


def _fixed_card_order(card_results):
    """card_results (card code -> per-sample scores) in the BEN_EQUAL_CARDS order. Card code =
    suit * 13 + rank index, rank index 0 = ace."""
    if _EQUAL_CARDS is None:
        return card_results
    low_first = _EQUAL_CARDS == "lowest"
    order = sorted(card_results, key=lambda c: (c // 13, -(c % 13) if low_first else c % 13))
    return {c: card_results[c] for c in order}


class DDSolver:

    # Default for dds_mode changed to 1
    # Transport table will be reused if same trump suit and the same or nearly the same cards distribution, deal.first can be different.
    # Always search to find the score. Even when the hand to play has only one card, with possible equivalents, to play.
    # If zero, we not always find the score
    # If 2 transport tables ignore trump

    def __init__(self, dds_mode=1, max_threads=0, verbose=False):
        self.dds_mode = dds_mode
        # max_threads is the size of the solver thread pool.
        # 0 = one thread per CPU core.
        workers = max_threads if (max_threads and max_threads > 0) else (os.cpu_count() or 4)
        self._workers = workers
        if _REAL_THREADS:
            self._pool = _NativeThreadPoolExecutor(max_workers=workers)
        else:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dds")
        if verbose:
            sys.stderr.write(f"DDSolver loaded — DDS {self.version()} - dds mode {dds_mode} - {workers} solver threads"
                             f"{' (native threads under gevent)' if _REAL_THREADS else ''}\n")
        # No-op unless --ddsrecord / BEN_DDS_RECORD asked for a recording. If the
        # entry point already opened one, this just labels it with the DDS build
        # actually in use. See ddsrecorder.py / ddsreplay.py.
        DDSRecorder.configure(dds_version=self.version(), dds_mode=dds_mode,
                              threads=workers)

    def version(self):
        # the solver of card play's solves: dds3 unless BEN_DDS_LIBRARY names a library
        return "3.1.0" if _clib is None else f"library {_clib.path} {_clib.info}".strip()

    def calculatepar(self, hand, vuln, print_result=True):
        with ModelTimer.time_call('dds_par'):
            t0 = time.perf_counter()
            result = self._calculatepar_impl(hand, vuln, print_result)
            if DDSRecorder.enabled:
                DDSRecorder.record_par(hand, vuln, result,
                                       (time.perf_counter() - t0) * 1000)
            return result

    def _calculatepar_impl(self, hand, vuln, print_result=True):
        # vulnerable
        # 0: None 1: Both 2: NS 3: EW
        v = 0
        if vuln[0]: v = 2
        if vuln[1]: v = 3
        if vuln[0] and vuln[1]: v = 1

        # calc_all_tables_pbn computes the DD table and (mode != -1) the par score.
        try:
            result = dds3.calc_all_tables_pbn(["N:" + hand], mode=v)
        except Exception as e:
            sys.stderr.write(f"Error calculating par: {e}, Hand {hand.encode('utf-8')}\n")
            raise

        par_results = result.get("par_results") or []
        if not par_results:
            sys.stderr.write(f"{Fore.RED}No par result for hand {hand.encode('utf-8')}{Style.RESET_ALL}\n")
            return None

        par = par_results[0]
        # par_score is a 2-element list of strings, e.g. "NS 420" / "EW -420".
        ns_par = par["par_score"][0]
        ew_par = par["par_score"][1]

        if print_result:
            print("NS score: {}".format(ns_par))
            print("EW score: {}".format(ew_par))

        return int(ns_par.split()[1])

    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    @staticmethod
    def _trick_number(hands_pbn, current_trick):
        """Derive trick number (1-13) from remaining cards in PBN hand."""
        pbn = hands_pbn[0]
        if ':' in pbn:
            pbn = pbn.split(':', 1)[1]
        remaining = sum(1 for c in pbn if c not in '. ')
        return (52 - remaining - len(current_trick)) // 4 + 1

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions, purpose=""):
        """Solve a batch of sampled hands with double-dummy.

        purpose is a short tag describing *why* DDS was invoked (e.g. "play",
        "claimcheck", "lead", "bid"). It is folded into the timing label so the
        MODEL TIMING SUMMARY separates, say, card-play evaluations from claim
        checks at the same trick instead of lumping them into one opaque count.
        """
        trick = self._trick_number(hands_pbn, current_trick)
        label = f'dds_solve_t{trick:02d}_{purpose}' if purpose else f'dds_solve_t{trick:02d}'
        with ModelTimer.time_call(label, items=len(hands_pbn)):
            t0 = time.perf_counter()
            results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn, solutions)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        if DDSRecorder.enabled:
            DDSRecorder.record_solve(strain_i, leader_i, current_trick, hands_pbn,
                                     solutions, purpose, trick, results, elapsed_ms)

        return results

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        trump = (strain_i - 1) % 5

        # The current trick is the same for every board in the batch.
        trick_suit = [0, 0, 0]
        trick_rank = [0, 0, 0]
        for i in range(min(3, len(current_trick))):
            trick_suit[i] = current_trick[i] // 13
            trick_rank[i] = 14 - current_trick[i] % 13
        trick_suit = tuple(trick_suit)
        trick_rank = tuple(trick_rank)
        dds_mode = self.dds_mode

        # Solve each board on the thread pool. solve_board_pbn releases the GIL
        # during the DDS search; each pool thread uses its own SolverContext.
        # With BEN_DDS_LIBRARY set, the same solves go through that library instead (_CLibrarySolver).
        def _solve_one_clib(pbn):
            return _clib.solve_board_pbn(pbn, trump, leader_i, trick_suit, trick_rank, -1, solutions, dds_mode)

        def _solve_one(pbn):
            if _clib is not None:
                return _solve_one_clib(pbn)
            return dds3.solve_board_pbn(
                pbn,
                trump=trump,
                first=leader_i,
                current_trick_suit=trick_suit,
                current_trick_rank=trick_rank,
                target=-1,
                solutions=solutions,
                mode=dds_mode,
                context=_thread_context(),
            )

        try:
            solved = list(self._pool.map(_solve_one, hands_pbn))
        except Exception as e:
            print(f"{Fore.RED}DDS error: {e} {hands_pbn[0].encode('utf-8')} {current_trick} {leader_i}{Style.RESET_ALL}")
            return None

        if solutions == 1:
            # Just return the maximum number of the side to play for each sample
            card_results = {}
            card_results["max"] = []
            card_results["min"] = []
            for fut in solved:
                card_results["max"].append(fut["score"][0])
                card_results["min"].append(fut["score"][fut["cards"] - 1])

        else:
            card_results = {}
            for fut in solved:
                for i in range(fut["cards"]):
                    suit_i = fut["suit"][i]
                    card = suit_i * 13 + 14 - fut["rank"][i]
                    if card not in card_results:
                        card_results[card] = []
                    card_results[card].append(fut["score"][i])
                    eq_cards_encoded = fut["equals"][i]
                    for k, rank_code in enumerate(card_rank):
                        if rank_code & eq_cards_encoded > 0:
                            eq_card = suit_i * 13 + k
                            if eq_card not in card_results:
                                card_results[eq_card] = []
                            card_results[eq_card].append(fut["score"][i])

        return card_results if solutions == 1 else _fixed_card_order(card_results)


    def solve_leads(self, strain_i, leader_i, hands_pbn, purpose=""):
        """Every legal opening lead of each board: a list with one dict {card code: tricks of the
        leader's side} a board. solve() with solutions = 3 collects all boards' cards in one dict,
        which only lines up when the leader holds the same cards on every board; here the leaders'
        hands may differ from board to board (the sampled deals of the bidding search)."""
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]
        trump = (strain_i - 1) % 5
        dds_mode = self.dds_mode
        clib = globals().get("_clib")

        def _solve_one(pbn):
            if clib is not None:
                fut = clib.solve_board_pbn(pbn, trump, leader_i, (0, 0, 0), (0, 0, 0), -1, 3, dds_mode)
            else:
                fut = dds3.solve_board_pbn(pbn, trump=trump, first=leader_i, current_trick_suit=(0, 0, 0),
                                           current_trick_rank=(0, 0, 0), target=-1, solutions=3, mode=dds_mode,
                                           context=_thread_context())
            cards = {}
            for i in range(fut["cards"]):
                suit_i = fut["suit"][i]
                cards[suit_i * 13 + 14 - fut["rank"][i]] = fut["score"][i]
                for k, rank_code in enumerate(card_rank):
                    if rank_code & fut["equals"][i] > 0:
                        cards[suit_i * 13 + k] = fut["score"][i]
            return cards

        label = f'dds_solve_leads_{purpose}' if purpose else 'dds_solve_leads'
        with ModelTimer.time_call(label, items=len(hands_pbn)):
            return list(self._pool.map(_solve_one, hands_pbn))

    def expected_tricks_dds(self, card_results):
        return {card:round((sum(values)/len(values)),2) for card, values in card_results.items()}

    def expected_tricks_dds_probability(self, card_results, probabilities_list : List[float]):
        # Convert to plain Python list to avoid numpy scalar overhead
        probs = [float(p) for p in probabilities_list]
        return {card: round(sum(p*res for p, res in zip(probs, result_list)),2) for card, result_list in card_results.items()}

    def p_made_target(self, tricks_needed):

        def fun(card_results):
            return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
        return fun

    def print_dd_results(self, dd_solved, print_result=True, xcards=False):
        print("DD Result\n".join(
            f"{Card.from_code(int(k))}: [{', '.join(f'{x:>2}' for x in v[:20])}{' ]' if len(v) <= 20 else ' ...]'}"
            for k, v in dd_solved.items()
        ))

        # Create a new dictionary to store sorted counts for each key
        sorted_counts_dict = {}

        # Loop through the dictionary and process each key-value pair
        for key, array in dd_solved.items():
            # Use Counter to count the occurrences of each element
            element_count = Counter(array)

            # Sort the counts by frequency in descending order
            sorted_counts = sorted(element_count.items(), key=lambda x: x[1], reverse=True)

            # Store the sorted result in the new dictionary
            sorted_counts_dict[key] = sorted_counts

        # Print the sorted counts for each key
        for key, sorted_counts in sorted_counts_dict.items():
            print(f"Sorted counts for {Card.from_code(int(key), xcards)} DD:")
            for value, count in sorted_counts:
                print(f"  Tricks: {value}, Count: {count}")
