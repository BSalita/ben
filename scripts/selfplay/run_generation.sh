#!/usr/bin/env bash
# One generation of the self-play improvement loop. See README.md here.
#
#   bash run_generation.sh <generation name>      (long runs: nohup setsid bash run_generation.sh g001 > /dev/null 2>&1 &)
#
# explore -> targets -> fine-tune -> gate 1 (networks alone) -> gate 2 (whole settings) -> verdict.txt
# METHOD says how the explore games become a network: targets (the best call of each deal), value (a value
# network rates the calls and names the targets) or pg (policy gradient, no targets).
# Every step is skipped when its result exists, so the script can be started again after a stop.
# It never changes the incumbent: promote.sh does that, after the checks the README names.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
GEN=${1:?usage: run_generation.sh <generation name>}
SP_ROOT=${SP_ROOT:-/var/lib/ben/selfplay}
BEN=${BEN_ROOT:-$(cd "$HERE/../.." && pwd)}             # the engine; these scripts may sit outside it
[ -d "$BEN/src/bidding" ] || BEN=$(cd "$HERE/../.." && pwd)
INCUMBENT=${INCUMBENT:-$SP_ROOT/conf/incumbent.conf}     # the whole setting that plays now; its bidder is the network to improve
ROBOTS=${ROBOTS:-config/Brill-21GF.conf}                 # network of North, East and West
ALSO=${ALSO-$ROBOTS}                                     # configs whose first choice is one more candidate ("" for none)
TABLE=${TABLE:-brill}
N_EXPLORE=${N_EXPLORE:-200000}
SEED=${SEED:-$(( $(printf %s "$GEN" | cksum | cut -d' ' -f1) % 800000 + 1 ))}
CAND_MIN_P=${CAND_MIN_P:-0.03}
CAND_MAX=${CAND_MAX:-4}
INJECT=${INJECT:-}                                       # calls tried whatever the network thinks: double compete game all:N
CAND_LIMIT=${CAND_LIMIT:-12}
BOARD_CYCLE=${BOARD_CYCLE:-16}                           # dealer and vulnerability of boards 1..N
EXPLORE_FROM=${EXPLORE_FROM:-}                           # use the explore games of this earlier generation (or of this folder) instead of new ones
EXPLORE_ALSO=${EXPLORE_ALSO:-}                           # more finished explore runs to learn from as well: generation names or folders
SAMPLE=${SAMPLE:-}                                       # explore: draw the call made among the network's candidates, output to the power 1/SAMPLE
METHOD=${METHOD:-targets}                                # targets, value or pg: see selfplay_build.py, selfplay_value.py, selfplay_train.py --pg
VALUE_ARGS=${VALUE_ARGS:-}                               # METHOD=value: more arguments of selfplay_value.py, for example --scale 5 --margin 0.5
PG_COEF=${PG_COEF:-2}                                    # METHOD=pg: weight of the expected utility in the loss
PG_ANCHOR=${PG_ANCHOR:-0}                                # METHOD=pg: cross-entropy weight of the call made
TARGET=${TARGET:-best}                                   # best or pair, see selfplay_build.py
PRIOR=${PRIOR-auto}                                      # sp.npz of earlier generations to train on as well; auto = all there are
WEIGHT=${WEIGHT:-mix}
EPOCHS=${EPOCHS:-3}
LR=${LR:-0.0001}
SP_SHARE=${SP_SHARE:-0.25}
TRAIN_MODE=${TRAIN_MODE:-finetune}                       # finetune: from the incumbent's weights. scratch: a new network (EPOCHS=30 LR=0.001)
TRAIN_ARGS=${TRAIN_ARGS:-}                               # more arguments of selfplay_train.py, for example --keep 0.8 --decay-steps 3000
NO_SP=${NO_SP:-}                                         # 1 with TRAIN_MODE=scratch: retrain on the base set alone (the control)
N_GATE1=${N_GATE1:-100000}
GATE1_SEED=${GATE1_SEED:-900001}
GATE2=${GATE2:-auto}                                     # auto: only after gate 1 says better; always; never
GATE2_DEALS=${GATE2_DEALS:-}                             # a deal file; empty = N_GATE2 random deals of GATE2_SEED
N_GATE2=${N_GATE2:-10000}
GATE2_SEED=${GATE2_SEED:-900002}
WORKERS=${WORKERS:-$(( $(nproc) * 3 / 4 ))}
SEARCH_WORKERS=${SEARCH_WORKERS:-40}
TEST_PBN=${TEST_PBN:-}                                   # held-out boards of the system (with double dummy tables): drift check
BRILL_FIELD=${BRILL_FIELD:-}                             # past Brill events (field.jsonl): both settings bid them, to be scored against the real field

G=$SP_ROOT/gen/$GEN
mkdir -p "$G"
# The engine copies its networks to the temp folder for every worker and leaves them there when the
# workers are stopped (about 9 GB a search run). Give this run its own temp folder and empty it as we go.
export TMPDIR=$SP_ROOT/tmp/$GEN.$$
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
sweep() { rm -rf "$TMPDIR"/* 2> /dev/null; }
log() { echo "$(date "+%F %T") $*" | tee -a "$G/run.log"; }
fail() { log "FAILED: $*"; touch "$G/failed"; exit 1; }
rm -f "$G/failed" "$G/done"
[ -f "$INCUMBENT" ] || fail "no incumbent config $INCUMBENT"
[ -f "$SP_ROOT/base.npz" ] || fail "no $SP_ROOT/base.npz (selfplay_build.py base <bin dir> $SP_ROOT/base.npz)"
[ "$(grep -c '^bidder *=' "$INCUMBENT")" = 1 ] || fail "$INCUMBENT must have exactly one 'bidder =' line"
START=$(cd "$HERE" && python -c "import sys, sp_common; print(sp_common.NetSpec(sys.argv[1]).bidder_path)" "$INCUMBENT") \
    || fail "cannot read the bidder of $INCUMBENT"
INC_ID=$(cat "$INCUMBENT" | cksum | cut -d' ' -f1)
ALSO_ARGS=(); for a in $ALSO; do ALSO_ARGS+=(--also "$a"); done
for a in $INJECT; do ALSO_ARGS+=(--inject "$a"); done
EXPLORE=$G/explore
[ -n "$EXPLORE_FROM" ] && EXPLORE=$SP_ROOT/gen/$EXPLORE_FROM/explore
case "$EXPLORE_FROM" in /*) EXPLORE=$EXPLORE_FROM ;; esac    # or a folder with games.jsonl made elsewhere
case "$METHOD" in targets|value|pg) ;; *) fail "METHOD must be targets, value or pg, not $METHOD" ;; esac
GAMES=("$EXPLORE")
for a in $EXPLORE_ALSO; do
    case "$a" in /*) ;; *) a=$SP_ROOT/gen/$a/explore ;; esac
    [ -f "$a/complete" ] || fail "no finished explore run in $a (EXPLORE_ALSO)"
    GAMES+=("$a")
done
log "generation $GEN: incumbent $INCUMBENT (network $START), $N_EXPLORE deals of seed $SEED, table $TABLE"

# 1. Self-play with every candidate tried on the deal
if [ -n "$EXPLORE_FROM" ]; then
    [ -f "$EXPLORE/complete" ] || fail "generation $EXPLORE_FROM has no finished explore run"
elif [ ! -f "$G/explore/complete" ]; then
    log "explore: $N_EXPLORE deals, $WORKERS workers${INJECT:+, forced candidates: $INJECT}${SAMPLE:+, calls drawn with temperature $SAMPLE}"
    python "$HERE/selfplay_gen.py" --mode explore --conf "$INCUMBENT" --robots "$ROBOTS" "${ALSO_ARGS[@]}" --table "$TABLE" \
        --cand-min-p "$CAND_MIN_P" --cand-max "$CAND_MAX" --cand-limit "$CAND_LIMIT" --board-cycle "$BOARD_CYCLE" \
        --n "$N_EXPLORE" --seed "$SEED" --workers "$WORKERS" ${SAMPLE:+--sample "$SAMPLE"} \
        --out "$G/explore" --resume >> "$G/explore.log" 2>&1
    [ "$(wc -l < "$G/explore/games.jsonl")" -ge "$N_EXPLORE" ] || fail "explore stopped early, see explore.log"
    touch "$G/explore/complete"
    log "$(tail -1 "$G/explore.log")"
fi

# 2. Training targets (METHOD=targets), or the turns with every tried call and its result (value, pg)
if [ "$METHOD" = targets ] && [ ! -f "$G/sp.npz" ]; then
    python "$HERE/selfplay_build.py" games --conf "$INCUMBENT" --weight "$WEIGHT" --target "$TARGET" --out "$G/sp.tmp.npz" "${GAMES[@]}" \
        > "$G/build.log" 2>&1 || fail "building the targets, see build.log"
    mv "$G/sp.tmp.json" "$G/sp.json"; mv "$G/sp.tmp.npz" "$G/sp.npz"
    log "targets: $(grep -E '^(rows|targets that are not|imps a deal)' "$G/build.log" | tr '\n' ';')"
fi
if [ "$METHOD" != targets ] && [ ! -f "$G/rl.npz" ]; then
    python "$HERE/selfplay_build.py" decisions --conf "$INCUMBENT" --out "$G/rl.tmp.npz" "${GAMES[@]}" \
        > "$G/build.log" 2>&1 || fail "building the decision file, see build.log"
    mv "$G/rl.tmp.json" "$G/rl.json"; mv "$G/rl.tmp.npz" "$G/rl.npz"
    log "decisions: $(grep -E '^(deals|rows|tried calls:|turns where)' "$G/build.log" | tr '\n' ';')"
fi
# 2b. METHOD=value: a value network rates every tried call (on deals it has not seen) and names the targets
if [ "$METHOD" = value ] && [ ! -f "$G/sp.npz" ]; then
    log "value network: $VALUE_ARGS"
    python "$HERE/selfplay_value.py" --start "$START" --rl "$G/rl.npz" --weight "$WEIGHT" --out "$G/value" $VALUE_ARGS \
        > "$G/value.log" 2>&1 || fail "the value network, see value.log"
    cp "$G/value/sp.json" "$G/sp.json"; mv "$G/value/sp.npz" "$G/sp.npz"
    log "value network: $(grep -E '^(Unseen ratings|Targets)' "$G/value.log" | tr '\n' ';')"
fi
[ "$METHOD" = pg ] && cp "$G/rl.json" "$G/sp.json"

# 3. Fine-tune the incumbent's network on its system's boards plus the self-play targets
NAME=${NAME:-selfplay-$GEN}
if ! ls "$G/train/model/${NAME}_"*"-E$(printf %02d "$EPOCHS").keras" > /dev/null 2>&1; then
    SP_ARGS=(--sp "$G/sp.npz")
    [ "$METHOD" = pg ] && SP_ARGS=(--pg "$G/rl.npz" --pg-coef "$PG_COEF" --pg-anchor "$PG_ANCHOR" --pg-weight "$WEIGHT")
    if [ "$PRIOR" = auto ]; then PRIOR=$(ls "$SP_ROOT"/gen/*/sp.npz 2> /dev/null | grep -v "^$G/" | tr '\n' ' '); fi
    for p in $PRIOR; do SP_ARGS+=(--sp "$p"); done
    [ -n "$NO_SP" ] && SP_ARGS=()
    FROM=(--start "$START"); [ "$TRAIN_MODE" = scratch ] && FROM=(--scratch)
    log "fine-tune: $TRAIN_MODE ${SP_ARGS[*]} $TRAIN_ARGS"
    python "$HERE/selfplay_train.py" "${FROM[@]}" $TRAIN_ARGS --base "$SP_ROOT/base.npz" "${SP_ARGS[@]}" --out "$G/train" \
        --name "$NAME" --epochs "$EPOCHS" --lr "$LR" --sp-share "$SP_SHARE" > "$G/train.log" 2>&1 \
        || fail "fine-tuning, see train.log"
    [ "$METHOD" = pg ] && log "policy gradient, $(grep '^turns kept out:' "$G/train.log" | tail -1)"
fi
MODEL=$(ls -t "$G/train/model/${NAME}_"*"-E$(printf %02d "$EPOCHS").keras" | head -1)
sed -E "s|^bidder *=.*|bidder = $MODEL|" "$INCUMBENT" > "$G/candidate.conf"
log "candidate network: $MODEL"

# Optional: how far the candidate has moved from the system it was trained on
if [ -n "$TEST_PBN" ] && [ ! -s "$G/system_check.txt" ]; then
    (cd "$BEN/scripts/training/bidding" && python eval_bidding_nn.py "$MODEL" "$TEST_PBN" --ns 1 --ew 1 --n_cards 24 \
        2> "$G/system_check.err" | grep -v "^$") > "$G/system_check.txt" || log "system check failed, see system_check.err"
fi

# 4. Gate 1: the two networks alone, each South at the same table on fresh deals
gate() {   # gate <mode> <folder name> <deal arguments...>: candidate and incumbent, then the comparison
    local mode=$1 name=$2; shift 2
    local inc=$SP_ROOT/gate/$INC_ID/$name workers=$WORKERS
    [ "$mode" = search ] && workers=$SEARCH_WORKERS
    for side in incumbent candidate; do
        local conf=$INCUMBENT out=$inc
        [ "$side" = candidate ] && conf=$G/candidate.conf && out=$G/$name/candidate
        if [ ! -f "$out/complete" ]; then
            mkdir -p "$out"
            log "$name: $side, $mode mode, $workers workers"
            python "$HERE/selfplay_gen.py" --mode "$mode" --conf "$conf" --robots "$ROBOTS" "$@" --workers "$workers" \
                --out "$out" --resume >> "$out/gen.log" 2>&1 || fail "$name $side, see $out/gen.log"
            touch "$out/complete"
            sweep
        fi
    done
    python "$HERE/selfplay_compare.py" "$G/$name/candidate" "$inc" --name-a "candidate $GEN" --name-b incumbent \
        --out "$G/$name/compare.txt" > /dev/null || fail "$name comparison"
    log "$name: $(grep '^all deals' "$G/$name/compare.txt"); $(grep '^verdict' "$G/$name/compare.txt")"
}
verdict_of() { grep '^verdict' "$1" | cut -d' ' -f2-; }

gate net gate1 --n "$N_GATE1" --seed "$GATE1_SEED"
V1=$(verdict_of "$G/gate1/compare.txt")

# 5. Gate 2: the two whole settings (network, candidates, sampling, rollouts) at the same table
V2="not run"
if [ "$GATE2" = always ] || { [ "$GATE2" = auto ] && [ "$V1" = better ]; }; then
    if [ -n "$GATE2_DEALS" ]; then gate search gate2 --deals "$GATE2_DEALS"; else gate search gate2 --n "$N_GATE2" --seed "$GATE2_SEED"; fi
    V2=$(verdict_of "$G/gate2/compare.txt")
fi

# 6. Past events: both settings bid South on the boards of BRILL_FIELD, the robots' calls taken from the
#    record (and from the robots' network once South leaves it). Scored against the real field elsewhere.
if [ -n "$BRILL_FIELD" ] && [ -f "$BRILL_FIELD" ]; then
    FIELD_ID=$(cksum < "$BRILL_FIELD" | cut -d' ' -f1)
    for side in incumbent candidate; do
        conf=$INCUMBENT; out=$SP_ROOT/gate/$INC_ID/field-$FIELD_ID
        [ "$side" = candidate ] && conf=$G/candidate.conf && out=$G/field
        if [ ! -f "$out/complete" ]; then
            mkdir -p "$out"
            log "past events: $side bids $(wc -l < "$BRILL_FIELD") boards"
            python "$BEN/scripts/match/brill/brill_replay.py" --conf "$conf" --field "$BRILL_FIELD" --out "$out" \
                --robots "$ROBOTS" --workers "$SEARCH_WORKERS" --resume >> "$out/replay.log" 2>&1 \
                || fail "past events, $side, see $out/replay.log"
            touch "$out/complete"
            sweep
        fi
    done
    mkdir -p "$G/field-incumbent" && cp "$SP_ROOT/gate/$INC_ID/field-$FIELD_ID/replay.jsonl" "$G/field-incumbent/replay.jsonl"
fi

if [ "$V1" != better ]; then RESULT="rejected at gate 1 ($V1): keep the incumbent"
elif [ "$V2" = better ]; then RESULT="passed both gates: ready for the check against the real robots, then promote.sh $GEN"
elif [ "$V2" = worse ]; then RESULT="rejected at gate 2 (worse): keep the incumbent"
elif [ "$V2" = "not run" ]; then RESULT="passed gate 1, gate 2 not run"
else RESULT="passed gate 1, gate 2 shows no difference: more gate 2 deals, or keep the incumbent"; fi
{
    echo "generation $GEN: $RESULT"
    echo "candidate: $MODEL"
    echo "gate 1 (networks alone):  $(grep '^all deals' "$G/gate1/compare.txt")"
    [ -f "$G/gate2/compare.txt" ] && echo "gate 2 (whole settings):  $(grep '^all deals' "$G/gate2/compare.txt")"
    [ -s "$G/system_check.txt" ] && echo "system check: $(tr '\n' ' ' < "$G/system_check.txt" | cut -c1-300)"
} > "$G/verdict.txt"
log "$RESULT"
touch "$G/done"
