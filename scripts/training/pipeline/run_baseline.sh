#!/usr/bin/env bash
# Train and evaluate the two bidding networks from a boards PBN with a held-out test set:
# conversion to binary, bidding and bidding-info trainers, evaluation. See README.md here.
#
#   bash run_baseline.sh <run name>          (long runs: nohup setsid bash run_baseline.sh <name> &)
#
# The run folder is $TRAIN_ROOT/<run name> and must hold train.pbn, test.pbn and sample.pbn
# (prep_split.py makes them). Every step is skipped when its result exists, so the script can
# be started again after a stop (the trainers continue from their last epoch). It ends by
# writing "done" or "failed" in the run folder; run.log has the step times.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
T=${TRAIN_ROOT:-/var/lib/ben/training}
NAME=${1:?usage: run_baseline.sh <run name>}
R=$T/$NAME
BID=$HERE/../bidding
INFO=$HERE/../bidding_info
EPOCHS=${EPOCHS:-30}
# Arguments of the bidding trainer (batch 512 and learning rate 0.001 are its defaults).
# Since 2026-11-17: learning rate times 0.96 every 12000 batches (was 3000) and dropout 0.05
# (keep=0.95, was 0.2), which scored 95.6% same call / 66.1% whole auction on the held-out
# boards against 94.9% / 61.5%. Set BID_ARGS to try another recipe, for example the old one:
#   BID_ARGS="lstm_size=256 decay_steps=3000" bash run_baseline.sh <name>
BID_ARGS=${BID_ARGS:-"lstm_size=256 decay_steps=12000 keep=0.95"}
# A shipped model trained on the same boards, evaluated next to the new one
SHIPPED=${SHIPPED:-$HERE/../../../models/TF2models/GIB-BBO-8730_2025-04-19-E30.keras}

log() { echo "$(date "+%F %T") $*" | tee -a "$R/run.log"; }
fail() { log "FAILED: $*"; touch "$R/failed"; exit 1; }
rm -f "$R/done" "$R/failed"

# 1. Binary training data (x.npy is about 12 GB)
if [ ! -f "$R/bin/complete" ]; then
    log "Converting train.pbn to binary"
    (cd $BID && python bidding_binary_keras.py "$R/train.pbn" None "$R/bin" NS=1 EW=1 rotate=True n_cards=24 \
        max_occurrences=50 > "$R/auctions.txt" 2> "$R/binary.log") || fail "binary step, see binary.log"
    touch "$R/bin/complete"
    log "$(grep created "$R/binary.log" | tail -1)"
fi

# 2. Both networks, each in its own folder (they both write to ./model). Side by side when the
#    container has the memory for both, else one after the other. Together they peaked at
#    41 GB when they still held x.npy in memory, and under a 24 GB container limit they
#    swapped and ran many times slower.
mkdir -p "$R/bidding" "$R/info"
train_bidding() {
    (cd "$R/bidding" && python $BID/bidding_nn_keras.py "$R/bin" "$NAME" $BID_ARGS         epochs="$EPOCHS" >> "$R/train_bidding.log" 2>&1)
}
train_info() {
    (cd "$R/info" && python $INFO/binfo_nn_keras.py "$R/bin" "$NAME-Info" epochs="$EPOCHS"         >> "$R/train_info.log" 2>&1)
}
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [ "$LIMIT" = max ] || [ "$LIMIT" -ge $((48 * 1024 * 1024 * 1024)) ]; then
    log "Training $NAME ($BID_ARGS) and $NAME-Info side by side, $EPOCHS epochs"
    train_bidding &
    P1=$!
    train_info &
    P2=$!
    wait $P1 || fail "bidding network, see train_bidding.log"
    wait $P2 || fail "bidding-info network, see train_info.log"
else
    log "Training $NAME ($BID_ARGS), then $NAME-Info, $EPOCHS epochs (memory limit too low for both at once)"
    train_bidding || fail "bidding network, see train_bidding.log"
    train_info || fail "bidding-info network, see train_info.log"
fi
log "Training finished"

# 3. New model and the shipped model trained on the same boards, on held-out and on trained
#    boards. One pass at a time: four at once on the one GPU took 3 min 39 s against 39 s each
#    alone (2026-11-17). The shipped model gives the same result for the same boards every
#    run, so its passes are kept in eval_cache (delete that folder to evaluate it again).
while [ ! -f "$R/sample.pbn" ]; do sleep 30; done
MODEL=$(ls -t "$R/bidding/model/${NAME}_"*.keras | head -1)
mkdir -p "$T/eval_cache"
: > "$R/eval.txt"
for m in "$MODEL" "$SHIPPED"; do
    for set in test sample; do
        echo "=== $(basename "$m") on $set.pbn" >> "$R/eval.txt"
        CACHED="$T/eval_cache/$(basename "$m").$(cksum < "$R/$set.pbn" | tr ' ' '-').txt"
        if [ "$m" = "$SHIPPED" ] && [ -s "$CACHED" ]; then
            log "Evaluation of $(basename "$m") on $set.pbn taken from eval_cache"
            cat "$CACHED" >> "$R/eval.txt"
            continue
        fi
        log "Evaluating $(basename "$m") on $set.pbn"
        if (cd $BID && python eval_bidding_nn.py "$m" "$R/$set.pbn" --ns 1 --ew 1 --n_cards 24 2>> "$R/eval.err" \
                | grep -v "^$") > "$R/eval.part"; then
            cat "$R/eval.part" >> "$R/eval.txt"
            [ "$m" = "$SHIPPED" ] && cp "$R/eval.part" "$CACHED"
        else
            echo "   evaluation failed, see eval.err" >> "$R/eval.txt"
        fi
        rm -f "$R/eval.part"
    done
done
log "Done. Bidding model: $MODEL"
touch "$R/done"
