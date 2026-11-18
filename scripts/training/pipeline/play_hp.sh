#!/usr/bin/env bash
# Recipe runs of the card-play networks. Trains one model per line of a queue file and scores
# it with eval_play_nn.py on the deals the trainer holds out (the last 5% of the binary files).
#
#   bash play_hp.sh <binary dir> <out dir> <queue file>
#
# Queue lines:  name | hand | trainer arguments     (# starts a comment; a finished name is skipped)
#   hand: decl_nt, decl_suit, dummy_nt, dummy_suit, lefty_nt, lefty_suit, righty_nt, righty_suit
# One line per run is appended to <out dir>/results.txt (acc, acc_held, nll as printed by
# eval_play_nn.py). The queue file is read again after every run, so lines can be added while
# it is running. While <out dir>/pause exists no new run is started.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
PLAY=$HERE/../playing
BIN=${1:?binary dir}; OUT=${2:?out dir}; QUEUE=${3:?queue file}
mkdir -p "$OUT"
rm -f "$OUT/finished"
while :; do
    while [ -f "$OUT/pause" ]; do sleep 10; done
    NEXT=""
    while IFS='|' read -r name hand args; do
        name=$(echo "$name" | xargs)
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        [ -f "$OUT/$name/done" ] && continue
        NEXT=$name; HAND=$(echo "$hand" | xargs); ARGS=$(echo "$args" | xargs); break
    done < "$QUEUE"
    [ -z "$NEXT" ] && break
    R=$OUT/$NEXT
    mkdir -p "$R"
    T0=$(date +%s)
    (cd "$R" && python "$PLAY/play_nn_keras.py" "$BIN" "$HAND" name="$NEXT" $ARGS > train.log 2>&1)
    T1=$(date +%s)
    MODEL=$(ls "$R"/model/*-best.keras 2>/dev/null | head -1)
    EPOCHS=$(grep -c "Epoch took" "$R/train.log")
    if [ -n "$MODEL" ]; then
        (cd "$PLAY" && python eval_play_nn.py "$BIN" "$HAND" "$MODEL" 2> "$R/eval.err") > "$R/eval.txt"
        SUMMARY=$(grep -m1 "^$(basename "$MODEL")" "$R/eval.txt" | awk '{print "acc " $2 " acc_held " $3 " nll " $4}')
        # Keep the best model only
        ls "$R"/model/*.keras | grep -v -- "-best.keras" | xargs -r rm -f
    else
        SUMMARY="training failed, see $R/train.log"
    fi
    echo "$NEXT | $HAND | $ARGS | train $((T1 - T0)) s, $EPOCHS epochs | $SUMMARY" >> "$OUT/results.txt"
    touch "$R/done"
done
touch "$OUT/finished"
