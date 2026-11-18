#!/usr/bin/env bash
# Recipe runs of the bidding-info network. Trains one model per line of a queue file on the
# binary data of an existing run and scores it on held-out boards with eval_binfo_nn.py.
#
#   bash info_hp.sh <train bin dir> <held-out bin dir> <out dir> <queue file>
#
# Queue lines:  name | trainer arguments      (# starts a comment; a finished name is skipped)
# The held-out bin dir comes from bidding_binary_keras.py on test.pbn with
# max_occurrences=1000000. One line per run is appended to <out dir>/results.txt; the queue
# file is read again after every run, so lines can be added while it is running.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
INFO=$HERE/../bidding_info
BIN=${1:?train bin dir}; TEST=${2:?held-out bin dir}; OUT=${3:?out dir}; QUEUE=${4:?queue file}
mkdir -p "$OUT"
rm -f "$OUT/finished"
while :; do
    NEXT=""
    while IFS='|' read -r name args; do
        name=$(echo "$name" | xargs)
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        [ -f "$OUT/$name/done" ] && continue
        NEXT=$name; ARGS=$(echo "$args" | xargs); break
    done < "$QUEUE"
    [ -z "$NEXT" ] && break
    R=$OUT/$NEXT
    mkdir -p "$R"
    T0=$(date +%s)
    (cd "$R" && python "$INFO/binfo_nn_keras.py" "$BIN" "$NEXT" $ARGS > train.log 2>&1)
    T1=$(date +%s)
    MODEL=$(ls -t "$R"/model/*.keras 2>/dev/null | head -1)
    if [ -n "$MODEL" ]; then
        (cd "$INFO" && python eval_binfo_nn.py "$TEST" "$MODEL" rounds=1 2> "$R/eval.err" | grep -v "^$") > "$R/eval.txt"
        SUMMARY=$(grep "summary:" "$R/eval.txt" | tail -1 | sed 's/ *summary: //')
        # Keep the last model only
        ls "$R"/model/*.keras | grep -v "$(basename "$MODEL")" | xargs -r rm -f
    else
        SUMMARY="training failed, see $R/train.log"
    fi
    echo "$NEXT | $ARGS | train $((T1 - T0)) s | $(grep "loss:" "$R/train.log" | tail -1 | sed 's/.*step - //') | $SUMMARY" >> "$OUT/results.txt"
    touch "$R/done"
done
touch "$OUT/finished"
