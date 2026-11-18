#!/usr/bin/env bash
# Recipe runs of the opening-lead networks. Trains one model per line of a queue file with
# lead_train.py on the WBC and Jack data together and scores it with lead_eval.py on the rows
# lead_train.py holds out (the last 5% of each data folder).
#
#   bash lead_hp.sh <lead data dir> <WBC play.txt> <Jack dir> <out dir> <queue file>
#
# <lead data dir> holds wbc/{suit,nt} and jack/{suit,nt} from gen_lead_data.py, made from the
# WBC play.txt and from BW5C_N, BW5C_S, JOS_N, JOS_S (in that order) of the Jack dir.
# Queue lines:  name | suit or nt | arguments of lead_train.py     (# starts a comment)
# Two lines per run (WBC and Jack held-out rows) are appended to <out dir>/results.txt.
# While <out dir>/pause exists no new run is started.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
LEAD="$HERE/../opening lead"
DATA=${1:?lead data dir}; WBC=${2:?WBC play.txt}; J=${3:?Jack dir}; OUT=${4:?out dir}; QUEUE=${5:?queue file}
JACK="$J/BW5C_N.txt,$J/BW5C_S.txt,$J/JOS_N.txt,$J/JOS_S.txt"
mkdir -p "$OUT"
rm -f "$OUT/finished"
while :; do
    while [ -f "$OUT/pause" ]; do sleep 10; done
    NEXT=""
    while IFS='|' read -r name kind args; do
        name=$(echo "$name" | xargs)
        [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
        [ -f "$OUT/$name/done" ] && continue
        NEXT=$name; KIND=$(echo "$kind" | xargs); ARGS=$(echo "$args" | xargs); break
    done < "$QUEUE"
    [ -z "$NEXT" ] && break
    R=$OUT/$NEXT
    mkdir -p "$R"
    T0=$(date +%s)
    (cd "$R" && python "$LEAD/lead_train.py" "$NEXT" "$DATA/wbc/$KIND" "$DATA/jack/$KIND" $ARGS > train.log 2>&1)
    T1=$(date +%s)
    MODEL=$(ls -t "$R"/model/*.keras 2>/dev/null | head -1)
    EPOCHS=$(grep -c "^Epoch [0-9]*/" "$R/train.log")
    if [ -n "$MODEL" ]; then
        (cd "$LEAD" && python lead_eval.py "wbc-$KIND=$DATA/wbc/$KIND:$WBC" "jack-$KIND=$DATA/jack/$KIND:$JACK" -- "$MODEL" \
            2> "$R/eval.err") > "$R/eval.txt"
        grep "^$(basename "$MODEL" | cut -c1-42)" "$R/eval.txt" | while read -r line; do
            echo "$NEXT | $KIND | $ARGS | train $((T1 - T0)) s, $EPOCHS epochs | $(echo "$line" | awk '{print $2 ": acc " $3 " nll " $4 " cover " $5 " DDOLAR " $6 " ADDOLAR " $7}')" >> "$OUT/results.txt"
        done
    else
        echo "$NEXT | $KIND | $ARGS | training failed, see $R/train.log" >> "$OUT/results.txt"
    fi
    touch "$R/done"
done
touch "$OUT/finished"
