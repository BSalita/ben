#!/usr/bin/env bash
# Make a generation's candidate the incumbent of the self-play loop: the next generation
# explores with it and is measured against it. The config before is kept beside the new one.
# This changes only the loop's own incumbent.conf, not any server.
#
#   bash promote.sh <generation name>
set -euo pipefail
GEN=${1:?usage: promote.sh <generation name>}
SP_ROOT=${SP_ROOT:-/var/lib/ben/selfplay}
INCUMBENT=${INCUMBENT:-$SP_ROOT/conf/incumbent.conf}
G=$SP_ROOT/gen/$GEN
[ -f "$G/candidate.conf" ] || { echo "no $G/candidate.conf"; exit 1; }
[ -f "$G/verdict.txt" ] && cat "$G/verdict.txt"
cp "$INCUMBENT" "$INCUMBENT.before-$GEN"
cp "$G/candidate.conf" "$INCUMBENT"
echo "$(date "+%F %T") $GEN promoted: $(grep '^bidder' "$INCUMBENT")" | tee -a "$SP_ROOT/history.log"
