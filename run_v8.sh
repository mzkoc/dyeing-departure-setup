#!/usr/bin/env bash
# Tonight's runs (ANALYSIS_PLAN.md, section 9), after the validation run:
#   1. E1c: the method of [7] with R; ablations CPM+R and CPM-blind     (~1.2 h)
#      on the validation set
#   2. budget robustness: H, ALNS+R, NSGA-II+R, CPM at 0.25, 0.5, 2 x    (~2.4 h)
#      the clock, replications 4-6
# Resumable; each step is called twice so that interrupted runs are redone.
#
#   bash run_v8.sh
set -euo pipefail
W=${W:-12}
F=${F:-0.47}
PY=${PY:-python}
SCALES=${SCALES:-"verify small medium large xlarge"}

fc() { $PY fair_compare.py "$@"; $PY fair_compare.py "$@"; }

HY=$($PY -c "import json; print(json.load(open('frozen.json'))['hybrid'])")

echo "== 1. [7]+R and the ablations on the validation set"
fc --scales $SCALES --reps 4,5,6,7,8,9 --workers $W --factor $F \
    --params frozen.json --out results_valid_extra \
    --methods "FFD+SA-EDDTS+R,CPM+R,CPM-blind"

echo "== 2. budget robustness"
for M in 0.25 0.5 2; do
  fc --scales $SCALES --reps 4,5,6 --workers $W --factor $F --clock-mult $M \
      --params frozen.json --out results_budget \
      --methods "$HY,ALNS+R,NSGA-II+R,CPM"
done
echo "== v8 runs finished"
