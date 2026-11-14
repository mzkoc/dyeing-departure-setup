#!/usr/bin/env bash
# Exploratory experiment E1 (ANALYSIS_PLAN.md, section 8): the method of [7]
# on the validation instances, same clock, after run_overnight.sh.
#
#   pip install pyomo==6.9.5      (once; the code of [7] imports it)
#   bash run_comp7.sh
set -euo pipefail
W=${W:-12}
F=${F:-0.47}
PY=${PY:-python}
SCALES=${SCALES:-"verify small medium large xlarge"}

fc() { $PY fair_compare.py "$@"; $PY fair_compare.py "$@"; }

fc --scales $SCALES --reps 4,5,6,7,8,9 --workers $W --factor $F \
    --out results_comp7 --methods "FFD+SA-EDDTS,FFD-EDDTS"

HY=$($PY -c "import json; print(json.load(open('frozen.json'))['hybrid'])")
M="$HY,CPM,ALNS+R,FFD+SA-EDDTS,FFD-EDDTS"
$PY analyze_fair.py results_comp7 results_valid --methods "$M" --tag comp7 \
    > analysis_comp7.txt
$PY confirm.py results_valid results_comp7 --curves-only --methods "$M" \
    >> analysis_comp7.txt
echo "written: analysis_comp7.txt"
