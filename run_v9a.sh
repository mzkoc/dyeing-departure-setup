#!/usr/bin/env bash
# v9a runs (ANALYSIS_PLAN.md, section 9 and amendment 4):
#   0. reproduction check of the adapter for [7]                     (~4 min)
#   1. sensitivity, one factor at a time: 8 variants x 36 instances  (~2.6 h)
#      x 4 methods, e > 0 cells at 60 and 150 jobs, reps 4-9
#   2. scalability: 200 and 300 jobs, m = 4/8/12, tau 0.5, reps 4-9  (~45 min)
#   3. CPM's hand-set parameters inside the hybrid, one at a time,   (~80 min)
#      calibration reps 2-3, 60 and 100 jobs
# Resumable; each step is called twice so that interrupted runs are redone.
#
#   bash run_v9a.sh
set -euo pipefail
W=${W:-12}
F=${F:-0.47}
PY=${PY:-python}

fc() { $PY fair_compare.py "$@"; $PY fair_compare.py "$@"; }

HY=$($PY -c "import json; print(json.load(open('frozen.json'))['hybrid'])")
M4="$HY,ALNS+R,NSGA-II+R,CPM"

echo "== 0. reproduction check of the adapter for [7]"
$PY check_comp7.py --reps 0,1 --workers 6 --out results_comp7repro

echo "== 1. sensitivity"
for V in perturb sigx2 sigx4 skew cluster c5 q300 size; do
  echo "-- variant $V"
  fc --scales medium xlarge --min-excess 1 --variant $V --reps 4,5,6,7,8,9 \
      --workers $W --factor $F --params frozen.json --out results_sens \
      --methods "$M4"
done

echo "== 2. scalability"
fc --scales x200 x300 --taus 0.5 --reps 4,5,6,7,8,9 --workers $W \
    --factor $F --params frozen.json --out results_scale --methods "$M4"

echo "== 3. CPM parameters inside the hybrid"
CP=$($PY -c "import cpm_params; print(','.join(cpm_params.variant_names('$HY')))")
fc --scales medium large --reps 2,3 --workers $W --factor $F \
    --params frozen.json --out results_cpmparams --methods "$CP"

echo "== v9a runs finished"
