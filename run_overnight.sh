#!/usr/bin/env bash
# Calibration -> freeze -> validation, in one unattended run.
# Every choice below follows ANALYSIS_PLAN.md, committed before this run.
#
#   bash run_overnight.sh                    (12 workers, factor 0.47)
#   W=12 F=0.47 bash run_overnight.sh
#
# Rerunning the script after an interruption resumes where it stopped:
# fair_compare.py skips finished runs and tune.py recomputes its choice from
# the same files. A line cut off by a power cut is set aside on resume, and
# runs interrupted by sleep (wall time > 1.25 x clock + 2 s) are run again;
# each fair_compare.py call is therefore made twice, the second normally with
# nothing to do (ANALYSIS_PLAN.md, amendment 1).
set -euo pipefail

fc() { $PY fair_compare.py "$@"; $PY fair_compare.py "$@"; }

W=${W:-12}
F=${F:-0.47}
PY=${PY:-python}
SCALES=${SCALES:-"verify small medium large xlarge"}
VREPS=${VREPS:-4,5,6,7,8,9}
EXTRA=${EXTRA:-}

A=$(printf 'ALNS+R#%d,' 0 1 2 3 4 5 6 7 8 9); A=${A%,}
N=$(printf 'NSGA-II+R#%d,' 0 1 2 3 4 5 6 7 8 9); N=${N%,}
H="CPM+ALNS+R:union@0.02,CPM+ALNS+R:union@0.05,CPM+ALNS+R:union@0.1,CPM+ALNS+R:union@0.2,CPM+ALNS+R:anchor@0.05"

echo "== 0. regression checks"
$PY regress.py

echo "== 1. calibration of ALNS+R and NSGA-II+R (replications 2, 3)"
fc --scales $SCALES --reps 2,3 $EXTRA --workers $W --factor $F \
    --out results_calib --methods "$A,$N"
$PY tune.py results_calib --prefix 'ALNS+R#' --write frozen.json --key ALNS+R
$PY tune.py results_calib --prefix 'NSGA-II+R#' --write frozen.json --key NSGA-II+R

echo "== 2. calibration of the hybrid with the frozen ALNS+R parameters"
fc --scales $SCALES --reps 2,3 $EXTRA --workers $W --factor $F \
    --params frozen.json --out results_calib --methods "$H"
$PY tune.py results_calib --variants "$H" --require-z2 --write frozen.json --key hybrid
HY=$($PY -c "import json; print(json.load(open('frozen.json'))['hybrid'])")
echo "frozen configuration:"; cat frozen.json; echo

echo "== 3. validation (replications $VREPS), run once"
fc --scales $SCALES --reps $VREPS $EXTRA --workers $W --factor $F \
    --params frozen.json --out results_valid \
    --methods "$HY,ALNS+R,NSGA-II+R,CPM,ALNS,NSGA-II"

echo "== 4. verification scale, replications 0-3, for the CP-SAT gap"
fc --scales verify --reps 0,1,2,3 $EXTRA --workers $W --factor $F \
    --params frozen.json --out results_cpsatgap --methods "$HY,ALNS+R,CPM"

echo "== pipeline finished"
