#!/usr/bin/env bash
# SA-EDDTS of the companion study at the budget of its own campaign
# (check of 16 Nov 2026). Run from the folder that holds the code and the
# result folders results_valid, results_comp7 and results_valid_extra:
#
#   W=12 bash run_own_budget.sh
#
# 30 jobs: 80 s per instance    (equal clock 3.29 s  x 24.316)
# 60 jobs: 133.3 s per instance (equal clock 63.92 s x 2.0859; its campaign
#          gave 120 s at 50 jobs and 160 s at 80 jobs)
# Validation instances, replications 4-9: 54 per scale, one run each.
# About 6 + 10 minutes with 12 workers. An interrupted run resumes where it
# stopped when the same command is given again.
set -euo pipefail
W=${W:-12}
PY=${PY:-python}
OUT=results_comp7own

for d in results_valid results_comp7 results_valid_extra comp7; do
  [ -d "$d" ] || { echo "ERROR: folder $d not found here"; exit 1; }
done

$PY fair_compare.py --scales small  --reps 4,5,6,7,8,9 --workers "$W" \
    --factor 0.47 --clock-mult 24.316 --out "$OUT" --methods FFD+SA-EDDTS
$PY fair_compare.py --scales medium --reps 4,5,6,7,8,9 --workers "$W" \
    --factor 0.47 --clock-mult 2.0859 --out "$OUT" --methods FFD+SA-EDDTS

$PY own_budget.py results_valid results_comp7 results_valid_extra "$OUT" \
    | tee own_budget_report.txt
echo
echo "Send back: own_budget_report.txt, own_budget_instances.csv and $OUT/results.jsonl"
