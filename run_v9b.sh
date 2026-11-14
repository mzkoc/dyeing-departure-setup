#!/usr/bin/env bash
# v9b: experiment E2 (ANALYSIS_PLAN.md, section 9, amendments 3 and 5).
# 108 instances (60 and 150 jobs, replications 4-9) x lambda 1 and 2 x
# H-max, H-mix, ALNS+R, [7]+R, plus H-mean at lambda 1: 972 runs, about
# 2.2 h on 12 workers. Resumable; called twice so interrupted runs are redone.
#
#   bash run_v9b.sh
set -euo pipefail
W=${W:-12}
F=${F:-0.47}
PY=${PY:-python}
for i in 1 2; do
  $PY e2.py --scales medium xlarge --workers $W --factor $F \
      --params frozen.json --out results_e2
done
$PY e2_analyze.py results_e2 --by-regime | tee results_e2/analysis_e2.txt
echo "== v9b finished"
