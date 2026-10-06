"""Equal wall-clock robustness check.

The main study equalises effort at 3,000 objective evaluations, which
reproduces exactly but is not equally expensive across methods: one
evaluation of the proposed method covers a whole colour support, with a
Hungarian solve per family inside it, where one evaluation of a
metaheuristic covers a single schedule. A reader is entitled to ask whether
the comparison survives when the budget is time instead.

This script answers that. Every method on an instance gets the same wall
clock, and the budget per scale is the mean CPU time the proposed method
actually used in the main study - so the competitors receive what it spent,
not less. Runs under a clock do not reproduce exactly; that is inherent to
the question and the reason this check is reported separately from the main
results rather than replacing them.

Usage:
    python walltime.py                    # defaults below
    python walltime.py --reps 3 --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

from benchmarks import Budget, alns, lpt_edd, nsga2
from core import evaluate_schedule, z2_lower_bound
from cpm import cpm_report
from instances import build_matrix, generate_instance
from runner import best_known, dominated_fraction, hypervolume, igd

# Seconds per scale: the mean the proposed method used in the main study,
# rounded up. Giving the competitors exactly that removes the obvious
# objection - that they were stopped early - without inventing a ladder of
# round numbers that favours nobody in particular.
SECONDS = {"small": 3.0, "medium": 17.0, "large": 70.0, "xlarge": 33.0}


def seconds_from_main(results_csvs) -> dict:
    """Per-scale clock = mean CPU time of CPM in the main study, rounded up.

    Read from the main results rather than typed in, so the check cannot
    drift out of step with the study it is checking.
    """
    import math
    import pandas as pd
    if isinstance(results_csvs, str):
        results_csvs = [results_csvs]
    df = pd.concat([pd.read_csv(p) for p in results_csvs])
    s = df[df.method == "CPM"].groupby("scale").seconds.mean()
    return {k: float(math.ceil(v)) for k, v in s.items() if k != "verify"}

METHODS = ("CPM", "CPM-blind", "NSGA-II", "ALNS", "LPT+EDD")
FIELDS = ["instance", "scale", "n", "machines", "families", "tau",
          "replication", "excess", "method", "seconds_budget",
          "z1_min", "z2_min", "front_size", "hypervolume", "igd",
          "z2_optimal", "seconds", "evaluations", "feasible"]


def _verified(inst, front) -> Tuple[List[Tuple[float, float]], bool]:
    """Re-check every returned point through the common evaluator, exactly as
    the main runner does. A method that reports a schedule it cannot defend
    is a bug, not a result."""
    pts, ok = [], True
    for p in front:
        ev = evaluate_schedule(inst, p.batches, p.schedule, strict=False)
        if (not ev.feasible or abs(ev.z1 - p.z1) > 1e-6
                or abs(ev.z2 - p.z2) > 1e-6):
            ok = False
            continue                      # never report a point we cannot defend
        pts.append((ev.z1, ev.z2))
    return pts, ok


def run_one(inst, seconds: float, seed: int) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    for label, variant in (("CPM", "cpm"), ("CPM-blind", "blind")):
        t0 = time.time()
        # A huge evaluation cap with a tight clock turns the safety net into
        # the binding constraint, which is what an equal-time run needs.
        # The clock is split across the outer passes exactly as the
        # evaluation budget is in the main study. Handing the whole clock to
        # the first pass, as an earlier version did, starved the L1 feedback
        # loop: CPM and CPM-blind then returned identical fronts on 38 of 72
        # instances, and the check said nothing about redistribution.
        front, diag = cpm_report(inst, variant=variant,
                                 eval_budget=10 ** 9,
                                 time_limit=seconds,
                                 safety_seconds=seconds * 1.5)
        pts, ok = _verified(inst, front)
        out[label] = {"points": pts, "feasible": ok,
                      "seconds": round(time.time() - t0, 3),
                      "evaluations": diag["evaluations"]}

    for label, fn in (("NSGA-II", nsga2), ("ALNS", alns), ("LPT+EDD", lpt_edd)):
        b = Budget(10 ** 9, seconds=seconds)
        t0 = time.time()
        sols = fn(inst, b) if label == "LPT+EDD" else fn(inst, b, seed=seed)
        pts, ok = _verified(inst, sols)
        out[label] = {"points": pts, "feasible": ok,
                      "seconds": round(time.time() - t0, 3),
                      "evaluations": b.used}
    return out


def main(out_dir: str, scales: Sequence[str], taus: Sequence[float],
         reps: int, budgets: dict | None = None) -> str:
    budgets = budgets or SECONDS
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "walltime.csv")
    ck_path = os.path.join(out_dir, "walltime_checkpoint.json")
    done = set(json.load(open(ck_path))) if os.path.exists(ck_path) else set()

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()

    cells = [c for c in build_matrix() if c.scale in scales]
    total = len(cells) * len(taus) * reps
    k = 0

    for cell in cells:
        for tau in taus:
            for rep in range(reps):
                inst = generate_instance(cell, tau, rep)
                k += 1
                if inst.name in done:
                    continue
                seconds = budgets[cell.scale]
                res = run_one(inst, seconds, seed=rep)

                allp = [p for v in res.values() for p in v["points"]]
                if not allp:
                    continue
                ref = (max(p[0] for p in allp) * 1.1 + 1.0,
                       max(p[1] for p in allp) * 1.1 + 0.1)
                bk = best_known(allp)
                z2lb = z2_lower_bound(inst)

                rows = []
                for m in METHODS:
                    v = res[m]
                    pts = v["points"]
                    z2min = min(p[1] for p in pts)
                    rows.append({
                        "instance": inst.name, "scale": cell.scale,
                        "n": cell.n, "machines": cell.machines,
                        "families": cell.n_families, "tau": tau,
                        "replication": rep, "excess": cell.excess,
                        "method": m, "seconds_budget": seconds,
                        "z1_min": round(min(p[0] for p in pts), 4),
                        "z2_min": round(z2min, 4),
                        "front_size": len(pts),
                        "hypervolume": round(hypervolume(pts, ref), 4),
                        "igd": round(igd(pts, bk), 4),
                        "z2_optimal": abs(z2min - z2lb) < 1e-6,
                        "seconds": v["seconds"],
                        "evaluations": v["evaluations"],
                        "feasible": v["feasible"],
                    })
                with open(csv_path, "a", newline="") as fh:
                    csv.DictWriter(fh, FIELDS).writerows(rows)
                done.add(inst.name)
                json.dump(sorted(done), open(ck_path, "w"))
                print(f"[{k}/{total}] {inst.name} @ {seconds}s")

    print("->", csv_path)
    return csv_path


def summarise(csv_path: str) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    order = ["small", "medium", "large", "xlarge"]
    print("\n=== equal wall clock ===")
    for sc in [s for s in order if s in set(df.scale)]:
        s = df[df.scale == sc]
        print(f"\n--- {sc}  ({s.seconds_budget.iloc[0]} s each) ---")
        t = s.groupby("method")[["hypervolume", "igd", "z1_min",
                                 "z2_min", "evaluations"]].mean().round(2)
        t["z2_opt_%"] = (100 * s.groupby("method").z2_optimal.mean()).round(1)
        print(t.reindex(list(METHODS)).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--scales", nargs="+",
                    default=["small", "medium", "large", "xlarge"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--main-results", nargs="+", default=None,
                    help="results.csv file(s) of the main study; set the clocks")
    a = ap.parse_args()
    b = seconds_from_main(a.main_results) if a.main_results else None
    if b:
        print("clock per scale:", b)
    summarise(main(a.out, a.scales, a.taus, a.reps, budgets=b))
