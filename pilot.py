"""Equal wall-clock pilot: CPM, CPM-blind, ALNS and ALNS+R.

Same instance subset and seeds as walltime.py (every cell of a scale, three
tightness levels, replications 0 and 1, seed = replication), same clock rule
(the mean CPU time of CPM in the main study at that scale), scaled by the
measured speed ratio of this machine to the one the main study ran on.

Each worker runs every method on its share of the instances, one after the
other, so every method sees the same concurrency. Every returned point is
re-checked through core.evaluate_schedule before it is written.

    python pilot.py --scale medium --worker 0 --of 2 --factor 0.37 --out pilot
"""

from __future__ import annotations

import argparse
import json
import os
import time

from alns_r import alns_r
from benchmarks import Budget, alns
from core import evaluate_schedule, z2_lower_bound
from cpm import cpm_report
from instances import build_matrix, generate_instance

MAIN_CLOCK = {"small": 7.0, "medium": 136.0, "large": 302.0, "xlarge": 281.0}
TAUS = (0.3, 0.5, 0.7)
REPS = (0, 1)
METHODS = ("CPM", "CPM-blind", "ALNS", "ALNS+R")


def verified(inst, sols):
    pts, ok = [], True
    for s in sols:
        ev = evaluate_schedule(inst, s.batches, s.schedule, strict=False)
        if (not ev.feasible or abs(ev.z1 - s.z1) > 1e-6
                or abs(ev.z2 - s.z2) > 1e-6):
            ok = False
            continue
        pts.append([ev.z1, ev.z2])
    return pts, ok


def run_method(inst, method, seconds, seed):
    t0 = time.time()
    if method == "CPM+R":
        from cpm_r import cpm_r
        front, evals = cpm_r(inst, seconds)
    elif method in ("CPM", "CPM-blind"):
        front, diag = cpm_report(inst,
                                 variant="cpm" if method == "CPM" else "blind",
                                 eval_budget=10 ** 9, time_limit=seconds,
                                 safety_seconds=seconds * 1.5)
        evals = diag["evaluations"]
    else:
        b = Budget(10 ** 9, seconds=seconds)
        fn = alns if method == "ALNS" else alns_r
        front = fn(inst, b, seed=seed)
        evals = b.used
    secs = time.time() - t0
    pts, ok = verified(inst, front)
    return pts, ok, secs, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--out", default="pilot")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    methods = tuple(a.methods.split(","))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.scale}_w{a.worker}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["instance"], r["method"]))

    clock = round(MAIN_CLOCK[a.scale] * a.factor, 2)
    insts = []
    for cell in [c for c in build_matrix() if c.scale == a.scale]:
        for tau in TAUS:
            for rep in REPS:
                insts.append((cell, tau, rep))
    for k, (cell, tau, rep) in enumerate(insts):
        if k % a.of != a.worker:
            continue
        inst = generate_instance(cell, tau, rep)
        # rotate the method order so no method always runs first or last
        r = k % len(methods)
        order = methods[r:] + methods[:r]
        for method in order:
            if (inst.name, method) in done:
                continue
            pts, ok, secs, evals = run_method(inst, method, clock, rep)
            rec = {"instance": inst.name, "scale": a.scale,
                   "machines": inst.machines, "families": len(inst.proc),
                   "tau": tau, "replication": rep, "excess": inst.excess,
                   "method": method, "clock": clock,
                   "seconds": round(secs, 3), "evaluations": evals,
                   "feasible": ok, "z2_bound": z2_lower_bound(inst),
                   "points": pts}
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(inst.name, method, round(secs, 1),
                  min((p[0] for p in pts), default=None), flush=True)


if __name__ == "__main__":
    main()
