"""
CP-SAT reference run on the verification scale.

Separate from `runner.py` on purpose. CP-SAT is only meaningful at n <= 20 and
is orders of magnitude slower than everything else, so folding it into the
main sweep would make that sweep unusable.

WHAT THIS DOES AND DOES NOT ESTABLISH. CP-SAT proves optimality here only
while the optimum is ZERO; once tardiness is positive it returns FEASIBLE with
a bound near zero even at n = 12 after 80 seconds. So a row's `cpsat_status`
decides how its numbers may be read:

    OPTIMAL   CPM's distance is a genuine optimality gap.
    FEASIBLE  CPM's distance is to a best-known value, nothing more. It can
              be negative - CPM beat CP-SAT - and at n = 30 it routinely is.

The script writes both and never collapses them into one "gap" column, because
reporting an unproven incumbent as an optimum is exactly the kind of claim a
reviewer catches.

Two things are cross-checked rather than assumed:

  * CP-SAT's independently minimised Z2 against P1's closed form. Two
    unrelated derivations agreeing is the strongest evidence either is right;
    a mismatch means the proof or the code is wrong.
  * Every CP-SAT schedule through `core.evaluate_schedule`, the same
    evaluator every other method answers to.

Colab use:

    from verify_run import run_verification
    run_verification(out_dir=OUT_DIR, det_time=30.0, reps=5)
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, Iterable, List, Sequence

import cpsat
from bounds import z1_lower_bound
from core import evaluate_schedule, z2_lower_bound
from cpm import cpm_report
from instances import build_matrix, generate_instance

RESULTS_CSV = "cpsat_verification.csv"
CHECKPOINT = "cpsat_checkpoint.json"

FIELDS = [
    "instance", "n", "machines", "families", "customers",
    "tau", "replication", "excess",
    "p1_z2_bound", "cpsat_z2_min", "cpsat_z2_status", "p1_confirmed",
    "cpsat_z1_min", "cpsat_z1_status", "cpsat_z1_optimal", "cpsat_seconds",
    "cpm_z1_min", "cpm_z2_min", "cpm_seconds", "cpm_evaluations",
    "cpm_distance_pct", "cpm_beat_cpsat",
    "z1_lower_bound", "cpsat_feasible", "cpm_feasible",
]


def _load(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return set(json.load(fh))


def _save(path: str, done: Iterable[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(sorted(done), fh)
    os.replace(tmp, path)


def _append(path: str, row: Dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------


def compare_one(inst, *, det_time: float = 30.0,
                eval_budget: int = 3000) -> Dict:
    """CPM first, so its solution can warm-start CP-SAT."""
    front, diag = cpm_report(inst, eval_budget=eval_budget)
    cpm_ok = True
    for p in front:
        ev = evaluate_schedule(inst, p.batches, p.schedule, strict=False)
        if not ev.feasible or abs(ev.z1 - p.z1) > 1e-6:
            cpm_ok = False
    anchor = min(front, key=lambda p: p.z1) if front else None

    p1_bound = z2_lower_bound(inst)

    # --- CP-SAT: min Z2, cross-checked against P1 ------------------------
    payoff = cpsat.lexicographic_payoff(inst, det_time=det_time,
                                        pool_slack=0)
    z2_min = payoff["z2_min"]
    p1_confirmed = (z2_min is not None
                    and abs(z2_min - p1_bound) < 1e-6)

    # --- CP-SAT: min Z1, warm-started from CPM ---------------------------
    from ortools.sat.python import cp_model
    mdl = cpsat.CpSatModel(inst, pool_slack=0)
    model = mdl.build(None)
    model.Minimize(mdl.vars["z1"])
    if anchor is not None:
        mdl.add_hint(model, anchor.batches, anchor.schedule)
    solver = cp_model.CpSolver()
    cpsat._configure(solver, det_time,
                     wall_clock_net=max(30.0, det_time * 20))
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    cps_z1 = None
    cps_ok = False
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        batches, schedule = mdl.extract(solver)
        ev = evaluate_schedule(inst, batches, schedule, strict=False)
        cps_z1, cps_ok = ev.z1, ev.feasible

    cpm_z1 = anchor.z1 if anchor else None
    dist = None
    beat = None
    if cps_z1 is not None and cpm_z1 is not None and cps_z1 > 1e-9:
        dist = round(100.0 * (cpm_z1 - cps_z1) / cps_z1, 2)
        beat = cpm_z1 < cps_z1 - 1e-9

    return {
        "instance": inst.name,
        "n": len(inst.jobs), "machines": inst.machines,
        "families": len(inst.proc),
        "customers": len(inst.customers),
        "tau": inst.meta.get("tau"),
        "replication": inst.meta.get("replication"),
        "excess": inst.excess,
        "p1_z2_bound": round(p1_bound, 4),
        "cpsat_z2_min": None if z2_min is None else round(z2_min, 4),
        "cpsat_z2_status": payoff["z2_status"],
        "p1_confirmed": p1_confirmed,
        "cpsat_z1_min": None if cps_z1 is None else round(cps_z1, 4),
        "cpsat_z1_status": solver.StatusName(status),
        "cpsat_z1_optimal": status == cp_model.OPTIMAL,
        "cpsat_seconds": round(elapsed, 2),
        "cpm_z1_min": None if cpm_z1 is None else round(cpm_z1, 4),
        "cpm_z2_min": None if not front else round(min(p.z2 for p in front), 4),
        "cpm_seconds": diag["seconds"],
        "cpm_evaluations": diag["evaluations"],
        "cpm_distance_pct": dist,
        "cpm_beat_cpsat": beat,
        "z1_lower_bound": round(z1_lower_bound(inst), 4),
        "cpsat_feasible": cps_ok,
        "cpm_feasible": cpm_ok,
    }


# --------------------------------------------------------------------------


def run_verification(out_dir: str, *,
                     taus: Sequence[float] = (0.3, 0.5, 0.7),
                     reps: int = 5,
                     det_time: float = 30.0,
                     eval_budget: int = 3000,
                     verbose: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, RESULTS_CSV)
    ckpt = os.path.join(out_dir, CHECKPOINT)
    done = _load(ckpt)

    cells = [c for c in build_matrix(include_verify=True)
             if c.scale == "verify"]
    total = len(cells) * len(taus) * reps
    seen = 0

    for cell in cells:
        for tau in taus:
            for rep in range(reps):
                seen += 1
                inst = generate_instance(cell, tau, rep)
                if inst.name in done:
                    continue
                row = compare_one(inst, det_time=det_time,
                                  eval_budget=eval_budget)
                _append(csv_path, row)
                done.add(inst.name)
                _save(ckpt, done)
                if verbose:
                    warn = "" if row["p1_confirmed"] else "  !! P1 UYUSMADI"
                    print(f"[{seen}/{total}] {inst.name}: "
                          f"CPM={row['cpm_z1_min']} "
                          f"CP-SAT={row['cpsat_z1_min']} "
                          f"({row['cpsat_z1_status']}) "
                          f"fark={row['cpm_distance_pct']}%{warn}")

    if verbose:
        print(f"\nresults -> {csv_path}")
    return csv_path


def summarise(csv_path: str) -> None:
    """Read the verification CSV the way it has to be read: split by whether
    CP-SAT actually proved anything."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"{len(df)} ornek")
    print(f"P1 dogrulandi: {100*df.p1_confirmed.mean():.0f}%  "
          f"(CP-SAT'in bagimsiz min Z2'si kapali formla ayni mi)")
    print(f"CPM fizibil: {100*df.cpm_feasible.mean():.0f}%  "
          f"CP-SAT fizibil: {100*df.cpsat_feasible.mean():.0f}%")
    print()
    proven = df[df.cpsat_z1_optimal]
    unproven = df[~df.cpsat_z1_optimal]
    print(f"--- CP-SAT optimali KANITLADI ({len(proven)} ornek) ---")
    if len(proven):
        print(f"  CPM optimallik bosslugu: ort {proven.cpm_distance_pct.mean():.2f}%  "
              f"medyan {proven.cpm_distance_pct.median():.2f}%  "
              f"max {proven.cpm_distance_pct.max():.2f}%")
        print(f"  CPM optimali buldu: {(proven.cpm_distance_pct.abs()<1e-9).sum()}/{len(proven)}")
    print(f"--- CP-SAT kanitlayamadi ({len(unproven)} ornek) ---")
    if len(unproven):
        print(f"  CPM'in en-iyi-bilinene uzakligi: ort "
              f"{unproven.cpm_distance_pct.mean():.2f}%")
        print(f"  CPM CP-SAT'i gecti: {unproven.cpm_beat_cpsat.sum()}/{len(unproven)}")
    print()
    print(f"Sureler: CPM ort {df.cpm_seconds.mean():.2f}s, "
          f"CP-SAT ort {df.cpsat_seconds.mean():.1f}s  "
          f"({df.cpsat_seconds.mean()/max(df.cpm_seconds.mean(),1e-9):.0f}x)")


if __name__ == "__main__":
    run_verification("./verification_results", reps=2, det_time=15.0)
