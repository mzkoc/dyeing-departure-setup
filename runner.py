"""
Experiment runner: all methods per instance, Drive-friendly, checkpointed.

One row per (instance, method). Every method's front is re-checked through
`core.evaluate_schedule` before anything is written, so a row can only exist
if its schedules are feasible and its objectives agree with the evaluator.

Checkpointing is per INSTANCE, not per method, because the multi-objective
indicators are relative: the best-known front an instance's IGD and C-metric
are measured against is assembled from all the methods that ran on it. Half an
instance is not a usable unit.

Colab disconnects. Finished instances append to the CSV immediately and are
recorded in a checkpoint file; re-running the same command resumes. Delete the
checkpoint to force a clean run.

Typical use in a Colab cell:

    from runner import run_experiments
    run_experiments(
        out_dir="/content/drive/MyDrive/dyeing/results",
        scales=("medium",),
        reps=10,
        eval_budget=10000,
    )
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmarks import Budget, alns, lpt_edd, nsga2
from bounds import gap_to_bound, z1_lower_bound
from core import evaluate_schedule, z2_lower_bound
from cpm import cpm_report
from instances import (build_matrix, cleaning_water_litres,
                       generate_instance)

RESULTS_CSV = "results.csv"
FRONTS_JSONL = "fronts.jsonl"
CHECKPOINT = "checkpoint.json"

FIELDS = [
    "instance", "scale", "n", "machines", "families", "customers",
    "tau", "replication", "excess", "method",
    "z1_min", "z2_min", "z1_at_z2min", "front_size",
    "hypervolume", "igd", "c_dominated_by_best",
    "z2_lower_bound", "z2_optimal", "z1_lower_bound", "z1_gap_pct",
    "cleaning_water_m3",
    "seconds", "evaluations", "timed_out", "feasible", "seed",
]

# CPM plus its L1 ablation, then the benchmarks.
METHODS = ("CPM", "CPM-blind", "ALNS", "NSGA-II", "LPT+EDD")


# --------------------------------------------------------------------------
# Multi-objective indicators
# --------------------------------------------------------------------------

SCALE_Z2 = 100.0     # Z2 is hours of setup, Z1 is job-hours; put them on
                     # comparable footing before any distance is taken


def hypervolume(points: Sequence[Tuple[float, float]],
                ref: Tuple[float, float]) -> float:
    h, prev = 0.0, ref[1]
    for z1, z2 in sorted(points, key=lambda t: t[0]):
        if z1 >= ref[0] or z2 >= prev:
            continue
        h += (ref[0] - z1) * (prev - z2)
        prev = z2
    return h


def igd(front: Sequence[Tuple[float, float]],
        reference: Sequence[Tuple[float, float]]) -> float:
    if not front or not reference:
        return float("nan")
    total = 0.0
    for r in reference:
        total += min(((r[0] - p[0]) ** 2
                      + (SCALE_Z2 * (r[1] - p[1])) ** 2) ** 0.5
                     for p in front)
    return total / len(reference)


def dominated_fraction(front: Sequence[Tuple[float, float]],
                       by: Sequence[Tuple[float, float]]) -> float:
    """Fraction of `front` dominated by some point of `by` (C-metric)."""
    if not front:
        return float("nan")
    n = 0
    for q in front:
        if any(p[0] <= q[0] + 1e-9 and p[1] <= q[1] + 1e-9
               and (p[0] < q[0] - 1e-9 or p[1] < q[1] - 1e-9) for p in by):
            n += 1
    return n / len(front)


def best_known(points: Sequence[Tuple[float, float]]
               ) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        if any(q[0] <= p[0] + 1e-9 and q[1] <= p[1] + 1e-9
               and (q[0] < p[0] - 1e-9 or q[1] < p[1] - 1e-9)
               for q in points):
            continue
        if any(abs(q[0] - p[0]) < 1e-9 and abs(q[1] - p[1]) < 1e-9
               for q in out):
            continue
        out.append(p)
    return sorted(out, key=lambda t: (t[1], t[0]))


# --------------------------------------------------------------------------
# Running one instance
# --------------------------------------------------------------------------


def _verified(inst, front) -> Tuple[List[Tuple[float, float]], bool]:
    """Re-check a benchmark front through the evaluator. Nothing is trusted."""
    pts, ok = [], True
    for s in front:
        ev = evaluate_schedule(inst, s.batches, s.schedule, strict=False)
        if (not ev.feasible or abs(ev.z1 - s.z1) > 1e-6
                or abs(ev.z2 - s.z2) > 1e-6):
            ok = False
            continue
        pts.append((ev.z1, ev.z2))
    return pts, ok


def run_one(inst, eval_budget: int, seed: int = 1) -> Dict[str, dict]:
    """Run every method on one instance and return their verified fronts.

    `seed` is the replication index, so a cell's ten replications sample ten
    random streams as well as ten instances. Holding it fixed - as an earlier
    version did - would have run the stochastic methods from the same stream
    every time, leaving their variance unmeasured and inviting the obvious
    objection that each was run once.
    """
    out: Dict[str, dict] = {}

    for label, variant in (("CPM", "cpm"), ("CPM-blind", "blind")):
        t0 = time.time()
        front, diag = cpm_report(inst, variant=variant,
                                 eval_budget=eval_budget)
        pts, ok = [], True
        for p in front:
            ev = evaluate_schedule(inst, p.batches, p.schedule, strict=False)
            if (not ev.feasible or abs(ev.z1 - p.z1) > 1e-6
                    or abs(ev.z2 - p.z2) > 1e-6):
                ok = False
                continue
            pts.append((ev.z1, ev.z2))
        out[label] = {"points": pts, "seconds": diag["seconds"],
                      "evaluations": diag["evaluations"],
                      "timed_out": diag["timed_out"], "feasible": ok}

    for label, fn in (("ALNS", alns), ("NSGA-II", nsga2),
                      ("LPT+EDD", lpt_edd)):
        budget = Budget(eval_budget)
        t0 = time.time()
        front = fn(inst, budget) if label == "LPT+EDD" \
            else fn(inst, budget, seed=seed)
        elapsed = time.time() - t0
        pts, ok = _verified(inst, front)
        out[label] = {"points": pts, "seconds": round(elapsed, 3),
                      "evaluations": budget.used, "timed_out": False,
                      "feasible": ok}
    return out


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def _load_checkpoint(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return set(json.load(fh))


def _save_checkpoint(path: str, done: Iterable[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(sorted(done), fh)
    os.replace(tmp, path)


def _append_rows(path: str, rows: Sequence[dict]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------


def run_experiments(out_dir: str, *,
                    scales: Sequence[str] = ("small",),
                    taus: Sequence[float] = (0.3, 0.5, 0.7),
                    reps: int = 10,
                    methods: Sequence[str] = METHODS,
                    eval_budget: int = 10000,
                    verbose: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, RESULTS_CSV)
    jsonl_path = os.path.join(out_dir, FRONTS_JSONL)
    ckpt_path = os.path.join(out_dir, CHECKPOINT)

    done = _load_checkpoint(ckpt_path)
    cells = [c for c in build_matrix(include_verify=True)
             if c.scale in scales]
    total = len(cells) * len(taus) * reps
    seen = 0

    for cell in cells:
        for tau in taus:
            for rep in range(reps):
                seen += 1
                inst = generate_instance(cell, tau, rep)
                if inst.name in done:
                    continue

                res = run_one(inst, eval_budget, seed=rep)
                res = {k: v for k, v in res.items() if k in methods}
                allp = [p for v in res.values() for p in v["points"]]
                if not allp:
                    if verbose:
                        print(f"[{seen}/{total}] {inst.name}: no fronts")
                    continue

                ref = (max(p[0] for p in allp) * 1.1 + 1.0,
                       max(p[1] for p in allp) * 1.1 + 0.1)
                bk = best_known(allp)
                z2lb = z2_lower_bound(inst)
                z1lb = z1_lower_bound(inst)

                rows = []
                for method, v in res.items():
                    pts = v["points"]
                    if not pts:
                        continue
                    z2min = min(p[1] for p in pts)
                    z1min = min(p[0] for p in pts)
                    rows.append({
                        "instance": inst.name, "scale": cell.scale,
                        "n": cell.n, "machines": cell.machines,
                        "families": cell.n_families,
                        "customers": cell.n_customers,
                        "tau": tau, "replication": rep,
                        "excess": cell.excess, "method": method,
                        "z1_min": round(z1min, 4),
                        "z2_min": round(z2min, 4),
                        "z1_at_z2min": round(
                            min(p[0] for p in pts
                                if abs(p[1] - z2min) < 1e-9), 4),
                        "front_size": len(pts),
                        "hypervolume": round(hypervolume(pts, ref), 4),
                        "igd": round(igd(pts, bk), 4),
                        "c_dominated_by_best": round(
                            dominated_fraction(pts, bk), 4),
                        "z2_lower_bound": round(z2lb, 4),
                        "z2_optimal": abs(z2min - z2lb) < 1e-6,
                        "cleaning_water_m3": round(
                            cleaning_water_litres(z2min) / 1000.0, 3),
                        "z1_lower_bound": round(z1lb, 4),
                        "z1_gap_pct": gap_to_bound(z1min, z1lb),
                        "seconds": v["seconds"],
                        "evaluations": v["evaluations"],
                        "timed_out": v["timed_out"],
                        "feasible": v["feasible"],
                        "seed": rep,
                    })

                _append_rows(csv_path, rows)
                with open(jsonl_path, "a") as fh:
                    for method, v in res.items():
                        fh.write(json.dumps({
                            "instance": inst.name, "method": method,
                            "front": [[round(a, 4), round(b, 4)]
                                      for a, b in v["points"]],
                        }) + "\n")

                done.add(inst.name)
                _save_checkpoint(ckpt_path, done)

                if verbose:
                    best = min(rows, key=lambda r: r["igd"])["method"]
                    bad = [r["method"] for r in rows if not r["feasible"]]
                    flag = "" if not bad else "  !! dogrulama: " + ",".join(bad)
                    print(f"[{seen}/{total}] {inst.name}: "
                          f"en iyi IGD={best}{flag}")

    if verbose:
        print(f"\nresults -> {csv_path}")
    return csv_path


# --------------------------------------------------------------------------
# LaTeX output for Overleaf
# --------------------------------------------------------------------------


def latex_table(csv_path: str, out_path: str, *,
                group_by: Sequence[str] = ("scale", "method"),
                caption: str = "Method comparison.",
                label: str = "tab:comparison") -> str:
    """Aggregate the results CSV into a booktabs table ready for Overleaf."""
    rows: List[dict] = []
    with open(csv_path) as fh:
        rows.extend(csv.DictReader(fh))

    buckets: Dict[tuple, List[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in group_by), []).append(r)

    def mean(rs, key, cast=float):
        vals = [cast(r[key]) for r in rs if r[key] not in ("", "nan")]
        return statistics.mean(vals) if vals else float("nan")

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(group_by) + "rrrrrr}",
        r"\toprule",
        " & ".join(list(group_by) + [
            "HV", "IGD", r"$Z_1^{\min}$", r"$|P|$",
            r"$Z_2$ opt.\ (\%)", "CPU (s)"]) + r" \\",
        r"\midrule",
    ]
    for key in sorted(buckets):
        rs = buckets[key]
        opt = 100.0 * sum(r["z2_optimal"] == "True" for r in rs) / len(rs)
        lines.append(" & ".join(list(key) + [
            f"{mean(rs,'hypervolume'):.1f}", f"{mean(rs,'igd'):.2f}",
            f"{mean(rs,'z1_min'):.1f}", f"{mean(rs,'front_size'):.1f}",
            f"{opt:.0f}", f"{mean(rs,'seconds'):.2f}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    text = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
#
# Scales can be run as separate processes, each into its own directory, and
# merged by analysis.py afterwards. Separate directories matter: several
# processes appending to one CSV can interleave rows.
#
#   python runner.py --scales verify small --out results/part_a
#   python runner.py --scales medium       --out results/part_b
#   python runner.py --scales large        --out results/part_c
#   python runner.py --scales xlarge       --out results/part_d

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--scales", nargs="+",
                    default=["verify", "small", "medium", "large", "xlarge"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--budget", type=int, default=10000)
    a = ap.parse_args()
    run_experiments(a.out, scales=a.scales, taus=a.taus, reps=a.reps,
                    eval_budget=a.budget)
