"""Where does CPM lose tardiness? Layer diagnosis on development instances.

For each instance, CPM and ALNS+R are run on the same clock. Every point of
ALNS+R's front whose schedule is block-contiguous with the largest-sigma block
last (i.e. lies in CPM's own search space) gives a support; CPM's L2b-L4
(layer234.solve_support) are then run on that support

  (a) with ALNS+R's own batches,
  (b) with the batches of CPM's least-tardiness point,
  (c) with (b) adapted to the support: solve -> redistribute the batches
      against that schedule's completion times -> solve again, while Z1
      improves.

If (a) reproduces ALNS+R's Z1, L2b-L4 are not where the loss is; the gap
between (a) and (b) is the cost of batches not matched to the support.

    python diagnose_layers.py --scales medium large --reps 0,1 --factor 0.37 --workers 2 --out diag
    python diagnose_layers.py --summary diag        (Table 10 of the paper)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

MAIN_CLOCK = {"small": 7.0, "medium": 136.0, "large": 302.0, "xlarge": 281.0}


def adapt(inst, support, batches, rounds=6):
    from alns_r import _redistributed
    from core import evaluate_schedule
    from layer234 import solve_support
    best = None
    B = batches
    for _ in range(rounds):
        sched, z1, z2 = solve_support(inst, B, support)
        if best is not None and z1 >= best - 1e-9:
            break
        best = z1
        ev = evaluate_schedule(inst, B, sched, strict=False)
        B = _redistributed(inst, B, ev.completion)
    return best


def contiguous_support(inst, s):
    bid = {b.id: b for b in s.batches}
    sup = {}
    for m, seq in s.schedule.items():
        fs = [bid[x].family for x in seq]
        runs = [f for i, f in enumerate(fs) if i == 0 or f != fs[i - 1]]
        if len(runs) != len(set(runs)):
            return None
        if runs and inst.sigma[runs[-1]] < max(inst.sigma[f] for f in runs) - 1e-9:
            return None
        for f in set(fs):
            sup.setdefault(f, []).append(m)
    return {f: tuple(sorted(ms)) for f, ms in sup.items()}


def run(scale, ci, tau, rep, clock):
    from alns_r import alns_r
    from benchmarks import Budget
    from cpm import cpm_report
    from instances import build_matrix, generate_instance
    from layer234 import solve_support

    cell = [c for c in build_matrix() if c.scale == scale][ci]
    inst = generate_instance(cell, tau, rep)
    cf, _ = cpm_report(inst, eval_budget=10 ** 9, time_limit=clock,
                       safety_seconds=clock * 1.5)
    b0 = min(cf, key=lambda p: p.z1).batches
    cpm_levels = sorted({round(p.z2, 6) for p in cf})
    sols = alns_r(inst, Budget(10 ** 9, seconds=clock), seed=rep)
    pts = []
    for s in sols:
        sup = contiguous_support(inst, s)
        if sup is None:
            pts.append({"z1": s.z1, "z2": s.z2, "contiguous": False})
            continue
        za = solve_support(inst, s.batches, sup)[1]
        zb = solve_support(inst, b0, sup)[1]
        zc = adapt(inst, sup, b0)
        pts.append({"z1": s.z1, "z2": s.z2, "contiguous": True,
                    "a_alns_batches": za, "b_cpm_batches": zb,
                    "c_adapted": zc,
                    "level_in_cpm": round(s.z2, 6) in cpm_levels})
    return {"instance": inst.name, "scale": scale, "machines": inst.machines,
            "cpm_front": [[p.z1, p.z2] for p in cf], "alns_points": pts}


def summary(folder):
    """Rows of Table 10 from <folder>/diagnosis.jsonl, per scale."""
    import numpy as np
    recs = [json.loads(l) for l in open(os.path.join(folder, "diagnosis.jsonl"))
            if l.strip()]
    rows = [("ALNS+R points inside CPM's space", []),
            ("L2b-L4 with ALNS+R's batches reproduce or beat them", []),
            ("  share of the tardiness gap due to L2b-L4", []),
            ("CPM's own batches worse, share of points", []),
            ("  median excess in tardiness", []),
            ("CPM-A closes, median share of that excess", []),
            ("ALNS+R points at setup levels CPM never visits", [])]
    scales = [s for s in ("medium", "large")
              if any(r["scale"] == s for r in recs)]
    eps = 1e-9
    for sc in scales:
        pts = [q for r in recs if r["scale"] == sc for q in r["alns_points"]]
        cont = [q for q in pts if q["contiguous"]]
        ra = np.array([(q["a_alns_batches"] - q["z1"]) / q["z1"] for q in cont])
        rb = np.array([(q["b_cpm_batches"] - q["z1"]) / q["z1"] for q in cont])
        worse = [q for q in cont if q["b_cpm_batches"] > q["z1"] * (1 + eps)]
        closed = [(q["b_cpm_batches"] - q["c_adapted"])
                  / (q["b_cpm_batches"] - q["z1"]) for q in worse]
        vals = [len(cont) / len(pts),
                np.mean([q["a_alns_batches"] <= q["z1"] * (1 + eps)
                         for q in cont]),
                np.clip(ra, 0, None).sum() / np.clip(rb, 0, None).sum(),
                len(worse) / len(cont),
                np.median(rb),
                np.median(closed),
                np.mean([not q.get("level_in_cpm", True) for q in cont])]
        for (_, col), v in zip(rows, vals):
            col.append(f"{100 * v:.1f}%" if col is rows[4][1]
                       else f"{100 * v:.0f}%")
    names = {"medium": "60 jobs", "large": "100 jobs"}
    print(f"{'':<54}" + "".join(f"{names[s]:>10}" for s in scales))
    for label, col in rows:
        print(f"{label:<54}" + "".join(f"{c:>10}" for c in col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["medium", "large"])
    ap.add_argument("--reps", default="0,1")
    ap.add_argument("--factor", type=float)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="diag")
    ap.add_argument("--summary", metavar="FOLDER",
                    help="print Table 10 from FOLDER/diagnosis.jsonl and stop")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    if a.factor is None:
        ap.error("--factor is required for a run")
    from instances import build_matrix
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "diagnosis.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["instance"] for l in open(path)}
    tasks = []
    for scale in a.scales:
        cells = [c for c in build_matrix() if c.scale == scale]
        for ci in range(len(cells)):
            for tau in (0.3, 0.5, 0.7):
                for rep in [int(x) for x in a.reps.split(",")]:
                    tasks.append((scale, ci, tau, rep,
                                  round(MAIN_CLOCK[scale] * a.factor, 2)))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, *t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            if r["instance"] in done:
                continue
            with open(path, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r["instance"], flush=True)


if __name__ == "__main__":
    main()
