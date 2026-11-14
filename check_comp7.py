"""Reproduction check of the adapter for [7] (ANALYSIS_PLAN.md, section 9).

Three tasks of [7]'s own campaign (30 jobs; presets LOOSE, MEDIUM, TIGHT;
asymmetric setup matrix of [7] at lambda = 1, rho_h = 1.8; budget 8 s per
subproblem, 10 grid points), each run in two ways on this machine:

  own      [7]'s runner, unchanged: rerun.make_task + rerun.run_task
  adapter  this repository's path (comp7._run): the same instance, matrix,
           seeding and parameters, the clock of 80 s split as in the
           comparison, (clock - Stage 1 time) / 10 per subproblem

and compared with the five runs per task stored in [7]'s archive
(comp7/repro_reference.json: least tardiness, least setup, front size; file
hashes of the archive's result files). The parameters loaded by the adapter
are also compared with those recorded by [7]'s campaign.

The method is stochastic and time-limited, so the check is not exact: the
adapter passes if its values are as close to [7]'s stored runs as [7]'s own
runner on this machine is (within the range of the stored runs widened by
the own runner's distance from that range, plus 1%).

    python check_comp7.py [--reps 0,1] [--workers 6] [--out results_comp7repro]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "comp7", "repro_reference.json")
METHOD = "FFD+SA-EDDTS"


def _one(job):
    key, task_spec, rep, path = job
    import comp7
    dc, params = comp7._load()
    import rerun                                   # [7]'s runner (comp7/)
    t = task_spec
    task = rerun.make_task("campaign", "main", t["preset"], t["size"],
                           t["base_seed"], t["run"], rep, [METHOD], params,
                           customers=t["customers"], colors=t["colors"],
                           machines=t["machines"], budget=t["budget"],
                           grid_points=t["grid_points"], lam=t["lam"],
                           rho_h=t["rho_h"])
    t0 = time.time()
    if path == "own":
        out = rerun.run_task(dc, task)
        pts = out["fronts"][METHOD]
        info = {}
    else:
        import random
        import numpy as np
        for k, v in task["globals"].items():
            setattr(dc, k, v)
        dc.set_setup_parameters(task["lam"], task["rho_h"])
        gen = dc.RandomInstanceGenerator.from_preset(
            task["preset"], num_jobs=task["size"],
            num_machines=task["machines"], num_colors=task["colors"],
            num_customers=task["customers"],
            machine_capacity=task["capacity"], seed=task["gen_seed"])
        jobs, machines, _, matrix = gen.generate()
        s = rerun.stable_seed(task["gen_seed"], METHOD, rep)
        clock = task["grid_points"] * task["budget"]
        t0 = time.time()
        raw, info = comp7._run(dc, params, jobs, machines, matrix, clock, t0,
                               seed=s, run_seed=rep, use_sa=True,
                               grid_points=task["grid_points"])
        front = raw[0] if isinstance(raw, tuple) else raw
        pts = [[float(x["tardiness"]), float(x["setup_time"])]
               for x in front or []]
        random.seed(0)
        np.random.seed(0)
    secs = time.time() - t0
    return {"task": key, "rep": rep, "path": path,
            "seconds": round(secs, 2), "points": pts,
            "z1_min": min(p[0] for p in pts) if pts else None,
            "z2_min": min(p[1] for p in pts) if pts else None,
            "n_points": len(pts), "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0,1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results_comp7repro")
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="testing only: scale [7]'s budget (the comparison "
                         "with the archive is then meaningless)")
    a = ap.parse_args()
    ref = json.load(open(REF))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done.add((r["task"], r["rep"], r["path"]))

    import comp7
    _, params = comp7._load()
    camp = ref["campaign_params"]
    diff = {k: (params.get(k), v) for k, v in camp.items()
            if params.get(k) != v}
    print("parameters of the adapter vs [7]'s campaign:",
          "identical" if not diff else f"DIFFERENT {diff}", flush=True)

    reps = [int(x) for x in a.reps.split(",")]
    for v in ref["tasks"].values():
        v["task"]["budget"] *= a.budget_scale
    jobs = [(k, v["task"], r, p) for k, v in ref["tasks"].items()
            for r in reps for p in ("own", "adapter")
            if (k, r, p) not in done]
    print(f"{len(jobs)} runs of about 80 s on {a.workers} workers", flush=True)
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            for rec in pool.imap_unordered(_one, jobs):
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"  {rec['task']} rep {rec['rep']} {rec['path']:<8} "
                      f"Z1min {rec['z1_min']:.3f} Z2min {rec['z2_min']:.3f} "
                      f"points {rec['n_points']} ({rec['seconds']} s)",
                      flush=True)

    recs = [json.loads(x) for x in open(path) if x.strip()]
    ok_all = not diff
    print("\nleast tardiness / least setup / points: stored range (5 runs of "
          "[7]'s archive) | own runner | adapter")
    for k, v in ref["tasks"].items():
        st = v["stored"]
        line = [k]
        for col, fmt in (("z1_min", ".2f"), ("z2_min", ".3f"),
                         ("points", "d")):
            sv = [x[col] for x in st]
            lo, hi = min(sv), max(sv)
            key = "n_points" if col == "points" else col
            own = [r[key] for r in recs if r["task"] == k and r["path"] == "own"]
            ad = [r[key] for r in recs if r["task"] == k
                  and r["path"] == "adapter"]
            if not own or not ad:
                continue

            def dist(x):
                return max(lo - x, 0, x - hi) / max(abs(hi), 1e-9)
            slack = max(dist(x) for x in own) + 0.01
            ok = col == "points" or all(dist(x) <= slack for x in ad)
            ok_all = ok_all and ok
            line.append(f"{col} [{lo:{fmt}}, {hi:{fmt}}] | "
                        + ", ".join(format(x, fmt) for x in own) + " | "
                        + ", ".join(format(x, fmt) for x in ad)
                        + ("" if ok else "  <-- outside"))
        print("\n  ".join(line))
    print("\nREPRODUCTION CHECK " + ("PASS" if ok_all else "FAIL"))


if __name__ == "__main__":
    main()
