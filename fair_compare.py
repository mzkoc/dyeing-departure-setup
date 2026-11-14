"""Equal wall-clock comparison (revision, round 1).

Methods
    CPM, CPM-blind, CPM+R                      deterministic
    ALNS, NSGA-II                              the submitted benchmarks
    ALNS+R, NSGA-II+R                          benchmarks with the step R
    ALNS+R#0..9, NSGA-II+R#0..9                calibration rows (params.py)
    CPM+ALNS+R[:mode][@share]                  the hybrid (hybrid.py)
    FFD+SA-EDDTS, FFD-EDDTS, FFD+SA-EDDTS+R    the method of [7], and with R (comp7.py)

Instances: every cell of the requested scales (the verification scale, 12-20
jobs, included), tightness 0.3/0.5/0.7, the replications given by --reps.
Replications 0-1 are the development set, 2-3 the calibration set, 4-9 the
validation set. --variant applies one sensitivity factor (variants.py) and
--min-excess keeps only the cells with e = |F| - m at least that large;
scales x200 and x300 are the scalability cells (variants.EXTRA_SCALES).
A method "<hybrid>~<knob>=<value>" runs the hybrid with one of CPM's
hand-set parameters changed (cpm_params.py).

Clock: the mean CPU time of CPM in the main study at that scale (7, 136, 302,
281 s for 30, 60, 100, 150 jobs), times --factor (measured by calibrate.py)
and --clock-mult (for budget robustness). Scales with no CPM anchor use a
rule fixed in advance: below 30 jobs the 30-job clock times n/30, above 150
jobs the 150-job clock times n/150.

--params names a JSON file of frozen parameters (written by tune.py); plain
ALNS+R and NSGA-II+R and the ALNS+R part of the hybrid then use them.

One task is one (instance, method, seed) run; tasks are spread over --workers
processes, longest first. Every returned point is re-checked through
core.evaluate_schedule before it is written. Each record carries the verified
front, the departures per family behind every point (so that any
departure-dependent water model can be evaluated afterwards), wall and CPU
seconds, evaluations, the parameters used and, for the hybrid, its parts.
Results are appended to <out>/results.jsonl; a rerun skips what is there.
On start the file is checked (jsonl_io.py): a line cut off by a power cut is
set aside, and runs whose wall time exceeds the clock by more than 25% plus
2 s (computer asleep or process stopped) are set aside and run again.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

CPM_ANCHOR = {30: 7.0, 60: 136.0, 100: 302.0, 150: 281.0}
BASE = ("CPM", "CPM+R", "CPM-blind", "ALNS", "ALNS+R", "NSGA-II",
        "NSGA-II+R", "CPM+ALNS+R", "FFD+SA-EDDTS", "FFD-EDDTS",
        "FFD+SA-EDDTS+R")
COMP7 = ("FFD+SA-EDDTS", "FFD-EDDTS", "FFD+SA-EDDTS+R")
DETERMINISTIC = {"CPM", "CPM+R", "CPM-blind"}


def clock_for(n: int) -> float:
    """Clock before --factor, from the CPM anchors."""
    if n in CPM_ANCHOR:
        return CPM_ANCHOR[n]
    if n < 30:
        return CPM_ANCHOR[30] * n / 30.0
    if n > 150:
        return CPM_ANCHOR[150] * n / 150.0
    raise ValueError(f"no clock rule for {n} jobs")


def check_method(m: str) -> None:
    from cpm_params import split
    from hybrid import parse_hybrid
    from params import parse_variant
    base, knob, _ = split(m)
    if knob is not None:
        if parse_hybrid(base) is None:
            raise ValueError(m)
        return
    if m in BASE or parse_hybrid(m) is not None or parse_variant(m):
        return
    raise ValueError(m)


def _cells(scale):
    from variants import cells
    return cells(scale)


def _departures(inst, s):
    bid = {b.id: b for b in s.batches}
    dep = Counter()
    for seq in s.schedule.values():
        fs = [bid[x].family for x in seq]
        for a, b in zip(fs, fs[1:]):
            if a != b:
                dep[a] += 1
    return dict(dep)


def run_task(scale, ci, tau, rep, method, seed, clock, frozen, mult,
             variant=""):
    from benchmarks import Budget, alns, nsga2
    from core import evaluate_schedule, z2_lower_bound
    from cpm import cpm_report
    from cpm_params import override, split
    from hybrid import cpm_alns_r, parse_hybrid
    from params import params_for, parse_variant
    from variants import make_instance, perturbed_points

    inst = make_instance(_cells(scale)[ci], tau, rep, variant)
    run_name, knob, knob_value = split(method)
    used_params = {}
    parts = None
    extra = None
    if method in COMP7:
        import comp7
        comp7._load()            # import the code of [7] before the clock
    w0, c0 = time.time(), time.process_time()
    hyb = parse_hybrid(run_name)
    var = parse_variant(method)
    base = var[0] if var else method
    if hyb is not None:
        used_params = params_for("ALNS+R", frozen)
        if knob is not None:
            used_params = dict(used_params, **{"cpm_" + knob: knob_value})
        with override(method):
            front, evals, parts = cpm_alns_r(
                inst, clock, seed=seed, share=hyb[1], mode=hyb[0],
                alns_params=params_for("ALNS+R", frozen),
                return_parts=True)
    elif method == "CPM+R":
        from cpm_r import cpm_r
        rr = params_for("ALNS+R", frozen).get("r_rounds", 3)
        used_params = {"r_rounds": rr}
        front, evals, extra = cpm_r(inst, clock, rounds=rr, return_info=True)
    elif method in ("CPM", "CPM-blind"):
        front, diag = cpm_report(inst,
                                 variant="cpm" if method == "CPM" else "blind",
                                 eval_budget=10 ** 9, time_limit=clock,
                                 safety_seconds=clock * 1.5)
        evals = diag["evaluations"]
    elif base in ("ALNS+R", "NSGA-II+R"):
        from alns_r import alns_r
        from nsga2_r import nsga2_r
        used_params = params_for(method, frozen)
        b = Budget(10 ** 9, seconds=clock)
        front = (alns_r if base == "ALNS+R" else nsga2_r)(
            inst, b, seed=seed, **used_params)
        evals = b.used
    elif method in ("ALNS", "NSGA-II"):
        b = Budget(10 ** 9, seconds=clock)
        front = (alns if method == "ALNS" else nsga2)(inst, b, seed=seed)
        evals = b.used
    elif method in COMP7:
        if method in comp7.WITH_R:
            rr = params_for("ALNS+R", frozen).get("r_rounds", 3)
            used_params = {"r_rounds": rr}
            front, extra = comp7.comp7_r_front(
                inst, clock, seed=seed, use_sa=comp7.METHODS[method],
                rounds=rr)
        else:
            front, extra = comp7.comp7_front(inst, clock, seed=seed,
                                             use_sa=comp7.METHODS[method])
        evals = None             # [7]'s code does not count evaluations
    else:
        raise ValueError(method)
    secs = time.time() - w0
    cpu = time.process_time() - c0

    pts, deps, origin, ok, dep_ok = [], [], [], True, True
    kept = []
    for i, s in enumerate(front):
        ev = evaluate_schedule(inst, s.batches, s.schedule, strict=False)
        if (not ev.feasible or abs(ev.z1 - s.z1) > 1e-6
                or abs(ev.z2 - s.z2) > 1e-6):
            ok = False
            continue
        dep = _departures(inst, s)
        if abs(sum(inst.sigma[f] * k for f, k in dep.items()) - ev.z2) > 1e-6:
            dep_ok = False
        pts.append([ev.z1, ev.z2])
        kept.append(s)
        deps.append(dep)
        if parts is not None:
            origin.append(parts["origin"][i])
    rec = {"instance": inst.name, "scale": scale, "n": len(inst.jobs),
           "machines": inst.machines, "families": len(inst.proc),
           "tau": tau, "replication": rep, "excess": inst.excess,
           "method": method, "seed": seed, "clock": clock,
           "clock_mult": mult, "params": used_params,
           "seconds": round(secs, 3), "cpu_seconds": round(cpu, 3),
           "evaluations": evals,
           "evals_per_second": (round(evals / secs, 1)
                                if evals is not None and secs > 0 else None),
           "feasible": ok, "departures_consistent": dep_ok,
           "z2_bound": z2_lower_bound(inst), "points": pts,
           "departures": deps}
    if variant:
        rec["variant"] = variant
        rec["capacity"] = inst.capacity
        rec["customers"] = len(inst.customers)
    if variant == "perturb":
        # after the clock: every planned schedule on five perturbed instances
        rec["perturbed"] = perturbed_points(inst, kept)
    if extra is not None:
        rec["comp7" if method in COMP7 else "r_info"] = extra
    if parts is not None:
        rec["origin"] = origin
        rec["parts"] = {
            "alns": [[s.z1, s.z2] for s in parts["alns"]],
            "cpm": [[s.z1, s.z2] for s in parts["cpm"]],
            "bound": [parts["bound"].z1, parts["bound"].z2],
            "seconds": parts.get("seconds")}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+",
                    default=["small", "medium", "large", "xlarge"])
    ap.add_argument("--methods", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--clock-mult", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--taus", default="0.3,0.5,0.7")
    ap.add_argument("--reps", default="0,1")
    ap.add_argument("--params", default="")
    ap.add_argument("--out", default="results_fair")
    ap.add_argument("--variant", default="",
                    help="sensitivity factor (variants.py)")
    ap.add_argument("--min-excess", type=int, default=None,
                    help="only cells with e = |F| - m at least this")
    ap.add_argument("--redo", default="",
                    help="comma-separated methods whose finished runs in "
                         "<out> are set aside (results.jsonl.redo) and run "
                         "again, e.g. after a change to that method")
    a = ap.parse_args()

    from jsonl_io import append_record, read_records
    from params import load_frozen
    from variants import VARIANTS, make_instance

    methods = [m for m in a.methods.split(",") if m]
    bad = []
    for m in methods:
        try:
            check_method(m)
        except ValueError:
            bad.append(m)
    if bad:
        raise SystemExit(f"unknown methods: {bad}")
    if a.variant and a.variant not in VARIANTS:
        raise SystemExit(f"unknown variant {a.variant}; one of {VARIANTS}")
    frozen = load_frozen(a.params)
    if a.params:
        print(f"frozen parameters from {a.params}: {frozen}", flush=True)
    taus = [float(x) for x in a.taus.split(",")]
    reps = [int(x) for x in a.reps.split(",")]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")

    recs = read_records(path, repair=True)
    redo = {m for m in a.redo.split(",") if m}
    if redo and any(r["method"] in redo for r in recs):
        import json
        keep = [r for r in recs if r["method"] not in redo]
        gone = [r for r in recs if r["method"] in redo]
        with open(path + ".redo", "a") as fh:
            for r in gone:
                fh.write(json.dumps(r) + "\n")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
        print(f"{len(gone)} run(s) of {sorted(redo)} set aside in "
              f"{path}.redo; they will be run again", flush=True)
        recs = keep
    done = set()
    for r in recs:
        done.add((r["instance"], r["method"], r["seed"],
                  r.get("clock_mult", 1.0)))

    tasks = []
    for scale in a.scales:
        cells = _cells(scale)
        if not cells:
            raise SystemExit(f"no cells for scale {scale}")
        for ci, cell in enumerate(cells):
            if a.min_excess is not None and cell.excess < a.min_excess:
                continue
            clock = round(clock_for(cell.n) * a.factor * a.clock_mult, 3)
            for tau in taus:
                for rep in reps:
                    name = make_instance(cell, tau, rep, a.variant).name
                    for m in methods:
                        seeds = ([rep] if m in DETERMINISTIC else
                                 [rep + 1000 * i for i in range(a.seeds)])
                        for s in seeds:
                            if (name, m, s, a.clock_mult) not in done:
                                tasks.append((scale, ci, tau, rep, m, s,
                                              clock, frozen, a.clock_mult,
                                              a.variant))
    tasks.sort(key=lambda t: (-t[6], t[0], t[1], t[2], t[3], t[5],
                              methods.index(t[4])))
    total = len(tasks)
    cpu = sum(t[6] for t in tasks)
    print(f"{total} runs to do ({len(done)} already in {path}); "
          f"at most {cpu / 3600:.1f} CPU-hours, about "
          f"{cpu / 3600 / max(a.workers, 1):.1f} h on {a.workers} workers",
          flush=True)
    if not tasks:
        return

    t0 = time.time()
    k = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, *t) for t in tasks]
        for f in as_completed(futs):
            rec = f.result()
            append_record(path, rec)
            k += 1
            z1 = min((p[0] for p in rec["points"]), default=None)
            flag = "" if rec["feasible"] else "  INFEASIBLE POINT DROPPED"
            if not rec["departures_consistent"]:
                flag += "  DEPARTURES != Z2"
            print(f"[{k}/{total}] {(time.time() - t0) / 60:6.1f} min  "
                  f"{rec['instance']:<28} {rec['method']:<24} "
                  f"seed {rec['seed']:<5} {rec['seconds']:7.1f} s  "
                  f"Z1min {z1}{flag}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
