"""E2: what A3 costs when setups in fact depend on both families
(ANALYSIS_PLAN.md, section 9, amendments 3 and 5; reviewer 1 comment 1,
reviewer 2 comment 6).

True setups follow Eq. (21) of the companion study [7] on this paper's
families, light to dark in the order of instances.FAMILY_POOL:

    sigma(f, g) = gamma_f * sigma_f * (1 + lambda * max(0, d_f - d_g) / 7)

for f != g and 0 for f = g; d_f is the position in FAMILY_POOL, gamma_f = 1.8
for the chromatic families Pastel and Brown (pink and brown in [7]) and 1
otherwise, lambda in {1, 2} ([7]'s baseline and its strongest level;
lambda = 0 with gamma = 1 is A3). Leaving a dark family for a lighter one
costs more than the reverse.

A planner who keeps A3 needs one setup value per family. sigma+(f) =
max over g != f of sigma(f, g) (a wash standard sized for the worst
successor) is the primary choice; sigma_mean(f) the variant; sigma-(f) =
min over g != f gives the lower end of Proposition 2's bracket.

Methods (one record per run, all objective values under the true matrix):

  H-max    the frozen hybrid on the sigma+ instance, on (1 - D_SHARE) of the
           clock; its planned front evaluated under the true matrix. The
           record also holds H-max+D: on each machine of each planned
           schedule the batches are stably sorted light to dark (which merges
           the pieces of a family) and R is applied under the true matrix,
           within the remaining D_SHARE of the clock; originals and sorted
           schedules together form the H-max+D front. Every planned schedule
           is checked against Proposition 2: true Z1 <= planned Z1 and true
           Z2 <= planned Z2 (a violation is a bug).
  H-mean   the same with sigma_mean (lambda = 1 only); no guarantee holds, so
           the optimism of the plan is logged instead.
  H-mix    B, CPM and CPM-A on the sigma+ instance, the ALNS+R part searching
           with the true matrix (union design, full clock); every schedule
           evaluated under the true matrix.
  ALNS+R   with the true matrix in the evaluator and in its constructive
           heuristics (benchmarks.py reads core.setup_time).
  [7]+R    FFD+SA-EDDTS of [7] with the true matrix (its own format), then R
           under the true matrix, as in E1c; [7]'s front before R is logged.

Per instance and lambda the record holds the certified bracket of the true
setup optimum Z2*: LB = Corollary 3 with sigma- (valid for every schedule,
split or not), UB = true setup of the Corollary 3 schedule planned with
sigma+, LB(sigma+) (Proposition 2: LB(sigma-) <= Z2* <= UB <= LB(sigma+)),
and Z2*_contig, the exact minimum true setup over schedules in which every
family is one block on one machine (Held-Karp over subsets of families and
a partition into at most m machines); Z2* <= Z2*_contig <= UB.

Instances: the validation set at 60 and 150 jobs (replications 4-9, all
three machine arms and tightness levels: 54 instances per scale).

    python e2.py --scales medium xlarge --workers 12 --factor 0.47 \
        --params frozen.json --out results_e2
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from instances import FAMILY_POOL

DEPTH = {name: i for i, (name, _, _) in enumerate(FAMILY_POOL)}
SPAN = len(FAMILY_POOL) - 1
CHROMATIC = {"Pastel": 1.8, "Brown": 1.8}
LAMBDAS = (1.0, 2.0)
D_SHARE = 0.05
METHODS = ("H-max", "H-mean", "H-mix", "ALNS+R", "[7]+R")
HYBRID = "CPM+ALNS+R:union@0.1"


# ---------------------------------------------------------------- matrices

def true_matrix(families, sigma, lam):
    """Eq. (21) of [7] on this paper's families; hours, 4 decimals."""
    out = {}
    for f in families:
        for g in families:
            if f == g:
                out[(f, g)] = 0.0
            else:
                drop = max(0, DEPTH[f] - DEPTH[g])
                out[(f, g)] = round(CHROMATIC.get(f, 1.0) * sigma[f]
                                    * (1.0 + lam * drop / SPAN), 4)
    return out


def planning_sigma(families, matrix, how):
    """One value per family: max, mean or min over the other families."""
    out = {}
    for f in families:
        vals = [matrix[(f, g)] for g in families if g != f]
        if not vals:                     # a single family: no departure
            out[f] = 0.0
        elif how == "max":
            out[f] = max(vals)
        elif how == "min":
            out[f] = min(vals)
        elif how == "mean":
            out[f] = round(sum(vals) / len(vals), 6)
        else:
            raise ValueError(how)
    return out


def e2_instances(base, lam):
    """(true, sigma+, sigma_mean, sigma-) instances for one base instance."""
    from core import Instance
    fams = base.families
    M = true_matrix(fams, base.sigma, lam)
    tag = f"{base.name}_l{lam:g}"

    def make(sig, matrix, kind):
        return Instance(jobs=base.jobs, machines=base.machines,
                        capacity=base.capacity, proc=dict(base.proc),
                        sigma=sig, name=f"{tag}_{kind}",
                        meta=dict(base.meta, e2=kind, lam=lam),
                        setup_matrix=matrix)
    true = make(dict(base.sigma), M, "true")
    true.name = tag
    return (true, make(planning_sigma(fams, M, "max"), None, "max"),
            make(planning_sigma(fams, M, "mean"), None, "mean"),
            make(planning_sigma(fams, M, "min"), None, "min"))


def exact_contiguous(families, matrix, m):
    """Minimum true setup when every family is one block on one machine:
    Held-Karp over subsets (best open path through each subset) and a
    partition of the families into at most m subsets."""
    fams = list(families)
    k = len(fams)
    full = (1 << k) - 1
    INF = float("inf")
    path = [[INF] * k for _ in range(1 << k)]
    for j in range(k):
        path[1 << j][j] = 0.0
    for S in range(1, 1 << k):
        for j in range(k):
            v = path[S][j]
            if v == INF:
                continue
            for i in range(k):
                if S >> i & 1:
                    continue
                T = S | (1 << i)
                c = v + matrix[(fams[j], fams[i])]
                if c < path[T][i]:
                    path[T][i] = c
    best = [0.0] + [min(path[S]) for S in range(1, 1 << k)]
    # G[S] with at most g groups, g = 1..m
    G = best[:]                                   # one group
    for _ in range(m - 1):
        H = G[:]
        for S in range(1, full + 1):
            low = S & -S
            T = S
            while T:
                if T & low:
                    c = best[T] + G[S ^ T]
                    if c < H[S]:
                        H[S] = c
                T = (T - 1) & S
        G = H
    return round(G[full], 6)


def bracket(true, plus, minus):
    """Proposition 2's bracket and the exact contiguous optimum."""
    from core import evaluate_schedule, z2_lower_bound
    from hybrid import bound_point
    b, _ = bound_point(plus)
    ev = evaluate_schedule(true, b.batches, b.schedule)
    return {"LB": round(z2_lower_bound(minus), 6),
            "contig": exact_contiguous(true.families, true.setup_matrix,
                                       true.machines),
            "UB": round(ev.z2, 6),
            "LB_plus": round(z2_lower_bound(plus), 6),
            "UB_z1": round(ev.z1, 6)}


def closure_conditions(true, plus):
    """Corollary 6: (i) the families that are not last on their machine in
    B+ have, as a set, the smallest sigma- values; (ii) in B+ each of them is
    followed by a family g with sigma(f, g) = sigma-_f. Together they give
    Z2(B+) = LB(sigma-) = Z2*. Returns (i, ii, number of non-final
    families, largest number of non-final families on one machine)."""
    from hybrid import bound_point
    b, _ = bound_point(plus)
    by_id = {x.id: x for x in b.batches}
    M = true.setup_matrix
    fams = true.families
    smin = {f: min((M[(f, g)] for g in fams if g != f), default=0.0)
            for f in fams}
    nonfinal, ok2, per_machine = [], True, 0
    for mach, seq in b.schedule.items():
        blocks = []
        for x in seq:
            f = by_id[x].family
            if not blocks or blocks[-1] != f:
                blocks.append(f)
        per_machine = max(per_machine, len(blocks) - 1)
        for f, g in zip(blocks, blocks[1:]):
            nonfinal.append(f)
            ok2 = ok2 and abs(M[(f, g)] - smin[f]) < 1e-9
    e = len(nonfinal)
    smallest = sorted(smin.values())[:e]
    ok1 = abs(sum(smin[f] for f in nonfinal) - sum(smallest)) < 1e-9
    return ok1, ok2, e, per_machine


# ---------------------------------------------------------------- methods

def _hybrid(frozen):
    """(mode, share) of the frozen hybrid."""
    from hybrid import parse_hybrid
    return parse_hybrid((frozen or {}).get("hybrid", HYBRID))


def light_to_dark(true, sol):
    """Batches of every machine stably sorted light to dark; returns the
    sorted solution evaluated under the true matrix and how many machines
    end on another family."""
    from benchmarks import Solution
    from core import evaluate_schedule
    by_id = {b.id: b for b in sol.batches}
    sched, changed = {}, 0
    for mach, seq in sol.schedule.items():
        new = sorted(seq, key=lambda x: DEPTH[by_id[x].family])
        if seq and by_id[seq[-1]].family != by_id[new[-1]].family:
            changed += 1
        sched[mach] = new
    ev = evaluate_schedule(true, sol.batches, sched)
    return Solution(sol.batches, sched, ev.z1, ev.z2), changed


def _true(true, sols):
    from benchmarks import Solution
    from core import evaluate_schedule
    out = []
    for s in sols:
        ev = evaluate_schedule(true, s.batches, s.schedule)
        out.append(Solution(s.batches, s.schedule, ev.z1, ev.z2))
    return out


def run_planned(true, plan, clock, seed, frozen, check_guarantee):
    """H-max / H-mean: hybrid on the planning instance, then +D."""
    from alns_r import r_on_front
    from benchmarks import Budget, filter_front
    from hybrid import cpm_alns_r
    from params import params_for
    t0 = time.time()
    mode, share = _hybrid(frozen)
    planned, evals = cpm_alns_r(plan, clock * (1 - D_SHARE), seed=seed,
                                share=share, mode=mode,
                                alns_params=params_for("ALNS+R", frozen))
    real = _true(true, planned)
    info = {"planned": [[s.z1, s.z2] for s in planned]}
    viol1 = [r.z1 - p.z1 for p, r in zip(planned, real) if r.z1 > p.z1 + 1e-6]
    viol2 = [r.z2 - p.z2 for p, r in zip(planned, real) if r.z2 > p.z2 + 1e-6]
    info["optimistic_z1"] = len(viol1)
    info["optimistic_z2"] = len(viol2)
    info["max_excess_z1"] = round(max(viol1, default=0.0), 6)
    info["max_excess_z2"] = round(max(viol2, default=0.0), 6)
    if check_guarantee:
        info["guarantee_ok"] = not viol1 and not viol2
    sorted_, changed = [], 0
    for s in sorted(real, key=lambda s: s.z1):
        d, c = light_to_dark(true, s)
        sorted_.append(d)
        changed += c
    left = max(clock - (time.time() - t0), 0.01)
    b = Budget(10 ** 9, seconds=left)
    improved, rinfo = r_on_front(true, sorted_, b,
                                 params_for("ALNS+R", frozen)["r_rounds"])
    info.update({"d_" + k: v for k, v in rinfo.items()})
    info["d_last_changed"] = changed
    info["d_evaluations"] = b.used
    plus_d = filter_front(real + improved)
    info["points_D"] = [[s.z1, s.z2] for s in plus_d]
    info["D_from_sorted"] = sum(1 for s in plus_d
                                if any(s is x for x in improved))
    return filter_front(real), evals + len(sorted_) + b.used, info, plus_d


def run_task(scale, ci, tau, rep, lam, method, seed, clock, frozen):
    from alns_r import alns_r
    from benchmarks import Budget, filter_front
    from core import evaluate_schedule
    from hybrid import cpm_alns_r
    from params import params_for
    from variants import cells, make_instance
    import comp7

    base = make_instance(cells(scale)[ci], tau, rep)
    true, plus, mean, minus = e2_instances(base, lam)
    br = bracket(true, plus, minus)
    w0, c0 = time.time(), time.process_time()
    extra, all_sols = {}, None
    if method == "H-max":
        front, evals, extra, all_sols = run_planned(
            true, plus, clock, seed, frozen, True)
    elif method == "H-mean":
        front, evals, extra, all_sols = run_planned(
            true, mean, clock, seed, frozen, False)
    elif method == "H-mix":
        mode, share = _hybrid(frozen)
        if mode != "union":
            raise ValueError("H-mix needs the union design")
        _, evals, parts = cpm_alns_r(
            plus, clock, seed=seed, share=share, mode=mode,
            alns_params=params_for("ALNS+R", frozen), return_parts=True,
            search_inst=true)
        cpm_true = _true(true, parts["cpm"])
        front = filter_front(cpm_true + list(parts["alns"]))
        extra["origin"] = ["cpm" if any(s is c for c in cpm_true) else "alns"
                           for s in front]
        extra["seconds_parts"] = parts["seconds"]
    elif method == "ALNS+R":
        b = Budget(10 ** 9, seconds=clock)
        front = alns_r(true, b, seed=seed, **params_for("ALNS+R", frozen))
        evals = b.used
    elif method == "[7]+R":
        rr = params_for("ALNS+R", frozen)["r_rounds"]
        front, info = comp7.comp7_r_front(true, clock, seed=seed,
                                          use_sa=True, rounds=rr)
        extra["comp7"] = info
        evals = None
    else:
        raise ValueError(method)
    secs = time.time() - w0
    cpu = time.process_time() - c0

    pts, ok = [], True
    for s in front:
        ev = evaluate_schedule(true, s.batches, s.schedule, strict=False)
        if (not ev.feasible or abs(ev.z1 - s.z1) > 1e-6
                or abs(ev.z2 - s.z2) > 1e-6):
            ok = False
            continue
        pts.append([ev.z1, ev.z2])
    if all_sols is not None:
        for s in all_sols:
            ev = evaluate_schedule(true, s.batches, s.schedule, strict=False)
            if (not ev.feasible or abs(ev.z1 - s.z1) > 1e-6
                    or abs(ev.z2 - s.z2) > 1e-6):
                ok = False
    rec = {"instance": true.name, "base_instance": base.name,
           "scale": scale, "n": len(base.jobs), "machines": base.machines,
           "families": len(base.families), "excess": base.excess,
           "tau": tau, "replication": rep, "lam": lam, "method": method,
           "seed": seed, "clock": clock, "clock_mult": 1.0,
           "seconds": round(secs, 3), "cpu_seconds": round(cpu, 3),
           "evaluations": evals,
           "evals_per_second": (round(evals / secs, 1)
                                if evals and secs > 0 else None),
           "feasible": ok, "bracket": br, "z2_bound": br["LB"],
           "points": pts}
    rec.update(extra)
    return rec


# ---------------------------------------------------------------- driver

def main():
    from fair_compare import _cells, clock_for
    from jsonl_io import append_record, read_records
    from params import load_frozen
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["medium", "xlarge"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--lambdas", default="1,2")
    ap.add_argument("--reps", default="4,5,6,7,8,9")
    ap.add_argument("--taus", default="0.3,0.5,0.7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--factor", type=float, default=0.47)
    ap.add_argument("--clock-mult", type=float, default=1.0,
                    help="testing only")
    ap.add_argument("--params", default="frozen.json")
    ap.add_argument("--out", default="results_e2")
    a = ap.parse_args()
    methods = a.methods.split(",")
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; one of {METHODS}")
    lams = [float(x) for x in a.lambdas.split(",")]
    reps = [int(x) for x in a.reps.split(",")]
    taus = [float(x) for x in a.taus.split(",")]
    frozen = load_frozen(a.params)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = {(r["instance"], r["method"], r["seed"])
            for r in read_records(path, repair=True)}
    from variants import make_instance
    tasks = []
    for scale in a.scales:
        for ci, cell in enumerate(_cells(scale)):
            clock = round(clock_for(cell.n) * a.factor * a.clock_mult, 3)
            for tau, rep, lam, m in itertools.product(taus, reps, lams,
                                                      methods):
                if m == "H-mean" and lam != 1.0:
                    continue
                name = f"{make_instance(cell, tau, rep).name}_l{lam:g}"
                if (name, m, rep) not in done:
                    tasks.append((scale, ci, tau, rep, lam, m, rep, clock,
                                  frozen))
    tasks.sort(key=lambda t: -t[7])
    total = len(tasks)
    cpu_h = sum(t[7] for t in tasks) / 3600
    print(f"{total} runs to do; {cpu_h:.1f} CPU-hours, about "
          f"{cpu_h / max(a.workers, 1):.1f} h on {a.workers} workers",
          flush=True)
    t0, k = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, *t) for t in tasks]
        for f in as_completed(futs):
            rec = f.result()
            append_record(path, rec)
            k += 1
            z1 = min((p[0] for p in rec["points"]), default=None)
            z2 = min((p[1] for p in rec["points"]), default=None)
            flag = "" if rec["feasible"] else "  INFEASIBLE POINT DROPPED"
            if rec.get("guarantee_ok") is False:
                flag += "  GUARANTEE VIOLATED"
            print(f"[{k}/{total}] {(time.time() - t0) / 60:6.1f} min  "
                  f"{rec['instance']:<30} {rec['method']:<7} "
                  f"{rec['seconds']:7.1f} s  Z1min {z1}  Z2min {z2}  "
                  f"[LB {rec['bracket']['LB']}, UB {rec['bracket']['UB']}]"
                  f"{flag}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
