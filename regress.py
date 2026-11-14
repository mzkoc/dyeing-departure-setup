"""Regression and guarantee checks for the v6 package. Run before any
calibration or validation run; every check must print PASS.

  1. golden    ALNS+R and NSGA-II+R with the default parameters (plain names
               and row #0) reproduce, point for point, the fronts of the v5
               code (golden_v5.json; evaluation-count budgets, so exact on
               any machine).
  2. bound     the Corollary 3 schedule attains the Z2 lower bound on every
               cell of the verification and 30-job scales.
  3. z2        the hybrid (union, anchor, seed) contains a Z2-optimal point
               even when the clock is far too short for CPM to start.
  4. records   fair_compare.run_task: every point feasible, the departures
               per family reproduce Z2, the hybrid's origin labels cover its
               front and its front is the non-dominated union of its parts.
  5. adapt     CPM-A never returns a worse point than it is given.
  6. comp7     the method of [7] (comp7.py): every returned schedule is
               feasible here and its objectives agree with those [7]'s code
               reports; skipped (not failed) if Pyomo is not installed.
  7. jsonl     a cut-off line and a suspended run are set aside on resume.
  8. v9a       the generator's new options leave every instance of the main
               study unchanged (hash); each sensitivity variant changes only
               its factor; the perturbation draws are reproducible and in
               range; the clocks of 200 and 300 jobs; CPM parameter
               overrides take effect and are restored; a perturb record and
               a record with a CPM parameter changed are complete.
  9. E2        the true matrix follows Eq. (21) of [7]; a matrix equal to A3
               evaluates every schedule as A3 does; Held-Karp equals brute
               force; Proposition 2's bracket is ordered and closes on the
               E2 instances checked; each E2 method gives feasible fronts,
               sigma+ planning is never optimistic, [7] agrees with the
               evaluator under the matrix.

    python regress.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys

from instances import build_matrix, generate_instance

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not ok:
        FAILS.append(name)


def inst_of(scale, ci, tau, rep):
    cells = [c for c in build_matrix(include_verify=True) if c.scale == scale]
    return generate_instance(cells[ci], tau, rep)


def golden():
    from alns_r import alns_r
    from benchmarks import Budget
    from nsga2_r import nsga2_r
    from params import params_for
    g = json.load(open(os.path.join(os.path.dirname(__file__) or ".",
                                    "golden_v5.json")))
    for sc, ci, tau, rep in g["cases"]:
        inst = inst_of(sc, ci, tau, rep)
        for name, fn in (("ALNS+R", alns_r), ("NSGA-II+R", nsga2_r)):
            want = g["gold"][f"{inst.name}|{name}"]
            for label, kw in (("default", {}),
                              ("#0", params_for(name + "#0", {}))):
                f = fn(inst, Budget(g["budget"]), seed=rep, **kw)
                got = sorted([round(s.z1, 4), round(s.z2, 4)] for s in f)
                check(f"golden {inst.name} {name} {label}", got == want,
                      "" if got == want else f"got {got} want {want}")


def bound():
    from core import z2_lower_bound
    from hybrid import bound_point
    bad = []
    n = 0
    for scale in ("verify", "small"):
        cells = [c for c in build_matrix(include_verify=True)
                 if c.scale == scale]
        for ci in range(len(cells)):
            for tau in (0.3, 0.5, 0.7):
                inst = inst_of(scale, ci, tau, 0)
                s, _ = bound_point(inst)
                n += 1
                if abs(s.z2 - z2_lower_bound(inst)) > 1e-9:
                    bad.append(inst.name)
    check(f"bound attains Z2 lower bound on {n} instances", not bad,
          f"failed: {bad}" if bad else "")


def z2_guarantee():
    from core import z2_lower_bound
    from hybrid import cpm_alns_r
    bad = []
    n = 0
    cells = [c for c in build_matrix(include_verify=True)
             if c.scale == "verify"]
    for ci in range(len(cells)):
        inst = inst_of("verify", ci, 0.5, 2)
        for mode in ("union", "anchor", "seed"):
            out, _ = cpm_alns_r(inst, 0.2, seed=2, share=0.001, mode=mode)
            n += 1
            if min(s.z2 for s in out) > z2_lower_bound(inst) + 1e-9:
                bad.append((inst.name, mode))
    check(f"hybrid keeps a Z2-optimal point at share 0.001, 0.2 s "
          f"({n} runs)", not bad, f"failed: {bad}" if bad else "")


def records():
    from analysis import _nondominated
    from fair_compare import run_task
    from params import load_frozen
    frozen = load_frozen("")
    tasks = [("verify", 2, 0.5, 2, "CPM+ALNS+R:union@0.05", 2, 1.0),
             ("small", 0, 0.5, 2, "CPM+ALNS+R:union@0.05", 2, 2.0),
             ("small", 1, 0.3, 3, "ALNS+R#4", 3, 1.0),
             ("small", 2, 0.7, 2, "NSGA-II+R#7", 2, 1.0),
             ("verify", 0, 0.3, 3, "CPM", 3, 0.5)]
    for sc, ci, tau, rep, m, seed, clock in tasks:
        r = run_task(sc, ci, tau, rep, m, seed, clock, frozen, 1.0)
        ok = r["feasible"] and r["departures_consistent"] and r["points"]
        detail = ""
        if "parts" in r:
            ok = ok and len(r["origin"]) == len(r["points"])
            allp = [tuple(p) for p in r["parts"]["alns"] + r["parts"]["cpm"]]
            nd = sorted(set((round(a, 6), round(b, 6))
                            for a, b in _nondominated(allp)))
            got = sorted(set((round(a, 6), round(b, 6))
                             for a, b in r["points"]))
            ok = ok and nd == got
            detail = f"origin {sorted(set(r['origin']))}"
        check(f"record {r['instance']} {m}", bool(ok), detail)


def adapt_check():
    from benchmarks import Solution
    from cpm import cpm_report
    from hybrid import adapt
    worse = 0
    n = 0
    for ci in range(3):
        inst = inst_of("small", ci, 0.5, 2)
        front, _ = cpm_report(inst)
        for p in front:
            s = adapt(inst, p.support,
                      Solution(p.batches, p.schedule, p.z1, p.z2))
            n += 1
            if s.z1 > p.z1 + 1e-9 or abs(s.z2 - p.z2) > 1e-9:
                worse += 1
    check(f"CPM-A never worsens a point ({n} points)", worse == 0)


def comp7_check():
    try:
        import comp7
        comp7._load()
    except SystemExit as e:
        print(f"SKIP  comp7: {e}")
        return
    bad, n, pts = 0, 0, 0
    for sc, ci in (("verify", 0), ("verify", 5), ("small", 1)):
        inst = inst_of(sc, ci, 0.5, 2)
        f, info = comp7.comp7_front(inst, 1.5, seed=2)
        n += 1
        pts += len(f)
        bad += info["rejected"] + (len(f) == 0)
        # [7]+R: R keeps every setup level of the front it improves
        fr, ir = comp7.comp7_r_front(inst, 1.5, seed=2)
        pre = {round(z2, 6) for _, z2 in ir["pre_r"]}
        bad += any(round(s.z2, 6) not in pre for s in fr)
        bad += ir["rejected"]
    check(f"comp7 and comp7+R fronts verified on {n} instances "
          f"({pts} points)", bad == 0)


def jsonl_check():
    import json
    import tempfile
    from jsonl_io import read_records
    d = tempfile.mkdtemp()
    p = os.path.join(d, "results.jsonl")
    ok = {"instance": "a", "method": "m", "seed": 0, "seconds": 10.0,
          "clock": 10.0}
    slept = dict(ok, instance="b", seconds=4000.0)
    with open(p, "w") as fh:
        fh.write(json.dumps(ok) + "\n" + json.dumps(slept) + "\n")
        fh.write('{"instance": "c", "meth')
    got = read_records(p, repair=True, quiet=True)
    again = read_records(p)
    check("jsonl cut-off line and suspended run set aside",
          [r["instance"] for r in got] == ["a"]
          and [r["instance"] for r in again] == ["a"]
          and os.path.exists(p + ".bad") and os.path.exists(p + ".suspended"))


INSTANCE_HASH = ("e5de499e39b965ee654586f68f3a7385"
                 "dabda3fb53fef70b7921e5d0b0d48747")


def v9a_check():
    import hashlib
    import statistics as st
    from collections import Counter
    import cpm_params as cp
    from fair_compare import clock_for, run_task
    from instances import TAU_LEVELS
    from params import load_frozen
    from variants import cells, make_instance, perturbed, VARIANTS
    h = hashlib.sha256()
    for c in build_matrix(include_verify=True):
        for t in TAU_LEVELS:
            for r in range(10):
                i = generate_instance(c, t, r)
                h.update(json.dumps(
                    [i.name, [(j.id, j.size, j.family, j.customer, j.due)
                              for j in i.jobs], i.capacity, i.proc, i.sigma],
                    sort_keys=True).encode())
    check("v9a instances of the main study unchanged (2520, hash)",
          h.hexdigest() == INSTANCE_HASH, h.hexdigest()[:16])
    bad = []
    skew = Counter()
    for sc in ("medium", "xlarge"):
        cs = [c for c in cells(sc) if c.excess > 0]
        if len(cs) != 1:
            bad.append(f"{sc}: {len(cs)} cells with e > 0")
            continue
        c = cs[0]
        for tau in TAU_LEVELS:
            b = make_instance(c, tau, 4)
            jobs_b = [(j.size, j.family, j.customer, j.due) for j in b.jobs]
            for v in VARIANTS:
                i = make_instance(c, tau, 4, v)
                jobs = [(j.size, j.family, j.customer, j.due) for j in i.jobs]
                if not i.name.endswith("_" + v):
                    bad.append(f"{v} name")
                if v in ("perturb", "sigx2", "sigx4"):
                    f = {"perturb": 1, "sigx2": 2, "sigx4": 4}[v]
                    if jobs != jobs_b or i.proc != b.proc or any(
                            abs(i.sigma[a] - f * b.sigma[a]) > 1e-3
                            for a in b.sigma):
                        bad.append(f"{v} {sc} {tau}")
                elif v == "q300":
                    if (i.capacity != 300 or [x[:3] for x in jobs]
                            != [x[:3] for x in jobs_b]
                            or st.mean(x[3] for x in jobs)
                            <= st.mean(x[3] for x in jobs_b)):
                        bad.append(f"q300 {sc} {tau}")
                elif v == "size":
                    if not all(100 <= x[0] <= 350 for x in jobs):
                        bad.append(f"size {sc} {tau}")
                elif v == "c5":
                    if len({x[2] for x in jobs}) != 5:
                        bad.append(f"c5 {sc} {tau}")
                elif v == "skew":           # shares 1 : 1/2 : 1/3, pooled
                    skew.update(x[2] for x in jobs)
                elif v == "cluster":
                    if ([x[:3] for x in jobs] != [x[:3] for x in jobs_b]
                            or len(set(x[3] for x in jobs)) < 2):
                        bad.append(f"cluster {sc} {tau}")
            p0, p0b, p1 = perturbed(b, 0), perturbed(b, 0), perturbed(b, 1)
            if (p0.proc != p0b.proc or p0.proc == p1.proc or not all(
                    0.8 <= p0.proc[a] / b.proc[a] <= 1.2
                    and 0.8 <= p0.sigma[a] / b.sigma[a] <= 1.2
                    for a in b.proc)):
                bad.append(f"perturb draws {sc} {tau}")
    tot = sum(skew.values())
    if not (skew["C1"] > skew["C2"] > skew["C3"]
            and abs(skew["C1"] / tot - 6 / 11) < 0.06):
        bad.append(f"skew shares {dict(skew)}")
    check("v9a sensitivity variants change only their factor", not bad,
          f"{bad}" if bad else "")
    clk = [round(clock_for(n) * 0.47, 1) for n in (200, 300)]
    check("v9a clocks of 200 and 300 jobs (176.1, 264.1 s)",
          clk == [176.1, 264.1] and len(cells("x200")) == 3
          and len(cells("x300")) == 3, f"{clk}")
    defaults = {k: v[0] for k, v in cp.LEVELS.items()}
    ok = cp.current() == defaults
    for m in cp.variant_names("H")[1:]:
        _, k, v = cp.split(m)
        with cp.override(m):
            ok = ok and cp.current()[k] == v
        ok = ok and cp.current() == defaults
    from cpm import cpm_report
    inst = inst_of("small", 0, 0.5, 2)
    _, d4 = cpm_report(inst, eval_budget=400)
    with cp.override("H~outer_passes=1"):
        _, d1 = cpm_report(inst, eval_budget=400)
    check("v9a CPM parameter overrides take effect and are restored",
          ok and d1["evaluations"] != d4["evaluations"],
          f"evaluations {d4['evaluations']} -> {d1['evaluations']}")
    frozen = load_frozen("")
    r = run_task("small", 0, 0.5, 4, "CPM+ALNS+R:union@0.1", 4, 1.5,
                 frozen, 1.0, "perturb")
    ok = (r["feasible"] and r["departures_consistent"]
          and r["variant"] == "perturb" and len(r["perturbed"]) == 5
          and all(len(x) == len(r["points"]) for x in r["perturbed"]))
    r2 = run_task("small", 0, 0.5, 4, "CPM+ALNS+R:union@0.1~stall_levels=3",
                  4, 1.5, frozen, 1.0)
    ok2 = (r2["feasible"] and r2["params"].get("cpm_stall_levels") == 3
           and cp.current() == defaults)
    check("v9a perturb record and CPM-parameter record complete",
          bool(ok and ok2))


def e2_check():
    import itertools
    import random as _r
    from core import Instance, evaluate_schedule
    from e2 import (bracket, e2_instances, exact_contiguous, run_task,
                    true_matrix)
    from instances import FAMILY_POOL
    from params import load_frozen
    from variants import cells, make_instance
    fams = [n for n, _, _ in FAMILY_POOL]
    sig = {n: round(s, 4) for n, _, s in FAMILY_POOL}
    M = true_matrix(fams, sig, 1.0)
    ok = (all(M[(f, f)] == 0 for f in fams)
          and abs(M[("Brown", "Ecru")] - round(1.8 * sig["Brown"] * (1 + 4 / 7),
                                                4)) < 1e-9
          and abs(M[("Ecru", "Brown")] - sig["Ecru"]) < 1e-9
          and abs(M[("Black", "Ecru")] - round(sig["Black"] * 2, 4)) < 1e-9)
    check("E2 matrix follows Eq. (21) of [7]", ok)
    base = make_instance(cells("small")[0], 0.5, 4)
    A3 = {(f, g): (0.0 if f == g else base.sigma[f])
          for f in base.families for g in base.families}
    twin = Instance(jobs=base.jobs, machines=base.machines,
                    capacity=base.capacity, proc=dict(base.proc),
                    sigma=dict(base.sigma), name="twin", setup_matrix=A3)
    from hybrid import bound_point
    from cpm import cpm_report
    front, _ = cpm_report(base, eval_budget=200)
    same = all(evaluate_schedule(base, p.batches, p.schedule).objectives()
               == evaluate_schedule(twin, p.batches, p.schedule).objectives()
               for p in front)
    check("E2 an A3 matrix evaluates as A3", same, f"{len(front)} schedules")
    rng = _r.Random(7)
    bad = 0
    for _ in range(10):
        k, m = rng.choice([3, 4, 5]), rng.choice([1, 2, 3])
        fs = sorted(rng.sample(fams, k))
        Mt = true_matrix(fs, sig, rng.choice([1.0, 2.0]))
        best = float("inf")
        for assign in itertools.product(range(m), repeat=k):
            tot = 0.0
            for h in range(m):
                g = [f for f, a_ in zip(fs, assign) if a_ == h]
                if len(g) > 1:
                    tot += min(sum(Mt[(p[i], p[i + 1])]
                                   for i in range(len(p) - 1))
                               for p in itertools.permutations(g))
            best = min(best, tot)
        bad += abs(exact_contiguous(fs, Mt, m) - best) > 1e-6
    check("E2 Held-Karp equals brute force (10 cases)", bad == 0)
    bad, closed = 0, 0
    for sc in ("medium", "xlarge"):
        for c in cells(sc):
            for lam in (1.0, 2.0):
                t, p, _, mi = e2_instances(make_instance(c, 0.5, 4), lam)
                br = bracket(t, p, mi)
                bad += not (br["LB"] <= br["contig"] + 1e-9
                            and br["contig"] <= br["UB"] + 1e-9
                            and br["UB"] <= br["LB_plus"] + 1e-9)
                closed += br["UB"] - br["LB"] < 1e-9
    check("E2 bracket LB <= contig <= UB <= LB(sigma+), closes",
          bad == 0 and closed == 12, f"closed on {closed} of 12")
    frozen = load_frozen("")
    good = True
    for m in ("H-max", "H-mean", "H-mix", "ALNS+R", "[7]+R"):
        if m == "[7]+R":
            try:
                import pyomo  # noqa: F401
            except ImportError:
                continue
        r = run_task("small", 0, 0.5, 4, 2.0, m, 4, 2.0, frozen)
        good = good and r["feasible"] and bool(r["points"])
        if m == "H-max":
            good = good and r["guarantee_ok"] and r["optimistic_z1"] == 0
        if m == "[7]+R":
            good = good and r["comp7"]["rejected"] == 0
    check("E2 methods feasible, sigma+ never optimistic, [7] agrees", good)


if __name__ == "__main__":
    golden()
    bound()
    z2_guarantee()
    records()
    adapt_check()
    comp7_check()
    jsonl_check()
    v9a_check()
    e2_check()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
