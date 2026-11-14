"""Summarise fair_compare.py output.

Indicators follow analysis.py: per instance, objectives are scaled by the
ideal and nadir of the union of every run of the compared methods (all seeds),
HV is taken against (1.1, 1.1) and IGD+ against the union's non-dominated
set. The union is the methods in this file, not the five of Table 6, so
absolute values differ from the paper; comparisons between methods do not
depend on that.

Per instance, a stochastic method's value is the mean over its seeds; paired
tests (Wilcoxon) and Friedman ranks use those means. Within a scale, the
p-values of the listed pairwise tests are Holm-adjusted per indicator (p_H).
Effect sizes are paired, because instances differ far more than methods do:
PS is the paired probability of superiority, (wins + ties/2) / n, the paired
analogue of Vargha-Delaney's A12 (PS > 0.5 favours the first method), and
med the median paired percentage difference in Z1min.
With more than one seed, the seed-to-seed spread is reported as well.

--methods restricts both the tables and the normalisation / IGD+ reference to
the listed methods (ANALYSIS_PLAN.md: the main table uses the six main
methods; ablations are computed separately, ablation.py).

--pairs first tests the first listed method against each of the others and
adjusts over exactly those comparisons (the confirmatory analysis of
ANALYSIS_PLAN.md, section 4: H against the five other main methods). The
default, --pairs auto, tests the pairs of PAIRS that are present and adjusts
over all of them (exploratory).

    python analyze_fair.py results_fair
    python analyze_fair.py results_valid --pairs first --methods CPM+ALNS+R:union@0.05,ALNS+R,NSGA-II+R,CPM,ALNS,NSGA-II
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from analysis import _hv2d, _igd_plus, _nondominated, coverage

ORDER = ["CPM", "CPM+R", "CPM-blind", "ALNS", "ALNS+R", "NSGA-II", "NSGA-II+R",
         "CPM+ALNS+R", "FFD+SA-EDDTS", "FFD+SA-EDDTS+R", "FFD-EDDTS"]
PAIRS = [("CPM+ALNS+R", "ALNS+R"), ("CPM+ALNS+R", "CPM"),
         ("CPM+ALNS+R", "NSGA-II+R"), ("CPM", "ALNS+R"), ("CPM", "NSGA-II+R"), ("CPM+R", "ALNS+R"),
         ("CPM", "ALNS"), ("CPM", "NSGA-II"), ("ALNS+R", "ALNS"),
         ("NSGA-II+R", "NSGA-II"), ("ALNS+R", "NSGA-II+R"),
         ("CPM-blind", "ALNS"), ("CPM", "CPM-blind"), ("CPM+R", "CPM"),
         ("CPM", "FFD+SA-EDDTS"), ("ALNS+R", "FFD+SA-EDDTS"),
         ("FFD-EDDTS", "FFD+SA-EDDTS"),
         ("CPM", "FFD+SA-EDDTS+R"), ("ALNS+R", "FFD+SA-EDDTS+R"),
         ("FFD+SA-EDDTS+R", "FFD+SA-EDDTS")]
COMP7 = ("FFD+SA-EDDTS", "FFD-EDDTS", "FFD+SA-EDDTS+R")


Z2_GRID = 12      # setups are multiples of 1/12 h (5 min) in instances.py


def snap(r, grid=Z2_GRID):
    """Put every Z2 of a record on the grid of 1/grid h and drop the points
    this makes dominated. Setup times are stored to four decimals (0.3333,
    0.4167, ...), so equal setup levels summed in a different order differ in
    the fourth decimal and count as two non-dominated points; snapping merges
    them. grid = 0 leaves the record unchanged."""
    if not grid:
        return r

    def g(z):
        return round(z * grid) / grid

    def pts(ps):
        return [list(p) for p in _nondominated([(a, g(b)) for a, b in ps])]
    r["points"] = pts(r["points"])
    r["z2_bound"] = g(r["z2_bound"])
    if "parts" in r:
        r["parts"]["alns"] = pts(r["parts"]["alns"])
        r["parts"]["cpm"] = pts(r["parts"]["cpm"])
        b = r["parts"]["bound"]
        r["parts"]["bound"] = [b[0], g(b[1])]
        # origin of the snapped points: found by the CPM part only, by the
        # ALNS+R part only, or by both
        cpm = {tuple(p) for p in r["parts"]["cpm"]}
        aln = {tuple(p) for p in r["parts"]["alns"]}
        r["origin"] = [("both" if tuple(p) in aln else "cpm")
                       if tuple(p) in cpm else "alns" for p in r["points"]]
    return r


def load(d, grid=Z2_GRID):
    from jsonl_io import read_records
    rows = []
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        for r in read_records(p):
            r.setdefault("seed", r["replication"])
            rows.append(snap(r, grid))
    return rows


def runs_table(rows, order=None):
    """One row per run, indicators against the per-instance union. `order`
    (the --methods list) fixes the order of the methods."""
    by = {}
    for r in rows:
        by.setdefault(r["instance"], []).append(r)
    common = None
    for rs in by.values():
        ms = {r["method"] for r in rs}
        common = ms if common is None else common & ms
    common = common or set()
    if order:
        methods = [m for m in order if m in common]
    else:
        others = sorted(m for m in common if m not in ORDER)
        methods = [m for m in ORDER if m in common] + others
    out, fronts = [], {}
    for inst, rs in by.items():
        rs = [r for r in rs if r["method"] in methods and r["points"]]
        allp = [tuple(p) for r in rs for p in r["points"]]
        lo = [min(p[k] for p in allp) for k in (0, 1)]
        hi = [max(p[k] for p in allp) for k in (0, 1)]
        rng = [(h - l) if h - l > 1e-12 else 1.0 for l, h in zip(lo, hi)]

        def nz(p):
            return ((p[0] - lo[0]) / rng[0], (p[1] - lo[1]) / rng[1])
        ref = [nz(p) for p in _nondominated(allp)]
        for r in rs:
            pts = [tuple(p) for p in r["points"]]
            fronts[(inst, r["method"], r["seed"])] = pts
            out.append({
                "instance": inst, "scale": r["scale"], "m": r["machines"],
                "tau": r["tau"], "replication": r["replication"],
                "excess": r["excess"], "method": r["method"],
                "seed": r["seed"], "z1_min": min(p[0] for p in pts),
                "z2_min": min(p[1] for p in pts),
                "z2_opt": float(min(p[1] for p in pts)
                                <= r["z2_bound"] + 1e-6),
                "front": len(pts),
                "hv_norm": _hv2d([nz(p) for p in pts], (1.1, 1.1)),
                "igd_plus": _igd_plus([nz(p) for p in pts], ref),
                "seconds": r["seconds"], "clock": r["clock"],
                "evals": r["evaluations"], "feasible": r["feasible"]})
    return pd.DataFrame(out), fronts, methods


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def paired(mean, a, b, col):
    x, y = mean.loc[a, col], mean.loc[b, col]
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx], y.loc[idx]
    better = int((x < y - 1e-9).sum())
    worse = int((x > y + 1e-9).sum())
    p = (wilcoxon(x, y).pvalue if (abs(x - y) > 1e-9).any() else 1.0)
    denom = y.abs().where(y.abs() > 1e-12)
    med = float((100 * (x - y) / denom).median()) if denom.notna().any() \
        else float("nan")
    return better, len(idx) - better - worse, worse, p, med


def report(scale, runs, fronts, methods, pair_mode="auto"):
    n = runs.instance.nunique()
    seeds = (runs[runs.method.isin(methods)]
             .groupby(["method", "instance"]).seed.nunique()
             .groupby("method").max())
    print(f"\n===== {scale}: {n} instances, clock {runs.clock.iloc[0]} s, "
          f"seeds per stochastic method: "
          f"{int(seeds.max()) if len(seeds) else 0} =====")
    cols = ["hv_norm", "igd_plus", "z1_min", "z2_min", "z2_opt", "front",
            "evals", "seconds"]
    mean = runs.groupby(["method", "instance"])[cols].mean()
    print(mean.groupby("method").mean().loc[methods].round(3).to_string())
    print("runs with a dropped infeasible point:", int((~runs.feasible).sum()))

    piv = mean["igd_plus"].unstack(0)[methods]
    print("mean IGD+ rank:", piv.rank(axis=1).mean().round(2).to_dict())
    if len(methods) >= 3 and n >= 3:
        print("Friedman IGD+ p =",
              f"{friedmanchisquare(*[piv[m] for m in methods]).pvalue:.2g}")
    hybrids = [m for m in methods if m.startswith("CPM+ALNS+R")]
    if pair_mode == "first":
        todo = [(methods[0], o) for o in methods[1:]]
    else:
        pairs = list(PAIRS)
        for h in hybrids:
            for o in ("ALNS+R", "CPM", "NSGA-II+R") + COMP7:
                if (h, o) not in pairs:
                    pairs.insert(0, (h, o))
        todo = [q for q in pairs if set(q) <= set(methods)]
    res = {q: (paired(mean, *q, "z1_min"), paired(mean, *q, "igd_plus"),
               paired(mean, *q, "hv_norm")) for q in todo}
    adj = {}
    for k in range(3):
        ph = holm([res[q][k][3] for q in todo])
        for q, p in zip(todo, ph):
            adj[(q, k)] = p
    def ps(w, t, l):
        return (w + 0.5 * t) / max(w + t + l, 1)
    print(f"pairwise ({len(todo)} comparisons, Holm over these per indicator):"
          " wins/ties/losses of the first method; p_H Holm-adjusted;"
          " PS paired probability of superiority (>0.5 favours the first);"
          " med median % difference in Z1min")
    for q in todo:
        z, i, h = res[q]
        print(f"{q[0]:>10} vs {q[1]:<10} "
              f"Z1min {z[0]}/{z[1]}/{z[2]} p_H={adj[(q, 0)]:.2g} "
              f"PS={ps(z[0], z[1], z[2]):.2f} med={z[4]:+.2f}% | "
              f"IGD+ {i[0]}/{i[1]}/{i[2]} p_H={adj[(q, 1)]:.2g} "
              f"PS={ps(i[0], i[1], i[2]):.2f} | "
              f"HV {h[2]}/{h[1]}/{h[0]} p_H={adj[(q, 2)]:.2g} "
              f"PS={ps(h[2], h[1], h[0]):.2f}")

    for x in [m for m in ["ALNS+R", "NSGA-II+R"] + hybrids
              if m in methods]:
        if "CPM" not in methods:
            break
        cA, cB = [], []
        for inst in runs.instance.unique():
            cpm = fronts.get((inst, "CPM", runs[(runs.instance == inst) & (
                runs.method == "CPM")].seed.iloc[0]))
            for s in runs[(runs.instance == inst) & (runs.method == x)].seed:
                fx = fronts[(inst, x, s)]
                cA.append(coverage(cpm, fx))
                cB.append(coverage(fx, cpm))
        print(f"coverage C(CPM, {x}) = {np.mean(cA):.2f}   "
              f"C({x}, CPM) = {np.mean(cB):.2f}")

    for xm in [m for m in ["ALNS+R"] + hybrids if m in methods]:
        if "CPM" not in methods:
            break
        x = mean.loc[xm, "z1_min"]
        y = mean.loc["CPM", "z1_min"].loc[x.index]
        gap = 100 * (x - y) / y
        print(f"{xm} Z1min relative to CPM: median {gap.median():+.2f}%  "
              f"mean {gap.mean():+.2f}%  range {gap.min():+.1f} to "
              f"{gap.max():+.1f}%")
    if "CPM" in methods and "ALNS+R" in methods:
        show = [m for m in ["CPM", "ALNS+R"] + hybrids if m in methods]
        mm = runs[runs.method.isin(show)].pivot_table(
            index="m", columns="method", values="z1_min", aggfunc="mean")
        print("Z1min by number of machines:\n" + mm.round(1).to_string())

    stoch = [m for m in methods if m in ("ALNS", "ALNS+R", "NSGA-II",
                                         "NSGA-II+R") + COMP7 or m in hybrids]
    if len(seeds) and seeds.max() > 1 and stoch:
        print("seed-to-seed spread (mean over instances of the per-instance "
              "std; CV in %):")
        for m in stoch:
            g = runs[runs.method == m].groupby("instance")
            sd = g.z1_min.std().mean()
            cv = (100 * g.z1_min.std() / g.z1_min.mean()).mean()
            print(f"  {m:<10} Z1min std {sd:8.2f}  CV {cv:5.2f}%   "
                  f"HVn std {g.hv_norm.std().mean():.3f}   "
                  f"IGD+ std {g.igd_plus.std().mean():.3f}")
    return mean.reset_index().assign(scale=scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="*", default=["results_fair"],
                    help="one or more result folders, analysed together; "
                         "the summaries are written to the first")
    ap.add_argument("--methods", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", choices=("auto", "first"), default="auto")
    ap.add_argument("--clock-mult", type=float, default=None,
                    help="only runs with this clock multiplier (records "
                         "without one count as 1.0); default: all")
    ap.add_argument("--z2-grid", type=int, default=Z2_GRID,
                    help="snap Z2 to 1/k h before the dominance filter "
                         "(0 = off; use 0 for perturbed setup times)")
    a = ap.parse_args()
    d = a.folder[0]
    rows = [r for f in a.folder for r in load(f, a.z2_grid)]
    if a.clock_mult is not None:
        rows = [r for r in rows
                if abs(r.get("clock_mult", 1.0) - a.clock_mult) < 1e-9]
    keep = []
    if a.methods:
        keep = [m for m in a.methods.split(",") if m]
        rows = [r for r in rows if r["method"] in keep]
    if a.pairs == "first" and not keep:
        raise SystemExit("--pairs first needs --methods (first = the method "
                         "tested against the others)")
    per_run, per_inst = [], []
    for scale in ("verify", "small", "medium", "large", "xlarge"):
        sub = [r for r in rows if r["scale"] == scale]
        if not sub:
            continue
        runs, fronts, methods = runs_table(sub, keep or None)
        if a.pairs == "first" and keep and methods[:1] != keep[:1]:
            print(f"\n{scale}: {keep[0]} has no runs here; skipped")
            continue
        if runs.empty or len(methods) < 2:
            continue
        per_run.append(runs)
        per_inst.append(report(scale, runs, fronts, methods, a.pairs))
    if per_run:
        sfx = f"_{a.tag}" if a.tag else ""
        f1 = os.path.join(d, f"summary_runs{sfx}.csv")
        f2 = os.path.join(d, f"summary_instances{sfx}.csv")
        pd.concat(per_run).to_csv(f1, index=False)
        pd.concat(per_inst).to_csv(f2, index=False)
        print(f"\nwritten: {f1}, {f2}")


if __name__ == "__main__":
    main()
