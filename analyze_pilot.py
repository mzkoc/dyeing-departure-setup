"""Summarise the equal wall-clock pilot.

Indicators use the definitions of analysis.py (per-instance normalisation by
the ideal and nadir of the union of the compared methods' points, HV against
(1.1, 1.1), IGD+ against the union's non-dominated set). The union here is
the four pilot methods only, so absolute values are not comparable with
Table 6 of the paper, where LPT+EDD stretches the nadir; rankings are.

    python analyze_pilot.py pilot
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from analysis import _dominates, _hv2d, _igd_plus, _nondominated, coverage

METHODS = ["CPM", "CPM-blind", "ALNS", "ALNS+R"]


def load(d):
    rows = []
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    return rows


def per_instance(rows):
    global METHODS
    present = {r["method"] for r in rows}
    METHODS = [m for m in ["CPM", "CPM+R", "CPM-blind", "ALNS", "ALNS+R"]
               if m in present]
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["method"]] = r
    out = []
    fronts = {}
    for inst, ms in by.items():
        if not set(METHODS) <= set(ms):
            continue
        allp = [tuple(p) for m in METHODS for p in ms[m]["points"]]
        lo = [min(p[k] for p in allp) for k in (0, 1)]
        hi = [max(p[k] for p in allp) for k in (0, 1)]
        rng = [(h - l) if h - l > 1e-12 else 1.0 for l, h in zip(lo, hi)]

        def nz(p):
            return ((p[0] - lo[0]) / rng[0], (p[1] - lo[1]) / rng[1])
        ref = [nz(p) for p in _nondominated(allp)]
        for m in METHODS:
            r = ms[m]
            pts = [tuple(p) for p in r["points"]]
            fronts[(inst, m)] = pts
            out.append({
                "instance": inst, "scale": r["scale"], "m": r["machines"],
                "tau": r["tau"], "excess": r["excess"], "method": m,
                "z1_min": min(p[0] for p in pts),
                "z2_min": min(p[1] for p in pts),
                "z2_opt": min(p[1] for p in pts) <= r["z2_bound"] + 1e-6,
                "front": len(pts),
                "hv_norm": _hv2d([nz(p) for p in pts], (1.1, 1.1)),
                "igd_plus": _igd_plus([nz(p) for p in pts], ref),
                "seconds": r["seconds"], "clock": r["clock"],
                "evals": r["evaluations"], "feasible": r["feasible"]})
    return pd.DataFrame(out), fronts


def pair(df, a, b, col, lower=True):
    x = df[df.method == a].set_index("instance")[col]
    y = df[df.method == b].set_index("instance")[col].loc[x.index]
    better = (x < y - 1e-9).sum() if lower else (x > y + 1e-9).sum()
    worse = (x > y + 1e-9).sum() if lower else (x < y - 1e-9).sum()
    ties = len(x) - better - worse
    d = (x - y).values
    p = wilcoxon(x, y).pvalue if np.any(np.abs(d) > 1e-9) else 1.0
    return better, ties, worse, p


def main(d):
    rows = load(d)
    frames = []
    for sc in ("small", "medium", "large", "xlarge"):
        sub = [r for r in rows if r["scale"] == sc]
        if not sub:
            continue
        g, fronts = per_instance(sub)
        if g.empty:
            continue
        frames.append(g)
        report(sc, g, fronts)
    pd.concat(frames).to_csv(os.path.join(d, "pilot_summary.csv"),
                             index=False)


def report(sc, g, fronts):
    if True:
        n = g.instance.nunique()
        print(f"\n===== {sc}: {n} instances, clock {g.clock.iloc[0]} s =====")
        t = g.groupby("method")[["hv_norm", "igd_plus", "z1_min", "z2_min",
                                 "z2_opt", "front", "evals", "seconds"]].mean()
        print(t.loc[METHODS].round(3).to_string())
        print("infeasible flags:", int((~g.feasible).sum()))
        ranks = g.pivot(index="instance", columns="method",
                        values="igd_plus")[METHODS].rank(axis=1)
        print("mean IGD+ rank:", ranks.mean().round(2).to_dict())
        if n >= 3 and len(METHODS) >= 3:
            piv = g.pivot(index="instance", columns="method",
                          values="igd_plus")[METHODS]
            print("Friedman IGD+ p =",
                  f"{friedmanchisquare(*[piv[m] for m in METHODS]).pvalue:.2g}")
        pairs = [("CPM", "ALNS+R"), ("CPM", "ALNS"), ("ALNS+R", "ALNS"),
                 ("CPM-blind", "ALNS"), ("CPM", "CPM-blind"),
                 ("CPM+R", "ALNS+R"), ("CPM+R", "CPM")]
        for a, b in [q for q in pairs if set(q) <= set(METHODS)]:
            bz = pair(g, a, b, "z1_min")
            bi = pair(g, a, b, "igd_plus")
            print(f"{a:>9} vs {b:<7}  Z1min better/tie/worse {bz[0]}/{bz[1]}/"
                  f"{bz[2]} (p={bz[3]:.2g})   IGD+ {bi[0]}/{bi[1]}/{bi[2]} "
                  f"(p={bi[3]:.2g})")
        cA = np.mean([coverage(fronts[(i, 'CPM')], fronts[(i, 'ALNS+R')])
                      for i in g.instance.unique()])
        cB = np.mean([coverage(fronts[(i, 'ALNS+R')], fronts[(i, 'CPM')])
                      for i in g.instance.unique()])
        print(f"coverage C(CPM, ALNS+R) = {cA:.2f}   C(ALNS+R, CPM) = {cB:.2f}")
        # Z1 gap of ALNS+R to CPM, in percent of CPM
        x = g[g.method == "ALNS+R"].set_index("instance").z1_min
        y = g[g.method == "CPM"].set_index("instance").z1_min.loc[x.index]
        gap = 100 * (x - y) / y
        print(f"ALNS+R Z1min vs CPM: median {gap.median():+.2f}%  mean "
              f"{gap.mean():+.2f}%  range {gap.min():+.1f}..{gap.max():+.1f}%")
        by_m = g[g.method.isin(["CPM", "ALNS+R"])].pivot_table(
            index="m", columns="method", values="z1_min", aggfunc="mean")
        print("Z1min by machines:\n", by_m.round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
