"""Claims C1-C7 of ANALYSIS_PLAN.md (section 5) and the secondary analyses
(section 6) from the validation runs, with every decision rule applied as
written there.

    python confirm.py results_valid --cpsatgap results_cpsatgap > confirm_valid.txt
    python confirm.py results_valid results_comp7 --curves-only \
        --methods "H,CPM,ALNS+R,FFD+SA-EDDTS,FFD-EDDTS"   (H = the hybrid's name)

H is the frozen hybrid (frozen.json, key "hybrid", or --hybrid).

C1, C2   H against the five other main methods; indicators as in
         analyze_fair.py (normalisation and IGD+ reference from the union of
         the six main methods); Wilcoxon, Holm over the five comparisons per
         scale and indicator; PS = (wins + ties/2) / n.
C3, C4   ablations, each normalised by the union of the two fronts compared:
         H against H_A+B (H's own ALNS+R part plus B) and against ALNS+R+B
         (the ALNS+R benchmark run plus B); B is the Corollary 3 schedule as
         logged in H's record (after CPM-A). Holm over the five scales.
C5       attainment of the Z2 bound.
C6       verification scale, replications 0-4: H's least tardiness against
         CP-SAT (results/cpsat_verification.csv); replications 0-3 from
         --cpsatgap, replication 4 from the validation folder.
C7       single-machine instances: front size and least-tardiness gap of H
         against CPM.
Secondary: setup-budget curves; origin of H's points.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis import _hv2d, _igd_plus, _nondominated
from analyze_fair import Z2_GRID, holm, load, paired, runs_table

MAIN = ["ALNS+R", "NSGA-II+R", "CPM", "ALNS", "NSGA-II"]
SCALES = ["verify", "small", "medium", "large", "xlarge"]
ALPHA, PS_MIN, C2_MED = 0.05, 0.60, 1.0
C3_SCALES = ("medium", "large", "xlarge")      # 60, 100, 150 jobs


def ps(b, t, w):
    return (b + 0.5 * t) / max(b + t + w, 1)


def per_scale(rows, h):
    out = {}
    for sc in SCALES:
        sub = [r for r in rows if r["scale"] == sc
               and r["method"] in [h] + MAIN]
        if not sub:
            continue
        runs, fronts, methods = runs_table(sub, [h] + MAIN)
        if methods != [h] + MAIN:
            print(f"{sc}: methods present {methods}; scale skipped")
            continue
        out[sc] = (runs, fronts)
    return out


def c1_c2(tabs, h):
    print("\n=== C1 front quality (IGD+) and C2 least tardiness against ALNS+R")
    print("    C1: p_H < 0.05 and PS >= 0.60; C2: not significantly worse "
          "and median difference <= +1.0%")
    claims = {}
    for sc, (runs, _) in tabs.items():
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "hv_norm", "z1_min"]].mean()
        res = {x: {k: paired(mean, h, x, k) for k in
                   ("igd_plus", "hv_norm", "z1_min")} for x in MAIN}
        adj = {k: dict(zip(MAIN, holm([res[x][k][3] for x in MAIN])))
               for k in ("igd_plus", "hv_norm", "z1_min")}
        print(f"\n{sc} ({runs.instance.nunique()} instances)")
        for x in MAIN:
            b, t, w, _, _ = res[x]["igd_plus"]
            p = adj["igd_plus"][x]
            c1 = p < ALPHA and ps(b, t, w) >= PS_MIN
            claims[(sc, x)] = c1
            hb, ht, hw, _, _ = res[x]["hv_norm"]      # 'better' = lower HV
            zb, zt, zw, _, zmed = res[x]["z1_min"]
            print(f"  vs {x:<10} IGD+ {b}/{t}/{w} p_H={p:.2g} "
                  f"PS={ps(b, t, w):.2f} -> C1 {'YES' if c1 else 'no'}  | "
                  f"HV {hw}/{ht}/{hb} p_H={adj['hv_norm'][x]:.2g} "
                  f"PS={ps(hw, ht, hb):.2f} | Z1min {zb}/{zt}/{zw} "
                  f"p_H={adj['z1_min'][x]:.2g} med={zmed:+.2f}%")
        zb, zt, zw, _, zmed = res["ALNS+R"]["z1_min"]
        worse = adj["z1_min"]["ALNS+R"] < ALPHA and zmed > 0
        c2 = (not worse) and zmed <= C2_MED
        print(f"  C2 (H vs ALNS+R, Z1min): significantly worse: "
              f"{'yes' if worse else 'no'}; median {zmed:+.2f}% -> "
              f"{'H matches ALNS+R' if c2 else 'loss reported'}")
    print("\nC1 'from k jobs upward' needs C1 at every larger scale:")
    for x in MAIN:
        ok = [sc for sc in tabs if claims.get((sc, x))]
        upward = None
        for sc in reversed(list(tabs)):
            if claims.get((sc, x)):
                upward = sc
            else:
                break
        print(f"  vs {x:<10} holds at {ok}; from {upward or '-'} upward")


def _ind(front, other):
    """HV and IGD+ of `front`, normalised by the union of both fronts."""
    allp = list(front) + list(other)
    lo = [min(p[k] for p in allp) for k in (0, 1)]
    hi = [max(p[k] for p in allp) for k in (0, 1)]
    rng = [(b - a) if b - a > 1e-12 else 1.0 for a, b in zip(lo, hi)]

    def nz(p):
        return ((p[0] - lo[0]) / rng[0], (p[1] - lo[1]) / rng[1])
    ref = [nz(p) for p in _nondominated(allp)]
    return (_hv2d([nz(p) for p in front], (1.1, 1.1)),
            _igd_plus([nz(p) for p in front], ref))


def _paired_test(x, y, higher_better):
    x, y = np.asarray(x), np.asarray(y)
    d = x - y if higher_better else y - x       # > 0: H better
    b, w = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    t = len(d) - b - w
    p = wilcoxon(x, y).pvalue if (np.abs(d) > 1e-9).any() else 1.0
    return b, t, w, p


def c3_c4(rows, h):
    print("\n=== C3 (H vs H_A+B, HV; kept if better at >= 2 of 60/100/150 "
          "jobs) and C4 (H vs ALNS+R+B)")
    alnsr = {(r["instance"], r["seed"]): r for r in rows
             if r["method"] == "ALNS+R"}
    alnsr_any = {}
    for r in rows:
        if r["method"] == "ALNS+R":
            alnsr_any.setdefault(r["instance"], r)
    data = {sc: {"c3hv": [], "c4hv": [], "c4igd": []} for sc in SCALES}
    for r in rows:
        if r["method"] != h or "parts" not in r:
            continue
        H = [tuple(p) for p in r["points"]]
        B = tuple(r["parts"]["bound"])
        ab = _nondominated([tuple(p) for p in r["parts"]["alns"]] + [B])
        hv_h, _ = _ind(H, ab)
        hv_ab, _ = _ind(ab, H)
        data[r["scale"]]["c3hv"].append((r["instance"], hv_h, hv_ab))
        a = alnsr.get((r["instance"], r["seed"])) or \
            alnsr_any.get(r["instance"])
        if a is not None:
            arb = _nondominated([tuple(p) for p in a["points"]] + [B])
            hv1, ig1 = _ind(H, arb)
            hv2, ig2 = _ind(arb, H)
            data[r["scale"]]["c4hv"].append((r["instance"], hv1, hv2))
            data[r["scale"]]["c4igd"].append((r["instance"], ig1, ig2))

    def by_instance(lst):
        df = pd.DataFrame(lst, columns=["instance", "h", "o"])
        return df.groupby("instance").mean()

    for key, label, hb in (("c3hv", "C3 H vs H_A+B   HV", True),
                           ("c4hv", "C4 H vs ALNS+R+B HV", True),
                           ("c4igd", "C4 H vs ALNS+R+B IGD+", False)):
        scs = [sc for sc in SCALES if data[sc][key]]
        tests = {}
        for sc in scs:
            df = by_instance(data[sc][key])
            tests[sc] = (_paired_test(df.h, df.o, hb),
                         df.h.mean(), df.o.mean())
        adj = dict(zip(scs, holm([tests[sc][0][3] for sc in scs])))
        print(f"\n{label} (Holm over {len(scs)} scales)")
        better = []
        for sc in scs:
            (b, t, w, _), mh, mo = tests[sc]
            sig_b = adj[sc] < ALPHA and ps(b, t, w) >= PS_MIN
            sig_w = adj[sc] < ALPHA and ps(b, t, w) < 0.5
            if sig_b:
                better.append(sc)
            print(f"  {sc:<7} H better/tie/worse {b}/{t}/{w}  p_H={adj[sc]:.2g}"
                  f"  PS={ps(b, t, w):.2f}  mean {mh:.4f} vs {mo:.4f}  "
                  f"{'H better' if sig_b else ('H WORSE' if sig_w else '')}")
        if key == "c3hv":
            n = sum(sc in better for sc in C3_SCALES)
            print(f"  -> C3: better at {n} of the 60/100/150-job scales: "
                  f"{'level traversal KEPT' if n >= 2 else 'claim REMOVED'}")


def c5(tabs):
    print("\n=== C5 attainment of the Z2 bound (share of runs)")
    rows = []
    for sc, (runs, _) in tabs.items():
        rows.append(runs.groupby("method").z2_opt.mean().rename(sc))
    df = pd.concat(rows, axis=1)
    print((100 * df).round(1).to_string())


def c6(rows, gap_rows, h, csv):
    print("\n=== C6 verification scale, replications 0-4: least tardiness "
          "against CP-SAT")
    if not os.path.exists(csv):
        print(f"  {csv} not found; skipped")
        return
    ref = pd.read_csv(csv).set_index("instance")
    pts = {}
    for r in list(gap_rows) + [r for r in rows if r["scale"] == "verify"
                               and r["replication"] == 4]:
        if r["scale"] != "verify" or not r["points"]:
            continue
        pts.setdefault((r["method"], r["instance"]), []).append(
            min(p[0] for p in r["points"]))
    for m in [h, "ALNS+R", "CPM"]:
        z = pd.Series({i: float(np.mean(v)) for (mm, i), v in pts.items()
                       if mm == m})
        z = z[z.index.isin(ref.index)]
        if z.empty:
            continue
        cp = ref.loc[z.index, "cpsat_z1_min"]
        opt = ref.loc[z.index, "cpsat_z1_optimal"].astype(bool)
        pos = cp > 1e-9
        gap = 100 * (z[pos] - cp[pos]) / cp[pos]
        hit = (z - cp).abs() <= 1e-6 * np.maximum(1.0, cp.abs())
        print(f"  {m:<24} {len(z)} instances ({int(opt.sum())} proven "
              f"optimal): attains CP-SAT value on {int(hit.sum())}; gap on "
              f"{int(pos.sum())} with positive value: median "
              f"{gap.median():.2f}%, max {gap.max():.2f}%; proven-optimal "
              f"only: median {gap[opt[pos]].median():.2f}%; below CP-SAT "
              f"(not proven) on {int(((z < cp - 1e-6) & ~opt).sum())}")
    if "cpm_distance_pct" in ref.columns:
        po = ref.cpsat_z1_optimal.astype(bool)
        print(f"  submitted paper, CPM distance to CP-SAT: median "
              f"{ref.cpm_distance_pct.median():.1f}% over all {len(ref)}; "
              f"{ref.cpm_distance_pct[po].median():.1f}% over the "
              f"{int(po.sum())} proven optimal (the 11.5% of the paper)")


def c7(tabs, h):
    print("\n=== C7 single-machine instances: H against CPM")
    runs, _ = tabs.get("verify", (None, None))
    if runs is None:
        print("  no verification-scale runs")
        return
    one = runs[runs.m == 1]
    if one.empty:
        print("  no single-machine instances")
        return
    g = one.groupby(["method", "instance"])[["front", "z1_min"]].mean()
    hh, cc = g.loc[h], g.loc["CPM"]
    gap = 100 * (hh.z1_min - cc.z1_min) / cc.z1_min.where(cc.z1_min > 0)
    print(f"  {len(hh)} instances; front size H {hh.front.mean():.2f} "
          f"(min {hh.front.min():.0f}) vs CPM {cc.front.mean():.2f}; "
          f"Z1min H vs CPM median {gap.median():+.2f}%, "
          f"range {gap.min():+.1f} to {gap.max():+.1f}%")


def budget_curves(rows, methods):
    print("\n=== Secondary: setup-budget curves. L_k = Z2* + 0.25 k h. "
          "Per k: median % excess of each method's least Z1 within L_k over "
          "the best method's, and in brackets the % of instances with no "
          "point within L_k")
    for sc in SCALES:
        sub = [r for r in rows if r["scale"] == sc and r["method"] in methods]
        if not sub:
            continue
        inst = {}
        for r in sub:
            inst.setdefault(r["instance"], {}).setdefault(
                r["method"], []).append((r["points"], r["z2_bound"]))
        print(f"\n{sc}")
        print("  k  " + "".join(f"{m[:16]:>22}" for m in methods))
        for k in range(9):
            cells = {m: [] for m in methods}
            for i, d in inst.items():
                best = {}
                for m in methods:
                    vals = []
                    for pts, zb in d.get(m, []):
                        lim = zb + 0.25 * k + 1e-9
                        z = [p[0] for p in pts if p[1] <= lim]
                        vals.append(min(z) if z else np.nan)
                    best[m] = np.nanmean(vals) if vals and not all(
                        np.isnan(vals)) else np.nan
                ref = np.nanmin(list(best.values())) if not all(
                    np.isnan(list(best.values()))) else np.nan
                for m in methods:
                    if np.isnan(best[m]):
                        cells[m].append(np.nan)
                    elif ref > 1e-9:
                        cells[m].append(100 * (best[m] - ref) / ref)
                    else:
                        cells[m].append(0.0 if best[m] <= 1e-9 else np.inf)
            line = f"  {k}  "
            for m in methods:
                v = np.array(cells[m], dtype=float)
                miss = 100 * np.isnan(v).mean()
                med = np.nanmedian(v) if (~np.isnan(v)).any() else np.nan
                line += f"{med:>13.2f}% [{miss:4.0f}%]"
            print(line)


def origin(rows, h):
    print("\n=== Secondary: origin of H's front points (share of points)")
    for sc in SCALES:
        o = [x for r in rows if r["scale"] == sc and r["method"] == h
             for x in r.get("origin", [])]
        if o:
            s = pd.Series(o).value_counts(normalize=True).round(3)
            print(f"  {sc:<7} {len(o)} points: {s.to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="+")
    ap.add_argument("--curves-only", action="store_true",
                    help="only the Z2-attainment table and the setup-budget "
                         "curves, for the methods in --methods")
    ap.add_argument("--methods", default="")
    ap.add_argument("--z2-grid", type=int, default=Z2_GRID)
    ap.add_argument("--cpsatgap", default="")
    ap.add_argument("--hybrid", default="")
    ap.add_argument("--frozen", default="frozen.json")
    ap.add_argument("--cpsat-csv",
                    default=os.path.join("results", "cpsat_verification.csv"))
    a = ap.parse_args()
    h = a.hybrid or json.load(open(a.frozen))["hybrid"]
    print(f"H = {h}")
    rows = [r for f in a.folder for r in load(f, a.z2_grid)]
    if a.curves_only:
        ms = [m for m in a.methods.split(",") if m] or [h] + MAIN
        sub = [r for r in rows if r["method"] in ms]
        z = pd.DataFrame([{"scale": r["scale"], "method": r["method"],
                           "z2_opt": min(p[1] for p in r["points"])
                           <= r["z2_bound"] + 1e-6} for r in sub
                          if r["points"]])
        print("\nattainment of the Z2 bound (% of runs)")
        print((100 * z.pivot_table(index="method", columns="scale",
                                   values="z2_opt", aggfunc="mean"))
              .round(1).reindex(columns=[c for c in SCALES
                                         if c in set(z.scale)]).to_string())
        budget_curves(sub, ms)
        return
    tabs = per_scale(rows, h)
    c1_c2(tabs, h)
    c3_c4(rows, h)
    c5(tabs)
    c6(rows, load(a.cpsatgap, a.z2_grid) if a.cpsatgap else [], h,
       a.cpsat_csv)
    c7(tabs, h)
    budget_curves(rows, [h] + MAIN)
    origin(rows, h)


if __name__ == "__main__":
    main()
