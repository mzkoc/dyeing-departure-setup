"""
Every number in the paper, rebuilt from the raw outputs.

Nothing in the manuscript should be typed in by hand. This script reads
results.csv / fronts.jsonl (from one or several runner output directories),
the CP-SAT verification file and the wall-clock file, and writes one CSV per
table plus a plain-text report.

    python analysis.py --results results/part_a results/part_b \
                       results/part_c results/part_d \
                       --cpsat results/cpsat_verification.csv \
                       --walltime results/walltime.csv \
                       --out tables/

Indicators. The runner stores raw hypervolume and a Z2-scaled IGD, which is
what the main tables report. This script additionally recomputes, from the
stored fronts, a NORMALISED hypervolume and IGD+ (Ishibuchi et al., 2015):
each instance's objectives are scaled to [0, 1] by the ideal and nadir of the
union of all methods' fronts, the reference point is (1.1, 1.1), and IGD+ is
measured against the best-known front of that instance. Both are
scale-free, so they can be averaged across instances without the largest
instances dominating the mean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

METHODS = ["CPM", "CPM-blind", "NSGA-II", "ALNS", "LPT+EDD"]
SCALES = ["verify", "small", "medium", "large", "xlarge"]
LABEL = {"verify": "12-20", "small": "30", "medium": "60",
         "large": "100", "xlarge": "150"}
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}  # Demsar 2006

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_results(dirs: Sequence[str]) -> Tuple[pd.DataFrame,
                                               Dict[Tuple[str, str], List]]:
    frames, fronts = [], {}
    for d in dirs:
        frames.append(pd.read_csv(os.path.join(d, "results.csv")))
        with open(os.path.join(d, "fronts.jsonl")) as fh:
            for line in fh:
                r = json.loads(line)
                fronts[(r["instance"], r["method"])] = r["front"]
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(["instance", "method"], keep="last")
    if dup.any():
        print(f"note: {dup.sum()} duplicate rows dropped (resumed runs)")
    df = df[~dup].copy()
    df["regime"] = np.sign(df.excess).map({-1: "e<0", 0: "e=0", 1: "e>0"})
    return df, fronts


def check_complete(df: pd.DataFrame) -> None:
    per = df.groupby("instance").method.nunique()
    bad = per[per != len(METHODS)]
    if len(bad):
        raise SystemExit(f"{len(bad)} instances lack some method: "
                         f"{list(bad.index[:5])}")
    if not df.feasible.all():
        raise SystemExit("infeasible rows present")
    exp = {"verify": 270, "small": 90, "medium": 90, "large": 90,
           "xlarge": 90}
    got = df.drop_duplicates("instance").scale.value_counts().to_dict()
    for s, n in exp.items():
        if got.get(s, 0) != n:
            print(f"warning: scale {s} has {got.get(s, 0)} instances, "
                  f"expected {n}")


# --------------------------------------------------------------------------
# Indicators recomputed from fronts
# --------------------------------------------------------------------------


def _dominates(p: Point, q: Point) -> bool:
    return (p[0] <= q[0] + 1e-9 and p[1] <= q[1] + 1e-9
            and (p[0] < q[0] - 1e-9 or p[1] < q[1] - 1e-9))


def _nondominated(pts: Sequence[Point]) -> List[Point]:
    out = []
    for p in pts:
        if any(_dominates(q, p) for q in pts):
            continue
        if any(abs(q[0] - p[0]) < 1e-9 and abs(q[1] - p[1]) < 1e-9
               for q in out):
            continue
        out.append(p)
    return out


def _hv2d(pts: Sequence[Point], ref: Point) -> float:
    h, prev = 0.0, ref[1]
    for a, b in sorted(pts):
        if a >= ref[0] or b >= prev:
            continue
        h += (ref[0] - a) * (prev - b)
        prev = b
    return h


def _igd_plus(front: Sequence[Point], reference: Sequence[Point]) -> float:
    tot = 0.0
    for r in reference:
        tot += min(math.hypot(max(p[0] - r[0], 0.0), max(p[1] - r[1], 0.0))
                   for p in front)
    return tot / len(reference)


def coverage(A: Sequence[Point], B: Sequence[Point]) -> float:
    """C(A, B): share of B dominated by some point of A."""
    return sum(any(_dominates(a, b) for a in A) for b in B) / len(B)


def normalised_indicators(df: pd.DataFrame, fronts) -> pd.DataFrame:
    rows = []
    for inst in df.instance.unique():
        fr = {m: [tuple(p) for p in fronts[(inst, m)]] for m in METHODS}
        allp = [p for v in fr.values() for p in v]
        lo = [min(p[k] for p in allp) for k in (0, 1)]
        hi = [max(p[k] for p in allp) for k in (0, 1)]
        rng = [(h - l) if h - l > 1e-12 else 1.0 for l, h in zip(lo, hi)]

        def nz(p):
            return ((p[0] - lo[0]) / rng[0], (p[1] - lo[1]) / rng[1])

        ref_front = [nz(p) for p in _nondominated(allp)]
        for m in METHODS:
            pts = [nz(p) for p in fr[m]]
            rows.append({"instance": inst, "method": m,
                         "hv_norm": _hv2d(pts, (1.1, 1.1)),
                         "igd_plus": _igd_plus(pts, ref_front)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table_main(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scale", "method"]).agg(
        HV=("hypervolume", "mean"), IGD=("igd", "mean"),
        HVn=("hv_norm", "mean"), IGDplus=("igd_plus", "mean"),
        Z1min=("z1_min", "mean"), Z2min=("z2_min", "mean"),
        water_m3=("cleaning_water_m3", "mean"),
        Z2opt_pct=("z2_optimal", "mean"),
        front=("front_size", "mean"),
        cpu_s=("seconds", "mean"), evals=("evaluations", "mean"))
    g["Z2opt_pct"] *= 100
    return g.reindex(pd.MultiIndex.from_product([SCALES, METHODS])).round(3)


def table_regime(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["scale", "regime", "method"]).agg(
        Z1min=("z1_min", "mean"), Z2min=("z2_min", "mean"),
        Z2opt_pct=("z2_optimal", "mean"), front=("front_size", "mean"),
        cpu_s=("seconds", "mean"), evals=("evaluations", "mean"))
    g["Z2opt_pct"] *= 100
    return g.round(3)


def best_counts(df: pd.DataFrame, col: str = "igd",
                lower_is_better: bool = True) -> pd.DataFrame:
    p = df.pivot_table(index=["instance", "scale", "regime"],
                       columns="method", values=col).reset_index()
    best = p[METHODS].min(axis=1) if lower_is_better \
        else p[METHODS].max(axis=1)
    p["cpm_best"] = (p.CPM <= best + 1e-9) if lower_is_better \
        else (p.CPM >= best - 1e-9)
    return p.groupby(["scale", "regime"]).cpm_best.agg(["sum", "count"])


def friedman_nemenyi(df: pd.DataFrame, col: str,
                     lower_is_better: bool = True) -> Tuple[pd.DataFrame,
                                                             pd.DataFrame]:
    rows, pairs = [], []
    k = len(METHODS)
    for s in SCALES:
        p = df[df.scale == s].pivot(index="instance", columns="method",
                                    values=col)[METHODS]
        x = p if lower_is_better else -p
        ranks = x.rank(axis=1, method="average").mean()
        stat = friedmanchisquare(*[x[m] for m in METHODS])
        n = len(p)
        cd = Q_ALPHA_05[k] * math.sqrt(k * (k + 1) / (6.0 * n))
        best = (x.CPM <= x.min(axis=1) + 1e-9).sum()
        rows.append({"scale": s, "N": n, "p": f"{stat.pvalue:.1e}", "CD": cd,
                     **{f"rank_{m}": ranks[m] for m in METHODS},
                     "cpm_best": int(best)})
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                d = abs(ranks[a] - ranks[b])
                pairs.append({"scale": s, "a": a, "b": b,
                              "rank_diff": round(d, 3),
                              "significant": d > cd})
    return pd.DataFrame(rows).round(4), pd.DataFrame(pairs)


def table_coverage(df: pd.DataFrame, fronts) -> pd.DataFrame:
    sc = df.drop_duplicates("instance").set_index("instance").scale
    rows = []
    for inst, s in sc.items():
        A = fronts[(inst, "CPM")]
        for m in METHODS[1:]:
            B = fronts[(inst, m)]
            rows.append({"scale": s, "method": m,
                         "C(CPM,B)": coverage(A, B),
                         "C(B,CPM)": coverage(B, A)})
    return pd.DataFrame(rows).groupby(["scale", "method"]).mean().round(3)


def table_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """CPM against CPM-blind. Gains are reported three ways because they
    differ: ratio of means, mean of per-instance ratios, and the median of
    per-instance ratios. The paper should state which one it uses."""
    rows = []
    for s in SCALES:
        d = df[df.scale == s]
        a = d[d.method == "CPM"].set_index("instance")
        b = d[d.method == "CPM-blind"].set_index("instance").loc[a.index]
        z1g = 100 * (b.z1_min - a.z1_min) / b.z1_min.replace(0, np.nan)
        hvg = 100 * (a.hv_norm - b.hv_norm) / b.hv_norm.replace(0, np.nan)
        wins = int((a.z1_min < b.z1_min - 1e-9).sum())
        ties = int((abs(a.z1_min - b.z1_min) < 1e-9).sum())
        try:
            p = wilcoxon(a.z1_min, b.z1_min).pvalue
        except ValueError:
            p = float("nan")
        rows.append({
            "scale": s, "N": len(a),
            "Z1_gain_ratio_of_means": 100 * (b.z1_min.mean() - a.z1_min.mean())
            / b.z1_min.mean(),
            "Z1_gain_mean_pct": z1g.mean(), "Z1_gain_median_pct": z1g.median(),
            "HVn_gain_mean_pct": hvg.mean(),
            "HVn_gain_median_pct": hvg.median(),
            "wins": wins, "ties": ties, "wilcoxon_p": f"{p:.1e}",
            "evals_CPM": a.evaluations.mean(),
            "evals_blind": b.evaluations.mean(),
            "cpu_CPM": a.seconds.mean(), "cpu_blind": b.seconds.mean()})
    return pd.DataFrame(rows).round(4)


def table_water(df: pd.DataFrame) -> pd.DataFrame:
    w = df.pivot_table(index="scale", columns="method",
                       values="cleaning_water_m3", aggfunc="mean")
    out = pd.DataFrame(index=SCALES)
    for m in METHODS[1:]:
        out[f"reduction_vs_{m}_pct"] = 100 * (1 - w.CPM / w[m])
    return out.round(1)


# --------------------------------------------------------------------------
# CP-SAT distance (Section 7.6), recomputed against the current CPM run
# --------------------------------------------------------------------------


def table_cpsat(df: pd.DataFrame, path: str) -> Dict[str, object]:
    c = pd.read_csv(path)
    cur = df[df.method == "CPM"].set_index("instance").z1_min
    c["cpm_z1_now"] = c.instance.map(cur)
    c["dist_pct"] = 100 * (c.cpm_z1_now - c.cpsat_z1_min) / c.cpsat_z1_min
    prov = c[c.cpsat_z1_optimal]
    rest = c[~c.cpsat_z1_optimal]
    return {
        "instances": len(c),
        "replications": sorted(c.replication.unique().tolist()),
        "p1_confirmed": int(c.p1_confirmed.sum()),
        "proven_by_n": prov.groupby("n").size().to_dict(),
        "proven_median_pct": prov.dist_pct.median(),
        "proven_mean_pct": prov.dist_pct.mean(),
        "proven_exact": int((prov.dist_pct.abs() < 1e-6).sum()),
        "rest_median_pct": rest.dist_pct.median(),
        "cpm_better_than_cpsat": int((c.dist_pct < -1e-6).sum()),
        "median_by_machines": c.groupby("machines").dist_pct.median()
        .round(2).to_dict(),
        "max_by_machines": c.groupby("machines").dist_pct.max()
        .round(2).to_dict(),
        "median_by_excess": c.groupby("excess").dist_pct.median()
        .round(2).to_dict(),
        "median_by_tau": c.groupby("tau").dist_pct.median()
        .round(2).to_dict(),
        "cpsat_mean_s": c.cpsat_seconds.mean(),
    }


# --------------------------------------------------------------------------
# Wall-clock check
# --------------------------------------------------------------------------


def table_walltime(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    w = pd.read_csv(path)
    w["regime"] = np.sign(w.excess).map({-1: "e<0", 0: "e=0", 1: "e>0"})
    g = w.groupby(["scale", "method"]).agg(
        clock=("seconds_budget", "first"), HV=("hypervolume", "mean"),
        IGD=("igd", "mean"), Z1min=("z1_min", "mean"),
        Z2min=("z2_min", "mean"), Z2opt_pct=("z2_optimal", "mean"),
        cpu_s=("seconds", "mean"), cpu_max=("seconds", "max"),
        evals=("evaluations", "mean"))
    g["Z2opt_pct"] *= 100
    rows = []
    for s in [x for x in SCALES if x in set(w.scale)]:
        p = w[w.scale == s].pivot(index="instance", columns="method",
                                  values="igd")[METHODS]
        ranks = p.rank(axis=1).mean()
        n = len(p)
        cd = Q_ALPHA_05[5] * math.sqrt(5 * 6 / (6.0 * n))
        rows.append({"scale": s, "N": n,
                     "p": f"{friedmanchisquare(*[p[m] for m in METHODS]).pvalue:.1e}",
                     "CD": cd, "cpm_best": int((p.CPM <= p.min(axis=1)
                                                + 1e-9).sum()),
                     **{f"rank_{m}": ranks[m] for m in METHODS}})
    a = w[w.method == "CPM"].set_index("instance")
    b = w[w.method == "CPM-blind"].set_index("instance").loc[a.index]
    same = ((a.z1_min - b.z1_min).abs() < 1e-9) & \
           ((a.hypervolume - b.hypervolume).abs() < 1e-6)
    over = w[w.seconds > 1.05 * w.seconds_budget][
        ["instance", "method", "seconds", "seconds_budget", "evaluations"]]
    diag = {"cpm_equals_blind": same.groupby(a.scale).sum().to_dict(),
            "overruns": over.to_dict("records")}
    return g.round(3), pd.DataFrame(rows).round(4), diag


# --------------------------------------------------------------------------


def small_scale_breakdown(df: pd.DataFrame, col: str = "igd_plus"):
    """Section 7.4: where the proposed method is weakest. On the 12-30 job
    instances, its share of best IGD+ by machine count, and which method
    wins when it does not."""
    p = df.pivot_table(index=["instance", "scale"], columns="method",
                       values=col).reset_index()
    p = p[p.scale.isin(["verify", "small"])]
    p["cpm_best"] = p.CPM <= p[METHODS].min(axis=1) + 1e-9
    m = df.drop_duplicates("instance").set_index("instance").machines
    p["machines"] = p.instance.map(m)
    by_m = p.groupby("machines").cpm_best.agg(["sum", "count", "mean"])
    lost = p[~p.cpm_best]
    winners = lost[METHODS].idxmin(axis=1).value_counts()
    return by_m, winners


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results"])
    ap.add_argument("--cpsat", default=None)
    ap.add_argument("--walltime", default=None)
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    df, fronts = load_results(a.results)
    check_complete(df)
    ind = normalised_indicators(df, fronts)
    df = df.merge(ind, on=["instance", "method"])

    report: List[str] = []

    def emit(title: str, obj) -> None:
        text = obj.to_string() if hasattr(obj, "to_string") else \
            json.dumps(obj, indent=2, default=str)
        report.append(f"\n=== {title} ===\n{text}")
        print(report[-1])

    t = table_main(df); t.to_csv(f"{a.out}/table6_main.csv")
    emit("Table 6 - main comparison", t)
    t = table_regime(df); t.to_csv(f"{a.out}/by_regime.csv")
    emit("By colour-excess regime", t)
    for col, low in (("igd", True), ("igd_plus", True), ("hv_norm", False)):
        fr, pairs = friedman_nemenyi(df, col, low)
        fr.to_csv(f"{a.out}/friedman_{col}.csv", index=False)
        pairs.to_csv(f"{a.out}/nemenyi_{col}.csv", index=False)
        emit(f"Friedman / Nemenyi on {col}", fr)
        emit(f"Nemenyi pairs on {col} (significant only)",
             pairs[pairs.significant])
    t = best_counts(df); t.to_csv(f"{a.out}/best_igd_by_regime.csv")
    emit("CPM best IGD by regime", t)
    t = best_counts(df, "igd_plus")
    t.to_csv(f"{a.out}/best_igd_plus_by_regime.csv")
    emit("CPM best IGD+ by regime (Section 7.4, Fig. 4b)", t)
    by_m, winners = small_scale_breakdown(df)
    by_m.to_csv(f"{a.out}/small_scale_by_machines.csv")
    emit("12-30 jobs: CPM best IGD+ by machine count (Section 7.4)", by_m)
    emit("12-30 jobs: winner where CPM is not best on IGD+", winners)
    t = table_coverage(df, fronts); t.to_csv(f"{a.out}/table8_coverage.csv")
    emit("Table 8 - coverage", t)
    t = table_ablation(df); t.to_csv(f"{a.out}/table9_ablation.csv",
                                     index=False)
    emit("Table 9 - ablation", t)
    t = table_water(df); t.to_csv(f"{a.out}/water_reduction.csv")
    emit("Cleaning-water reduction (CPM vs each)", t)
    if a.cpsat:
        emit("Section 7.7 - CP-SAT distance", table_cpsat(df, a.cpsat))
    if a.walltime:
        g, fr, diag = table_walltime(a.walltime)
        g.to_csv(f"{a.out}/walltime_summary.csv")
        fr.to_csv(f"{a.out}/walltime_friedman.csv", index=False)
        emit("Wall-clock check", g)
        emit("Wall-clock Friedman on IGD", fr)
        emit("Wall-clock diagnostics", diag)

    with open(f"{a.out}/report.txt", "w") as fh:
        fh.write("\n".join(report))
    print(f"\n-> {a.out}/")


if __name__ == "__main__":
    main()
