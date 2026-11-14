"""Analysis of E2 (ANALYSIS_PLAN.md, section 9, amendments 3 and 5).

Each record of e2.py becomes one or two fronts: H-max and H-max+D (and
H-mean, H-mean+D) from the same run, [7] (before R) and [7]+R from the same
run. All objective values are under the true matrix; no Z2 grid.

Per scale and lambda, normalised by the union of the main set
(H-max, H-max+D, H-mix, ALNS+R, [7]+R): H-max against each of the others,
IGD+ PS with Holm over the four, median least-tardiness difference (% of the
other method's, positive: H-max lower), and the share of instances on which
each method attains the true setup optimum (the bracket of Proposition 2
closes on every E2 instance, so Z2* = LB = UB). Then by regime, H-mean
against H-max at lambda = 1, the guarantee check and the +D statistics.

    python e2_analyze.py results_e2 [--by-regime]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from analyze_fair import holm, load, paired, runs_table

MAIN = ["H-max", "H-max+D", "H-mix", "ALNS+R", "[7]+R"]
SCALES = [("medium", "60"), ("xlarge", "150")]


def _nd(pts):
    """Non-dominated subset of (z1, z2) points."""
    out, best = [], float("inf")
    for z1, z2 in sorted(set(map(tuple, pts))):
        if z2 < best - 1e-9:
            out.append([z1, z2])
            best = z2
    return out


def expand(rows):
    """Fronts per record. H-max+B (exploratory, added after the run): the
    H-max front with the Corollary 3 schedule planned with sigma+ (B+, which
    the hybrid computes first) kept in it; its true objectives are in the
    record's bracket (UB_z1, UB)."""
    out = []
    for r in rows:
        base = dict(r)
        out.append(base)
        if r["method"] == "H-max":
            b = [r["bracket"]["UB_z1"], r["bracket"]["UB"]]
            out.append(dict(r, method="H-max+B", points=_nd(r["points"] + [b])))
        if "points_D" in r:
            out.append(dict(r, method=r["method"] + "+D", points=r["points_D"]))
        if r["method"] == "[7]+R" and "comp7" in r:
            out.append(dict(r, method="[7]", points=r["comp7"]["pre_r"]))
    for r in out:
        r["z2_bound"] = r["bracket"]["LB"]
    return out


def compare(sub, ms, ref="H-max"):
    runs, _, _ = runs_table(sub, ms)
    mean = runs.groupby(["method", "instance"])[
        ["igd_plus", "hv_norm", "z1_min", "z2_min", "z2_opt"]].mean()
    others = [m for m in ms if m != ref]
    res = {x: (paired(mean, ref, x, "igd_plus"),
               paired(mean, ref, x, "z1_min")) for x in others}
    pi = dict(zip(others, holm([res[x][0][3] for x in others])))
    pz = dict(zip(others, holm([res[x][1][3] for x in others])))
    agg = runs.groupby("method")[["hv_norm", "igd_plus"]].mean()
    lines = [f"    {ref}: Z2* attained {100 * mean.loc[ref].z2_opt.mean():.0f}%"
             f"  HV {agg.loc[ref, 'hv_norm']:.3f}  IGD+ "
             f"{agg.loc[ref, 'igd_plus']:.3f}"]
    for x in others:
        (w, t, l, _, _), (wz, tz, lz, _, med) = res[x]
        ps = (w + 0.5 * t) / max(w + t + l, 1)
        lines.append(
            f"    vs {x:<9} IGD+ W/T/L {w}/{t}/{l} PS {ps:.2f}"
            f"{'*' if pi[x] < 0.05 else ' '} (p {pi[x]:.3g})  "
            f"dZ1min {-med:+.1f}%{'*' if pz[x] < 0.05 else ' '}  "
            f"{x} attains Z2* {100 * mean.loc[x].z2_opt.mean():.0f}%, "
            f"least Z2 {mean.loc[x].z2_min.mean():.2f} h "
            f"(Z2* {mean.loc[ref].z2_min.mean():.2f})  HV "
            f"{agg.loc[x, 'hv_norm']:.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--by-regime", action="store_true")
    a = ap.parse_args()
    rows = expand([r for f in a.folders for r in load(f, 0)])
    print(f"{len(rows)} fronts")
    raw = [r for r in rows if r["method"] in ("H-max", "H-mean", "H-mix",
                                              "ALNS+R", "[7]+R")]
    print("feasible:", all(r["feasible"] for r in raw),
          "| H-max guarantee violated:",
          sum(1 for r in raw if r.get("guarantee_ok") is False),
          "| [7] schedules rejected:",
          sum(r["comp7"]["rejected"] for r in raw if "comp7" in r))
    br = Counter((r["bracket"]["UB"] - r["bracket"]["LB"]) < 1e-9
                 for r in raw)
    print(f"bracket closed (LB = UB) in {br[True]} of {sum(br.values())} "
          "records")
    over = [r["seconds"] / r["clock"] for r in raw]
    print(f"wall/clock: median {np.median(over):.3f} max {max(over):.3f}")
    for lam in sorted({r["lam"] for r in rows}):
        for sc, lab in SCALES:
            sub = [r for r in rows if r["scale"] == sc and r["lam"] == lam]
            if not sub:
                continue
            print(f"\n== {lab} jobs, lambda {lam:g}")
            print("\n".join(compare(sub, MAIN)))
            if a.by_regime:
                for g, name in ((1, "e>0"), (0, "e=0"), (-1, "e<0")):
                    part = [r for r in sub if np.sign(r["excess"]) == g]
                    if part:
                        print(f"  -- {name}")
                        print("\n".join(compare(part, MAIN)))
            seven = [r for r in sub if r["method"] in ("[7]", "[7]+R",
                                                       "H-max")]
            print("  [7] before and after R:")
            print("\n".join(compare(seven, ["H-max", "[7]", "[7]+R"])))
            if lam == 1.0:
                mm = [r for r in sub if r["method"] in
                      ("H-max", "H-max+D", "H-mean", "H-mean+D")]
                if any(r["method"] == "H-mean" for r in mm):
                    print("  planning value: sigma+ against sigma_mean")
                    print("\n".join(compare(mm, ["H-max", "H-max+D",
                                                 "H-mean", "H-mean+D"])))
            for m in ("H-max", "H-mean"):
                rr = [r for r in sub if r["method"] == m]
                if not rr:
                    continue
                print(f"  {m}: planned schedules with true Z1 > planned "
                      f"{sum(r['optimistic_z1'] for r in rr)}, true Z2 > "
                      f"planned {sum(r['optimistic_z2'] for r in rr)}; "
                      f"+D points on the front "
                      f"{sum(r['D_from_sorted'] for r in rr)}, machines "
                      f"whose last family changed "
                      f"{sum(r['d_last_changed'] for r in rr)}")


if __name__ == "__main__":
    sys.exit(main())
