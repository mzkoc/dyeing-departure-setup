"""Results by setup regime, e = |F| - m < 0, = 0, > 0 (descriptive).

For e <= 0 a zero-setup schedule exists; the companion study [7] holds
|F| > m throughout, so e > 0 is the regime its method was designed for.

    python regime.py results_valid results_comp7 --methods "H,FFD+SA-EDDTS,ALNS+R,CPM" --ref H
    python regime.py results_valid --c4          (H against ALNS+R+B by regime)

Per scale and regime, for each method against --ref: IGD+ and HV wins/ties/
losses of the reference with the paired probability of superiority PS and an
unadjusted Wilcoxon p (descriptive), median paired % difference of the least
tardiness (reference minus method, relative to the method), share of runs
attaining the Z2 bound and mean least setup. Normalisation and IGD+ reference
per instance from the union of the listed methods, as in analyze_fair.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from analysis import _nondominated
from analyze_fair import Z2_GRID, load, runs_table
from confirm import _ind

SCALES = ["verify", "small", "medium", "large", "xlarge"]


def regime(e):
    return "e<0" if e < 0 else ("e=0" if e == 0 else "e>0")


def _test(x, y, lower_better=True):
    d = (y - x) if lower_better else (x - y)          # > 0: x better
    b, w = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    t = len(d) - b - w
    p = wilcoxon(x, y).pvalue if (np.abs(d) > 1e-9).any() else 1.0
    return b, t, w, (b + 0.5 * t) / max(len(d), 1), p


def by_regime(rows, methods, ref):
    print(f"reference: {ref}; per regime: IGD+ W/T/L of the reference, PS, "
          f"p (unadjusted) | HV likewise | Z1min median % (ref vs method) | "
          f"Z2 bound attained % | mean least setup h")
    for sc in SCALES:
        sub = [r for r in rows if r["scale"] == sc and r["method"] in methods]
        if not sub:
            continue
        runs, _, ms = runs_table(sub, methods)
        if ref not in ms:
            continue
        runs["regime"] = runs.excess.map(regime)
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "hv_norm", "z1_min", "z2_min", "z2_opt"]].mean()
        reg = runs.groupby("instance").regime.first()
        print(f"\n{sc}")
        for g in ("e<0", "e=0", "e>0"):
            ins = reg[reg == g].index
            if len(ins) == 0:
                continue
            r0 = mean.loc[ref].loc[ins]
            print(f"  {g} ({len(ins)} instances)  {ref[:24]}: Z2 bound "
                  f"{100 * r0.z2_opt.mean():.0f}%, least setup "
                  f"{r0.z2_min.mean():.2f} h")
            for m in ms:
                if m == ref:
                    continue
                o = mean.loc[m].loc[ins]
                i = _test(r0.igd_plus.values, o.igd_plus.values, True)
                h = _test(r0.hv_norm.values, o.hv_norm.values, False)
                den = o.z1_min.where(o.z1_min.abs() > 1e-12)
                med = float((100 * (r0.z1_min - o.z1_min) / den).median())
                print(f"     vs {m[:22]:<22} IGD+ {i[0]}/{i[1]}/{i[2]} "
                      f"PS={i[3]:.2f} p={i[4]:.2g} | HV {h[0]}/{h[1]}/{h[2]} "
                      f"PS={h[3]:.2f} | Z1min {med:+.1f}% | {m[:12]} Z2 bound "
                      f"{100 * o.z2_opt.mean():.0f}%, least setup "
                      f"{o.z2_min.mean():.2f} h")


def c4_by_regime(rows, h):
    print(f"C4 by regime: {h} against ALNS+R+B (B as logged in H's record); "
          f"each pair normalised by the union of its two fronts; HV and IGD+ "
          f"W/T/L of H, PS, unadjusted Wilcoxon p. C3 (descriptive): share of "
          f"instances where the ladder adds HV to H's own ALNS+R part plus B "
          f"(it cannot remove any), and the mean gain.")
    alnsr = {r["instance"]: r for r in rows if r["method"] == "ALNS+R"}
    for sc in SCALES:
        data = {}
        for r in rows:
            if r["method"] != h or r["scale"] != sc or "parts" not in r:
                continue
            a = alnsr.get(r["instance"])
            if a is None:
                continue
            H = [tuple(p) for p in r["points"]]
            B = tuple(r["parts"]["bound"])
            ab = _nondominated([tuple(p) for p in r["parts"]["alns"]] + [B])
            arb = _nondominated([tuple(p) for p in a["points"]] + [B])
            c3 = _ind(H, ab)[0] - _ind(ab, H)[0]
            (h1, i1), (h2, i2) = _ind(H, arb), _ind(arb, H)
            data.setdefault(regime(r["excess"]), []).append(
                (c3, h1, h2, i1, i2))
        if not data:
            continue
        print(f"\n{sc}")
        for g in ("e<0", "e=0", "e>0"):
            v = np.array(data.get(g, []))
            if not len(v):
                continue
            hv = _test(v[:, 1], v[:, 2], False)
            ig = _test(v[:, 3], v[:, 4], True)
            print(f"  {g} ({len(v)}): C4 HV {hv[0]}/{hv[1]}/{hv[2]} "
                  f"PS={hv[3]:.2f} p={hv[4]:.2g} | IGD+ {ig[0]}/{ig[1]}/{ig[2]} "
                  f"PS={ig[3]:.2f} p={ig[4]:.2g} | C3 adds on "
                  f"{int((v[:, 0] > 1e-9).sum())}/{len(v)}, mean "
                  f"+{v[:, 0].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="+")
    ap.add_argument("--methods", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--frozen", default="frozen.json")
    ap.add_argument("--z2-grid", type=int, default=Z2_GRID)
    a = ap.parse_args()
    rows = [r for f in a.folder for r in load(f, a.z2_grid)]
    h = json.load(open(a.frozen))["hybrid"]
    if a.c4:
        c4_by_regime(rows, h)
        return
    ms = [m.replace("H", h) if m == "H" else m
          for m in a.methods.split(",") if m]
    ref = h if a.ref in ("", "H") else a.ref
    by_regime(rows, ms, ref)


if __name__ == "__main__":
    main()
