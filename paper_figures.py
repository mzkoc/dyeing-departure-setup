"""Figures of the revised paper, drawn from the result folders.

    python paper_figures.py budget results_valid --out fig3_budget.pdf

budget: setup-budget view of the fronts (ANALYSIS_PLAN.md, section 6). For
each budget L_k = Z2* + 0.25 k h, the least tardiness of each method among its
schedules with Z2 <= L_k, relative to the best of the six main methods
(median over instances, among instances where the method has such a
schedule), and the share of instances on which it has none. Four methods are
drawn; ALNS and NSGA-II, 12-29% above the best, are given in the caption.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_fair import load

H = "CPM+ALNS+R:union@0.1"
SIX = [H, "ALNS+R", "NSGA-II+R", "CPM", "ALNS", "NSGA-II"]
DRAW = [  # method, label, colour, marker, line style
    (H, "CPM+ALNS+R (hybrid)", "#2a78d6", "o", "-"),
    ("ALNS+R", "ALNS+R", "#eb6834", "s", "-"),
    ("NSGA-II+R", "NSGA-II+R", "#1baf7a", "^", "-"),
    ("CPM", "CPM (ladder alone)", "#0b0b0b", "D", "--"),
]
SCALES = [("medium", "60 jobs"), ("large", "100 jobs"), ("xlarge", "150 jobs")]
K = range(9)


def curves(rows, scale):
    inst = {}
    for r in rows:
        if r["scale"] == scale and r["method"] in SIX:
            inst.setdefault(r["instance"], {})[r["method"]] = r
    excess = {m: [] for m in SIX}
    missing = {m: [] for m in SIX}
    for k in K:
        ex = {m: [] for m in SIX}
        miss = {m: 0 for m in SIX}
        for d in inst.values():
            best = {}
            for m in SIX:
                r = d[m]
                lim = r["z2_bound"] + 0.25 * k + 1e-9
                z = [p[0] for p in r["points"] if p[1] <= lim]
                best[m] = min(z) if z else np.nan
            ref = np.nanmin(list(best.values()))
            for m in SIX:
                if np.isnan(best[m]):
                    miss[m] += 1
                elif ref > 1e-9:
                    ex[m].append(100 * (best[m] - ref) / ref)
                else:
                    ex[m].append(0.0)
        for m in SIX:
            excess[m].append(np.median(ex[m]) if ex[m] else np.nan)
            missing[m].append(100 * miss[m] / len(inst))
    return excess, missing


def budget(folders, out):
    rows = [r for f in folders for r in load(f)]
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "font.family": "serif"})
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.9), sharex=True,
                           sharey="row")
    x = [0.25 * k for k in K]
    for j, (sc, title) in enumerate(SCALES):
        ex, miss = curves(rows, sc)
        for m, lab, col, mk, ls in DRAW:
            ax[0, j].plot(x, ex[m], color=col, marker=mk, ms=4, lw=1.4,
                          ls=ls, label=lab)
            ax[1, j].plot(x, miss[m], color=col, marker=mk, ms=4, lw=1.4,
                          ls=ls)
        ax[0, j].set_title(title, fontsize=9)
        ax[1, j].set_xlabel("setup budget above the optimum (h)")
        for a in ax[:, j]:
            a.grid(axis="y", lw=0.4, color="#d9d8d4")
            a.spines[["top", "right"]].set_visible(False)
            a.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax[0, 0].set_ylabel("tardiness above best (%)")
    ax[1, 0].set_ylabel("instances without\na schedule (%)")
    ax[1, 0].set_ylim(-3, 100)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    for sc, _ in SCALES:
        ex, miss = curves(rows, sc)
        print(sc, "ALNS/NSGA-II excess range:",
              f"{np.nanmin(ex['ALNS'] + ex['NSGA-II']):.1f}-"
              f"{np.nanmax(ex['ALNS'] + ex['NSGA-II']):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["budget"])
    ap.add_argument("folder", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    budget(a.folder, a.out)
