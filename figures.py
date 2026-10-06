"""
Publication figures.

Design rules, applied to every figure:

  * PDF output. Vector, so it survives any scaling the typesetter applies,
    and \\includegraphics takes it directly.
  * Colour AND marker AND linestyle carry the same information. Many readers
    print in greyscale and some cannot separate the hues; a figure that needs
    colour to be read is a figure half the audience cannot read.
  * Single-column width (3.4 in) by default. Figures are drawn at the size
    they will be printed, so 9 pt text stays 9 pt instead of being shrunk
    into illegibility by the journal's layout.
  * No chartjunk: no gridlines fighting the data, no boxes on three sides,
    no legend frame.

Usage:
    python figures.py --results results/part_a results/part_b \
        results/part_c results/part_d --out figures
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

COL = 3.4        # single-column width, inches
WIDE = 7.0       # double-column width

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito palette: distinguishable under the common colour vision
# deficiencies, and still separable when printed in greyscale.
STYLE = {
    "CPM":       dict(color="#000000", marker="o", ls="-",  zorder=5),
    "CPM-blind": dict(color="#0072B2", marker="s", ls="--", zorder=4),
    "NSGA-II":   dict(color="#D55E00", marker="^", ls="-.", zorder=3),
    "ALNS":      dict(color="#009E73", marker="v", ls=":",  zorder=2),
    "LPT+EDD":   dict(color="#999999", marker="D", ls=(0, (1, 1)), zorder=1),
}
METHODS = ["CPM", "CPM-blind", "NSGA-II", "ALNS", "LPT+EDD"]
SCALES = ["verify", "small", "medium", "large", "xlarge"]
SCALE_N = {"verify": "12–20", "small": "30", "medium": "60",
           "large": "100", "xlarge": "150"}


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300)
    plt.close(fig)
    return path



# --------------------------------------------------------------------------
# Figure 1 — what the problem looks like
# --------------------------------------------------------------------------

# Light to dark, matching the tone semantics of the colour families: the
# reader can see at a glance that darker shades run longer and cost more to
# wash out, which is the whole reason the second objective exists.
FAMILY_FILL = {
    "Ecru": "#f7f4e9", "Pastel": "#e8dcc0", "Light": "#cdb891",
    "Medium": "#a88b5e", "Brown": "#7d5f3a", "Dark": "#55402a",
    "Navy": "#2b3a55", "Black": "#1a1a1a",
}
FAMILY_TEXT = {
    "Ecru": "#333333", "Pastel": "#333333", "Light": "#333333",
    "Medium": "#ffffff", "Brown": "#ffffff", "Dark": "#ffffff",
    "Navy": "#ffffff", "Black": "#ffffff",
}


def fig_gantt(out_dir: str, label: str = "ve_n20_m3_f3_c2",
              tau: float = 0.5, rep: int = 0) -> str:
    """A real schedule, not a mock-up.

    Small enough to read, real enough that nobody has to wonder whether the
    picture is achievable. It carries four things the text would otherwise
    have to assert: batches group jobs, a machine runs one colour block at a
    time, a colour change costs a wash-out slice, and tardiness is measured
    per job against the batch that carries it.
    """
    from core import evaluate_schedule
    from cpm import cpm_report
    from instances import build_matrix, generate_instance

    cell = [c for c in build_matrix(include_verify=True)
            if c.label == label][0]
    inst = generate_instance(cell, tau, rep)
    front, _ = cpm_report(inst)
    pt = sorted(front, key=lambda q: q.z2)[len(front) // 2]
    ev = evaluate_schedule(inst, pt.batches, pt.schedule, strict=False)
    by_id = {b.id: b for b in pt.batches}

    fig, ax = plt.subplots(figsize=(WIDE, 2.5))
    bar_h = 0.52

    for m in sorted(pt.schedule):
        seq = pt.schedule[m]
        y = inst.machines - 1 - m
        prev = None
        for bid in seq:
            b = by_id[bid]
            s, e = ev.start[bid], ev.completion[bid]
            if prev is not None and prev != b.family:
                setup = inst.sigma[prev]
                ax.barh(y, setup, left=s - setup, height=bar_h,
                        color="#ffffff", edgecolor="#c0392b",
                        hatch="////", linewidth=0.8, zorder=3)
            ax.barh(y, e - s, left=s, height=bar_h,
                    color=FAMILY_FILL[b.family], edgecolor="black",
                    linewidth=0.5, zorder=2)
            if e - s > 1.2:
                ax.text((s + e) / 2, y, f"B{bid}", ha="center", va="center",
                        fontsize=6.5, color=FAMILY_TEXT[b.family], zorder=4)
            prev = b.family

    # One annotated tardy job, so the first objective is not just a symbol.
    worst = None
    for b in pt.batches:
        c = ev.completion.get(b.id)
        for j in b.jobs:
            t = c - inst.job(j).due
            if t > 0 and (worst is None or t > worst[0]):
                worst = (t, j, b, c)
    if worst:
        t, j, b, c = worst
        d = inst.job(j).due
        m = next(k for k, s in pt.schedule.items() if b.id in s)
        y = inst.machines - 1 - m
        ax.annotate("", xy=(c, y - 0.46), xytext=(d, y - 0.46),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b",
                                    lw=0.9, shrinkA=0, shrinkB=0))
        ax.plot([d, d], [y - 0.62, y - 0.30], color="#c0392b", lw=1.1)
        ax.text((c + d) / 2, y - 0.72, f"$T_j={t:.1f}$", ha="center",
                va="top", fontsize=7, color="#c0392b")

    ax.set_yticks(range(inst.machines))
    ax.set_yticklabels([f"$M_{{{inst.machines - k}}}$"
                        for k in range(inst.machines)])
    ax.set_xlabel("Time (h)")
    ax.set_xlim(-0.4, ev.makespan + 1.4)
    ax.set_ylim(-1.0, inst.machines - 0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fams = sorted({b.family for b in pt.batches},
                  key=lambda f: inst.proc[f])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=FAMILY_FILL[f],
                             ec="black", lw=0.5) for f in fams]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="#c0392b",
                                 hatch="////", lw=0.8))
    ax.legend(handles, fams + ["wash-out"], frameon=False, ncol=len(fams) + 1,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.3, columnspacing=1.2)
    return _save(fig, out_dir, "fig1_gantt")


# --------------------------------------------------------------------------
# Figure 2 — what P1 says
# --------------------------------------------------------------------------


def fig_p1(out_dir: str) -> str:
    """The theorem, drawn.

    Left: block order does not change Z2 - only which colour sits last does,
    so the machine's wash-out bill is fixed once its colour set is fixed.
    Right: putting a colour on one more machine costs exactly +sigma_f, which
    is where every step of the Pareto front comes from.

    Both panels use the same three colours and the same sigma values, so the
    arithmetic in the labels can be checked against the picture.
    """
    sig = {"A": 0.25, "B": 0.50, "C": 0.35}
    proc = {"A": 4.0, "B": 6.0, "C": 5.0}
    fill = {"A": "#e8dcc0", "B": "#1a1a1a", "C": "#a88b5e"}
    txt = {"A": "#333333", "B": "#ffffff", "C": "#ffffff"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDE, 2.3))
    bar_h = 0.5

    def draw(ax, y, order, counts=None):
        t, prev = 0.0, None
        for f in order:
            if prev is not None and prev != f:
                ax.barh(y, sig[prev], left=t, height=bar_h, color="white",
                        edgecolor="#c0392b", hatch="////", linewidth=0.8)
                t += sig[prev]
            n = (counts or {}).get(f, 1)
            ax.barh(y, proc[f] * n, left=t, height=bar_h, color=fill[f],
                    edgecolor="black", linewidth=0.5)
            ax.text(t + proc[f] * n / 2, y, f, ha="center", va="center",
                    fontsize=8, color=txt[f])
            t += proc[f] * n
            prev = f
        return t

    # ---- left: order is free, the last block is not ----------------------
    for k, order in enumerate([("A", "C", "B"), ("C", "A", "B"),
                               ("A", "B", "C")]):
        y = 2 - k
        draw(axL, y, order)
        z2 = sum(sig[f] for f in order[:-1])
        axL.text(16.8, y, f"$Z_2={z2:.2f}$", va="center", fontsize=7.5,
                 fontweight="bold" if z2 < 0.7 else "normal")
    axL.text(16.8, 3.05, "", fontsize=7)
    axL.set_yticks([2, 1, 0])
    axL.set_yticklabels(["A C B", "C A B", "A B C"], fontsize=7.5)
    axL.set_xlim(-0.3, 21.5)
    axL.set_ylim(-0.9, 2.9)
    axL.set_xlabel("Time (h)")
    axL.set_title("Only the last block matters", fontsize=8.5, pad=14)
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="y", length=0)
    axL.annotate("same colour set,\nsame $Z_2$ whenever\nthe last block is B",
                 xy=(0.02, -0.55), xycoords="axes fraction", fontsize=7,
                 color="#444444", va="top")

    # ---- right: splitting a colour costs exactly +sigma_f -----------------
    # The receiving machine must already end on a later block. A colour moved
    # onto an otherwise-empty machine becomes its own last block and costs
    # NOTHING - which is the parenthetical in Corollary 4 and the reason an
    # earlier draft of this figure was wrong.
    draw(axR, 2.0, ("A", "C", "B"), {"A": 2})
    draw(axR, 1.2, ("C", "B"))
    axR.text(25.6, 1.6, "$Z_2=0.95$", va="center", fontsize=7.5)

    draw(axR, 0.1, ("A", "C", "B"))
    draw(axR, -0.7, ("A", "C", "B"))
    axR.text(25.6, -0.3, "$Z_2=1.20$", va="center", fontsize=7.5)

    axR.annotate("", xy=(24.9, -0.3), xytext=(24.9, 1.6),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
    axR.text(24.5, 0.65, "$+\\sigma_A$", ha="right", va="center",
             fontsize=7.5, color="#c0392b")

    axR.set_yticks([2.0, 1.2, 0.1, -0.7])
    axR.set_yticklabels(["$M_1$", "$M_2$", "$M_1$", "$M_2$"], fontsize=7.5)
    axR.set_xlim(-0.3, 31.0)
    axR.set_ylim(-1.7, 3.0)
    axR.set_xlabel("Time (h)")
    axR.set_title("Splitting a colour costs $+\\sigma_f$", fontsize=8.5,
                  pad=14)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    axR.annotate("one A batch moves to $M_2$, which already\nends on B, so A is paid for twice",
                 xy=(0.02, -0.55), xycoords="axes fraction", fontsize=7,
                 color="#444444", va="top")

    for ax in (axL, axR):
        ax.text(0.99, 1.0, "$\\sigma_A{=}0.25$  $\\sigma_B{=}0.50$  "
                           "$\\sigma_C{=}0.35$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                color="#666666")
    return _save(fig, out_dir, "fig2_p1")


# Figures 3 and 4 of the paper — computed from the raw outputs
# --------------------------------------------------------------------------

def _load(result_dirs: Sequence[str]):
    """Raw results and fronts, with the normalised indicators of analysis.py."""
    import analysis as A
    df, fronts = A.load_results(result_dirs)
    ind = A.normalised_indicators(df, fronts)
    return df.merge(ind, on=["instance", "method"]), fronts


def fig_scaling(df: pd.DataFrame, out_dir: str) -> str:
    """Fig. 3: (a) share of instances with the proven Z2 optimum,
    (b) cleaning water at the least-setup point, log scale."""
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.8))
    x = range(len(SCALES))
    g = df.groupby(["scale", "method"])
    for m in METHODS:
        st = STYLE[m]
        opt = [100 * g.get_group((s, m)).z2_optimal.mean() for s in SCALES]
        wat = [g.get_group((s, m)).cleaning_water_m3.mean() for s in SCALES]
        kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"],
                  zorder=st["zorder"], label=m, ms=4)
        ax[0].plot(x, opt, **kw)
        ax[1].plot(x, wat, **kw)
    ax[0].set_ylabel("Instances with proven\noptimal $Z_2$ (%)")
    ax[0].set_ylim(-3, 105)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Cleaning water (m$^3$)")
    for a, t in zip(ax, "ab"):
        a.set_xticks(list(x))
        a.set_xticklabels([SCALE_N[s] for s in SCALES])
        a.set_xlabel("Number of jobs")
        a.set_title(f"({t})", loc="left")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig34_scaling")


def fig_evidence(df: pd.DataFrame, fronts, out_dir: str,
                 instance: str = "me_n60_m3_f5_c3_t0.5_r0") -> str:
    """Fig. 4: (a) fronts on one 60-job instance with e = +2 and the bound of
    Corollary 3, (b) share of instances on which the proposed method attains
    the best IGD+, by colour excess."""
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.9))
    lb = df[df.instance == instance].z2_lower_bound.iloc[0]
    for m in METHODS:
        st = STYLE[m]
        pts = sorted(fronts[(instance, m)])
        ax[0].plot([p[1] for p in pts], [p[0] for p in pts], ls="none",
                   marker=st["marker"], color=st["color"], ms=5, label=m)
    ax[0].axvline(lb, color="0.4", ls="--", lw=0.8)
    ax[0].set_xlim(0, 3.2)
    ax[0].set_ylim(330, 480)
    ax[0].annotate("$Z_2^\\star$ (Cor. 3)", xy=(lb, 478), xytext=(4, -10),
                   textcoords="offset points", fontsize=7, color="0.4")
    ax[0].annotate("LPT+EDD off scale\n($Z_2 \\geq$ 4.8 h)", xy=(3.15, 470),
                   ha="right", va="top", fontsize=7, color="0.4")
    ax[0].set_xlabel("Total setup time $Z_2$ (h)")
    ax[0].set_ylabel("Total tardiness $Z_1$ (h)")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    ax[0].set_title("(a)", loc="left")

    p = df.pivot_table(index=["instance", "scale", "regime"], columns="method",
                       values="igd_plus").reset_index()
    p["best"] = p["CPM"] <= p[METHODS].min(axis=1) + 1e-9
    share = p.groupby(["scale", "regime"]).best.mean().unstack() * 100
    w = 0.26
    regs = [("e<0", "white", "////", "$e<0$"), ("e=0", "#9ecae1", "", "$e=0$"),
            ("e>0", "#08519c", "", "$e>0$")]
    for k, (r, c, h, lab) in enumerate(regs):
        ax[1].bar([i + (k - 1) * w for i in range(len(SCALES))],
                  [share.loc[s, r] for s in SCALES], w, color=c, hatch=h,
                  edgecolor="k", lw=0.6, label=lab)
    ax[1].set_xticks(list(range(len(SCALES))))
    ax[1].set_xticklabels([SCALE_N[s] for s in SCALES])
    ax[1].set_xlabel("Number of jobs")
    ax[1].set_ylabel("Best IGD$^+$ (% of instances)")
    ax[1].set_ylim(0, 105)
    ax[1].legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, 1.13))
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_evidence")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build Figs. 1-4 of the paper.")
    ap.add_argument("--results", nargs="+",
                    default=["results/part_a", "results/part_b",
                             "results/part_c", "results/part_d"])
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    df, fronts = _load(a.results)
    for p in (fig_gantt(a.out), fig_p1(a.out), fig_scaling(df, a.out),
              fig_evidence(df, fronts, a.out)):
        print("wrote", p)


if __name__ == "__main__":
    main()
