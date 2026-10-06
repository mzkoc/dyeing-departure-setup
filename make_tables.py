"""
LaTeX bodies of Tables 6-10 of the paper, built from the CSV files that
analysis.py writes. Only the tabular environments are produced; captions are
part of the manuscript.

Usage:
    python make_tables.py --tables results/tables --out results/tables/latex
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

SCALES = ["verify", "small", "medium", "large", "xlarge"]
LABEL = {"verify": "12--20", "small": "30", "medium": "60",
         "large": "100", "xlarge": "150"}
METHODS = ["CPM", "CPM-blind", "NSGA-II", "ALNS", "LPT+EDD"]
SC = {m: r"\textsc{" + m + "}" for m in METHODS}


def bold(x: str, cond: bool) -> str:
    return r"\textbf{" + x + "}" if cond else x


def sci(p: float) -> str:
    m, e = f"{float(p):.1e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def table6(d: str) -> str:
    t = pd.read_csv(os.path.join(d, "table6_main.csv"), index_col=[0, 1])
    rows = []
    for s in SCALES:
        g = t.loc[s]
        best = dict(HVn=g.HVn.max(), IGDplus=g.IGDplus.min(),
                    Z1min=g.Z1min.min(), Z2min=g.Z2min.min(),
                    water_m3=g.water_m3.min(), Z2opt_pct=g.Z2opt_pct.max())
        for k, m in enumerate(METHODS):
            r = g.loc[m]
            cells = [bold(f"{r.HVn:.3f}", abs(r.HVn - best["HVn"]) < 1e-9),
                     bold(f"{r.IGDplus:.3f}", abs(r.IGDplus - best["IGDplus"]) < 1e-9),
                     bold(f"{r.Z1min:.1f}", abs(r.Z1min - best["Z1min"]) < 1e-6),
                     bold(f"{r.Z2min:.2f}", abs(r.Z2min - best["Z2min"]) < 1e-9),
                     bold(f"{r.water_m3:.2f}", abs(r.water_m3 - best["water_m3"]) < 1e-9),
                     bold(f"{r.Z2opt_pct:.1f}", abs(r.Z2opt_pct - best["Z2opt_pct"]) < 1e-9),
                     f"{r.front:.1f}", f"{r.cpu_s:.2f}"]
            lead = r"\multirow{5}{*}{" + LABEL[s] + "}" if k == 0 else ""
            rows.append(f"{lead} & {SC[m]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if s != "xlarge" else r"\bottomrule")
    head = (r"\begin{tabular}{@{}llrrrrrrrr@{}}" "\n" r"\toprule" "\n"
            r"$n$ & Method & HV$_n$ & IGD$^+$ & $\Zone^{\min}$ & $\Ztwo^{\min}$ &"
            "\n" r"Water (m$^3$) & $\Ztwo$ opt.\ (\%) & Front & CPU (s) \\"
            "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\end{tabular}"


def table7(d: str) -> str:
    f = pd.read_csv(os.path.join(d, "friedman_igd_plus.csv")).set_index("scale")
    rows = []
    for s in SCALES:
        r = f.loc[s]
        v = [r[f"rank_{m}"] for m in METHODS]
        cells = [bold(f"{x:.2f}", abs(x - min(v)) < 1e-9) for x in v]
        rows.append(f"{LABEL[s]} & " + " & ".join(cells)
                    + f" & {int(r.cpm_best)}/{int(r.N)} \\\\")
    head = (r"\begin{tabular}{@{}lrrrrrl@{}}" "\n" r"\toprule" "\n"
            r" & \multicolumn{5}{c}{Mean rank} & \\" "\n" r"\cmidrule(lr){2-6}" "\n"
            r"$n$ & CPM & CPM-b & NSGA & ALNS & LPT & Best \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def table8(d: str) -> str:
    c = pd.read_csv(os.path.join(d, "table8_coverage.csv")).set_index(["scale", "method"])
    rows = []
    for s in SCALES:
        cells = [f"{c.loc[(s, m), 'C(CPM,B)']:.2f} / {c.loc[(s, m), 'C(B,CPM)']:.2f}"
                 for m in METHODS[1:]]
        rows.append(f"{LABEL[s]} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r" & \multicolumn{4}{c}{$C(\textsc{CPM}, B)$ \,/\, $C(B, \textsc{CPM})$} \\"
            "\n" r"\cmidrule(l){2-5}" "\n"
            r"$n$ & CPM-blind & NSGA-II & ALNS & LPT+EDD \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def table9(d: str) -> str:
    a = pd.read_csv(os.path.join(d, "table9_ablation.csv")).set_index("scale")
    rows = []
    for s in SCALES:
        r = a.loc[s]
        rows.append(f"{LABEL[s]} & {r.Z1_gain_median_pct:.1f} & {r.Z1_gain_mean_pct:.1f} & "
                    f"{r.HVn_gain_median_pct:.1f} & {int(r.wins)}/{int(r.N)} & "
                    f"{int(r.ties)} & {sci(r.wilcoxon_p)} \\\\")
    head = (r"\begin{tabular}{@{}lrrrrrr@{}}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{$\Zone$ gain (\%)} & HV$_n$ gain (\%) & & & \\"
            "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}" "\n"
            r"$n$ & median & mean & median & Wins & Ties & $p$ \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def table10(d: str) -> str:
    w = pd.read_csv(os.path.join(d, "walltime_summary.csv"), index_col=[0, 1])
    meth = METHODS[:4]
    rows = []
    for s in ["small", "medium", "large", "xlarge"]:
        g = w.loc[s].loc[meth]
        for k, m in enumerate(meth):
            r = g.loc[m]
            ev = f"{r.evals:,.0f}".replace(",", r"\,")
            lead = r"\multirow{4}{*}{" + LABEL[s] + "}" if k == 0 else ""
            rows.append(f"{lead} & {SC[m]} & "
                        f"{bold(f'{r.IGD:.2f}', abs(r.IGD - g.IGD.min()) < 1e-9)} & "
                        f"{bold(f'{r.Z1min:.1f}', abs(r.Z1min - g.Z1min.min()) < 1e-9)} & "
                        f"{r.Z2min:.2f} & {ev} & {r.Z2opt_pct:.1f} \\\\")
        if s != "xlarge":
            rows.append(r"\addlinespace[1pt]")
    head = (r"\begin{tabular}{@{}llrrrrr@{}}" "\n" r"\toprule" "\n"
            r"$n$ & Method & IGD & $\Zone^{\min}$ & $\Ztwo^{\min}$ & Evals & $\Ztwo$ opt.\,(\%) \\"
            "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def main() -> None:
    ap = argparse.ArgumentParser(description="LaTeX bodies of Tables 6-10.")
    ap.add_argument("--tables", default="results/tables",
                    help="directory with the CSV files written by analysis.py")
    ap.add_argument("--out", default="results/tables/latex")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in [("table6_main", table6), ("table7_friedman", table7),
                     ("table8_coverage", table8), ("table9_ablation", table9),
                     ("table10_walltime", table10)]:
        path = os.path.join(a.out, name + ".tex")
        with open(path, "w") as fh:
            fh.write(fn(a.tables) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
