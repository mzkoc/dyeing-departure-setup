"""LaTeX tables of the revised paper, computed from the result folders.

    python paper_tables.py main   results_valid            > tab_main.tex
    python paper_tables.py comp7  results_valid results_comp7 [results_valid_extra] > tab_comp7.tex
    python paper_tables.py water  results_valid            > tab_water.tex
    python paper_tables.py budget results_valid results_budget > tab_budget.tex
    python paper_tables.py sens   results_sens             > tab_sens.tex
    python paper_tables.py perturb results_sens
    python paper_tables.py scale  results_scale            > tab_scale.tex
    python paper_tables.py cpm    results_cpmparams        > tab_cpm.tex
    python paper_tables.py e2     results_e2               > tab_e2.tex
    python paper_tables.py eddts  results_valid results_comp7 > tab_eddts.tex
    python paper_tables.py e2regime results_e2             > tab_e2regime.tex
    python paper_tables.py calib  results_calib            > tab_calib.tex
    python paper_tables.py repro  results_comp7repro       > tab_repro.tex

Indicators, normalisation, tests and the Z2 grid as in analyze_fair.py; the
main table uses the six main methods and the confirmatory pairs (H against
the other five, Holm over those five per scale and indicator).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from analyze_fair import holm, load, paired, runs_table

H = "CPM+ALNS+R:union@0.1"
MAIN = [H, "ALNS+R", "NSGA-II+R", "CPM", "ALNS", "NSGA-II"]
NAME = {H: "\\textsc{CPM+ALNS+R}", "ALNS+R": "\\textsc{ALNS+R}",
        "NSGA-II+R": "\\textsc{NSGA-II+R}", "CPM": "\\textsc{CPM}",
        "ALNS": "\\textsc{ALNS}", "NSGA-II": "\\textsc{NSGA-II}",
        "FFD+SA-EDDTS": "[7]", "FFD+SA-EDDTS+R": "[7]+R",
        "CPM+R": "\\textsc{CPM+R}", "CPM-blind": "\\textsc{CPM-blind}"}
SCALE = [("verify", "12--20"), ("small", "30"), ("medium", "60"),
         ("large", "100"), ("xlarge", "150")]


def _rows(folders):
    return [r for f in folders for r in load(f)]


def main_table(folders):
    rows = _rows(folders)
    out = []
    for sc, lab in SCALE:
        sub = [r for r in rows if r["scale"] == sc and r["method"] in MAIN]
        runs, _, ms = runs_table(sub, MAIN)
        mean = runs.groupby(["method", "instance"])[
            ["hv_norm", "igd_plus", "z1_min", "z2_opt", "front"]].mean()
        agg = mean.groupby("method").mean()
        others = MAIN[1:]
        res = {x: (paired(mean, H, x, "igd_plus"), paired(mean, H, x, "z1_min"))
               for x in others}
        pi = dict(zip(others, holm([res[x][0][3] for x in others])))
        pz = dict(zip(others, holm([res[x][1][3] for x in others])))
        best = {c: (agg[c].max() if c == "hv_norm" else agg[c].min())
                for c in ("hv_norm", "igd_plus", "z1_min")}
        n = runs.instance.nunique()
        for k, m in enumerate(MAIN):
            a = agg.loc[m]

            def b(c, fmt):
                s = format(a[c], fmt)
                return f"\\textbf{{{s}}}" if abs(a[c] - best[c]) < 1e-9 else s
            first = (f"\\multirow{{6}}{{*}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}"
                     f"{lab}\\\\({n})\\end{{tabular}}}}" if k == 0 else "")
            if m == H:
                tail = "-- & --"
            else:
                (w, t, l, _, _), (_, _, _, _, med) = res[m]
                ps = (w + 0.5 * t) / max(w + t + l, 1)
                sig_i = "$^{\\ast}$" if pi[m] < 0.05 else ""
                sig_z = "$^{\\ast}$" if pz[m] < 0.05 else ""
                v = -med
                dz = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$")
                tail = f"{ps:.2f}{sig_i} & {dz}{sig_z}"
            out.append(f"{first} & {NAME[m]} & {b('hv_norm', '.3f')} & "
                       f"{b('igd_plus', '.3f')} & {b('z1_min', '.1f')} & "
                       f"{100 * a['z2_opt']:.0f} & {a['front']:.1f} & {tail} \\\\")
        out.append("\\midrule" if sc != "xlarge" else "\\bottomrule")
    print("\n".join(out))


def comp7_table(folders):
    """[7] (and [7]+R if present) against H, CPM, ALNS+R and ALNS, by
    regime: H's IGD+ wins/ties/losses and median Z1min difference against
    [7]; [7]'s attainment of the Z2 bound and mean least setup against the
    optimum."""
    rows = _rows(folders)
    ms = [H, "CPM", "ALNS+R", "ALNS", "FFD+SA-EDDTS"]
    if any(r["method"] == "FFD+SA-EDDTS+R" for r in rows):
        ms.append("FFD+SA-EDDTS+R")
    out = []
    for sc, lab in SCALE:
        sub = [r for r in rows if r["scale"] == sc and r["method"] in ms]
        runs, _, got = runs_table(sub, ms)
        runs["reg"] = np.sign(runs.excess).astype(int)
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "z1_min", "z2_min", "z2_opt"]].mean()
        reg = runs.groupby("instance").reg.first()
        for g, gl in ((-1, "$e<0$"), (0, "$e=0$"), (1, "$e>0$")):
            ins = reg[reg == g].index
            h = mean.loc[H].loc[ins]
            cells = []
            for x in [m for m in got if m.startswith("FFD")]:
                o = mean.loc[x].loc[ins]
                d = o.igd_plus - h.igd_plus
                w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                z = float((100 * (o.z1_min - h.z1_min) / h.z1_min.where(
                    h.z1_min > 1e-12)).median())
                zs = ("0.0" if abs(z) < 0.05 else f"{z:+.1f}").replace("-", "$-$")
                cell = (f"{w}/{len(ins) - w - l}/{l} & {zs} & "
                        f"{100 * o.z2_opt.mean():.0f}")
                if not x.endswith("+R"):      # R leaves setups unchanged
                    cell += f" & {o.z2_min.mean():.2f}"
                cells.append(cell)
            a = mean.loc["ALNS"].loc[ins]
            za = float((100 * (a.z1_min - h.z1_min) / h.z1_min.where(
                h.z1_min > 1e-12)).median())
            out.append(f"{lab if g == -1 else ''} & {gl} & "
                       f"{h.z2_min.mean():.2f} & " + " & ".join(cells)
                       + " & " + f"{za:+.1f}".replace("-", "$-$") + " \\\\")
        out.append("\\midrule" if sc != "xlarge" else "\\bottomrule")
    print("\n".join(out))


def water_table(folders):
    from instances import FAMILY_POOL
    sig = {f: s for f, _, s in FAMILY_POOL}
    smin = min(sig.values())
    rows = [r for r in _rows(folders) if r["method"] in MAIN]

    def w(dep, w0):           # m3 at liquor ratio 1:8, Q = 400 kg
        V = 400 * 8.0
        return sum(d * (w0 * V + V / smin * sig[f])
                   for f, d in dep.items()) / 1000

    least = {}
    for r in rows:
        # departures are logged per point before snapping: take the point
        # of least setup by its logged departures
        tot = [sum(sig[f] * d for f, d in dep.items())
               for dep in r["departures"]]
        k = int(np.argmin(tot))
        least[(r["scale"], r["method"], r["instance"])] = r["departures"][k]
    out = []
    for sc, lab in SCALE:
        ins = sorted({i for (s, m, i) in least if s == sc and m == H})
        vals = {m: [w(least[(sc, m, i)], 0) for i in ins] for m in MAIN}
        red = {}
        for m in MAIN[1:]:
            red[m] = [100 * (1 - sum(w(least[(sc, H, i)], w0) for i in ins)
                             / sum(w(least[(sc, m, i)], w0) for i in ins))
                      for w0 in (0.0, 1.0)]
        out.append(f"{lab} & " + " & ".join(f"{np.mean(vals[m]):.2f}"
                                            for m in (H, "ALNS+R",
                                                      "NSGA-II+R"))
                   + " & " + " & ".join(f"{red[m][0]:.0f} ({red[m][1]:.0f})"
                                        for m in ("ALNS+R", "NSGA-II+R"))
                   + " \\\\")
    print("\n".join(out))


def budget_table(folders):
    """Budget robustness (ANALYSIS_PLAN.md, section 9): H, ALNS+R,
    NSGA-II+R and CPM at 0.25, 0.5, 1 and 2 times the clock, replications
    4-6. Per multiplier and scale, normalised by the union of the four
    methods at that multiplier: H's IGD+ PS against each other method
    (* Holm p < 0.05 over the three), the median paired difference of the
    method's least tardiness relative to H's (%), and the share of
    instances on which the method attains the setup optimum."""
    rows = [r for r in _rows(folders)
            if r["replication"] in (4, 5, 6) and r["method"] in MAIN[:4]]
    ms = MAIN[:4]
    out = []
    for cm in (0.25, 0.5, 1.0, 2.0):
        for k, (sc, lab) in enumerate(SCALE):
            sub = [r for r in rows if r["scale"] == sc
                   and abs(r.get("clock_mult", 1.0) - cm) < 1e-9]
            runs, _, _ = runs_table(sub, ms)
            mean = runs.groupby(["method", "instance"])[
                ["igd_plus", "z1_min", "z2_opt"]].mean()
            others = ms[1:]
            res = {x: (paired(mean, H, x, "igd_plus"),
                       paired(mean, H, x, "z1_min")) for x in others}
            pi = dict(zip(others, holm([res[x][0][3] for x in others])))
            pz = dict(zip(others, holm([res[x][1][3] for x in others])))
            cells = []
            for x in others:
                (w, t, l, _, _), (_, _, _, _, med) = res[x]
                ps = (w + 0.5 * t) / max(w + t + l, 1)
                v = -med
                dz = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$")
                cells.append(f"{ps:.2f}{'$^{\\ast}$' if pi[x] < 0.05 else ''}"
                             f" & {dz}{'$^{\\ast}$' if pz[x] < 0.05 else ''}"
                             f" & {100 * mean.loc[x].z2_opt.mean():.0f}")
            first = f"$\\times{cm:g}$" if k == 0 else ""
            out.append(f"{first} & {lab} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if cm != 2.0 else "\\bottomrule")
    print("\n".join(out))


def _vs_h(sub, ms):
    """H against each other method of ms: IGD+ PS (* Holm p < 0.05 over the
    others), median least-tardiness difference of the method relative to
    H's (% of the method's; * likewise) and its setup-optimum attainment."""
    runs, _, _ = runs_table(sub, ms)
    mean = runs.groupby(["method", "instance"])[
        ["igd_plus", "z1_min", "z2_opt", "z2_min"]].mean()
    others = ms[1:]
    res = {x: (paired(mean, H, x, "igd_plus"), paired(mean, H, x, "z1_min"))
           for x in others}
    pi = dict(zip(others, holm([res[x][0][3] for x in others])))
    pz = dict(zip(others, holm([res[x][1][3] for x in others])))
    cells = []
    for x in others:
        (w, t, l, _, _), (_, _, _, _, med) = res[x]
        ps = (w + 0.5 * t) / max(w + t + l, 1)
        v = -med
        dz = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$")
        st = "$^{\\ast}$"
        cells.append(f"{ps:.2f}{st if pi[x] < 0.05 else ''} & "
                     f"{dz}{st if pz[x] < 0.05 else ''} & "
                     f"{100 * mean.loc[x].z2_opt.mean():.0f}")
    return cells, mean


SENS = [("perturb", "base"), ("sigx2", "$\\sigma \\times 2$"),
        ("sigx4", "$\\sigma \\times 4$"), ("skew", "skewed customers"),
        ("cluster", "clustered due dates"), ("c5", "$|C| = 5$"),
        ("q300", "$Q = 300$~kg"), ("size", "sizes $U[100,350]$")]


def sens_table(folders):
    """Supplementary Table S6: one factor at a time, e > 0 cells at 60 and
    150 jobs, replications 4-9; normalised within factor and scale."""
    rows = _rows(folders)
    ms = MAIN[:4]
    out = []
    for k, (sc, lab) in enumerate((("medium", "60"), ("xlarge", "150"))):
        for j, (v, name) in enumerate(SENS):
            sub = [r for r in rows if r["scale"] == sc
                   and r.get("variant") == v]
            cells, mean = _vs_h(sub, ms)
            first = lab if j == 0 else ""
            opt = {r["instance"]: r["z2_bound"] for r in sub}
            out.append(f"{first} & {name} & "
                       f"{np.mean(list(opt.values())):.2f} & "
                       + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k == 0 else "\\bottomrule")
    print("\n".join(out))


def perturb_summary(folders):
    """Perturbed evaluation (variant perturb, five draws, no Z2 grid): per
    scale, each method's median change of least tardiness from nominal to
    perturbed (mean over draws), H's attainment of the perturbed setup
    optimum, and H against ALNS+R on instance means over the draws."""
    from core import z2_lower_bound
    from variants import PERTURB_DRAWS, cells, make_instance, perturbed
    rows = [r for f in folders for r in load(f, 0)
            if r.get("variant") == "perturb"]
    ms = MAIN[:4]
    for sc in ("medium", "xlarge"):
        sub = [r for r in rows if r["scale"] == sc]
        nom, _, _ = runs_table(sub, ms)
        nomm = nom.groupby(["method", "instance"]).z1_min.mean()
        prs = []
        for r in sub:
            c = [c for c in cells(sc) if c.machines == r["machines"]][0]
            inst = make_instance(c, r["tau"], r["replication"], "perturb")
            for d in range(PERTURB_DRAWS):
                r2 = dict(r, instance=f"{r['instance']}|d{d}",
                          points=r["perturbed"][d],
                          z2_bound=z2_lower_bound(perturbed(inst, d)))
                prs.append(r2)
        runs, _, _ = runs_table(prs, ms)
        runs["base"] = runs.instance.str.split("|").str[0]
        line = [sc]
        for m in ms:
            g = runs[runs.method == m].groupby("base").z1_min.mean()
            nm = nomm.loc[m].loc[g.index]
            line.append(f"{m}: Z1min {100 * ((g - nm) / nm).median():+.1f}%")
        hz = runs[runs.method == H]
        line.append(f"H setup-optimal in {100 * hz.z2_opt.mean():.0f}% of "
                    "draws")
        mean = runs.groupby(["method", "base"])[["igd_plus", "z1_min"]].mean()
        mean.index = mean.index.set_names(["method", "instance"])
        cells_, _ = _vs_h_mean(mean, ms)
        line.append("vs ALNS+R (PS, dZ1, -): " + cells_[0])
        print("\n  ".join(line))


def _vs_h_mean(mean, ms):
    others = ms[1:]
    res = {x: (paired(mean, H, x, "igd_plus"), paired(mean, H, x, "z1_min"))
           for x in others}
    pi = dict(zip(others, holm([res[x][0][3] for x in others])))
    out = []
    for x in others:
        (w, t, l, _, _), (_, _, _, _, med) = res[x]
        out.append(f"{(w + 0.5 * t) / max(w + t + l, 1):.2f} "
                   f"(p {pi[x]:.3f}), {-med:+.1f}")
    return out, None


def scale_table(folders):
    """Supplementary Table S7: 200 and 300 jobs, by regime and overall."""
    rows = _rows(folders)
    ms = MAIN[:4]
    out = []
    for k, (sc, lab) in enumerate((("x200", "200"), ("x300", "300"))):
        sub = [r for r in rows if r["scale"] == sc]
        for j, (g, name) in enumerate(((4, "$e > 0$"), (0, "$e = 0$"),
                                       (-4, "$e < 0$"), (None, "all"))):
            part = [r for r in sub if g is None or r["excess"] == g]
            cells, mean = _vs_h(part, ms)
            opt = {r["instance"]: r["z2_bound"] for r in part}
            out.append(f"{lab if j == 0 else ''} & {name} & "
                       f"{np.mean(list(opt.values())):.2f} & "
                       + " & ".join(cells) + " \\\\")
        out.append("\\midrule" if k == 0 else "\\bottomrule")
    print("\n".join(out))


def cpm_table(folders):
    """Supplementary Table S4: CPM's hand-set parameters one at a time
    inside H, calibration replications 2-3 at 60 and 100 jobs; normalised
    by the union of the 15 configurations. Mean HV and IGD+, median and
    mean least-tardiness change against the default (%), Wilcoxon p on
    IGD+ against the default (unadjusted; screening)."""
    from scipy.stats import wilcoxon
    import cpm_params
    names = cpm_params.variant_names(H)
    rows = [r for r in _rows(folders) if r["method"] in names]
    lab = {"outer_passes": "outer passes", "first_share": "first-sweep share",
           "pareto_passes": "ladder levels per sweep",
           "stall_levels": "stall levels",
           "block_perm": "exhaustive block orders",
           "joint_orders": "exact joint orders below",
           "adapt_rounds": "CPM-A rounds"}
    res = {}
    for sc in ("medium", "large"):
        runs, _, _ = runs_table([r for r in rows if r["scale"] == sc], names)
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "hv_norm", "z1_min"]].mean()
        d = mean.loc[H]
        for m in names:
            o = mean.loc[m]
            dz = 100 * (o.z1_min - d.z1_min) / d.z1_min.where(d.z1_min > 0)
            if m == H:
                p = None
            else:
                try:
                    p = wilcoxon(o.igd_plus, d.igd_plus).pvalue
                except ValueError:
                    p = 1.0
                if p != p:
                    p = 1.0
            res[(sc, m)] = (o.hv_norm.mean(), o.igd_plus.mean(),
                            dz.median(), dz.mean(), p)
    out = []
    for m in names:
        if m == H:
            name, lvl = "default", ""
        else:
            k, v = m.split("~")[1].split("=")
            name, lvl = lab[k], v
        cells = []
        for sc in ("medium", "large"):
            hv, ig, md, mn, p = res[(sc, m)]
            mds = ("0.0" if abs(md) < 0.05 else f"{md:+.1f}").replace("-", "$-$")
            mns = ("0.00" if abs(mn) < 0.005 else f"{mn:+.2f}").replace("-", "$-$")
            cells.append(f"{hv:.4f} & {ig:.4f} & {mds} & {mns} & "
                         + ("--" if p is None else f"{p:.3f}"))
        out.append(f"{name} & {lvl} & " + " & ".join(cells) + " \\\\")
    print("\n".join(out))


def e2_table(folders):
    """Table of E2 (section 7.4): per scale and lambda, the true setup
    optimum (mean, h), the share of instances on which each front attains
    it, H-max's IGD+ PS against H-mix, ALNS+R and [7]+R (* Holm p < 0.05
    over the four comparisons of the main set, H-max+D included), and the
    median least-tardiness difference of ALNS+R and [7]+R relative to H-max
    (% of the method's; positive: H-max lower; * likewise). Normalised by
    the union of H-max, H-max+D, H-mix, ALNS+R and [7]+R; no Z2 grid."""
    _e2_rows(folders, by_regime=False)


def e2_regime_table(folders):
    """Supplementary Table S8: Table 9 by regime e = |F| - m (normalisation
    within scale, lambda and regime)."""
    _e2_rows(folders, by_regime=True)


def _e2_rows(folders, by_regime):
    from e2_analyze import MAIN as E2MAIN, expand
    rows = expand([r for f in folders for r in load(f, 0)])
    out = []
    groups = (((-1, "$e<0$"), (0, "$e=0$"), (1, "$e>0$")) if by_regime
              else ((None, ""),))
    for k, (sc, lab) in enumerate((("medium", "60"), ("xlarge", "150"))):
      for j, lam in enumerate((1.0, 2.0)):
        for i, (g, gname) in enumerate(groups):
            sub = [r for r in rows if r["scale"] == sc and r["lam"] == lam
                   and (g is None or np.sign(r["excess"]) == g)]
            runs, _, _ = runs_table([r for r in sub
                                     if r["method"] in E2MAIN], E2MAIN)
            mean = runs.groupby(["method", "instance"])[
                ["igd_plus", "z1_min", "z2_min", "z2_opt"]].mean()
            others = E2MAIN[1:]
            res = {x: (paired(mean, "H-max", x, "igd_plus"),
                       paired(mean, "H-max", x, "z1_min")) for x in others}
            pi = dict(zip(others, holm([res[x][0][3] for x in others])))
            pz = dict(zip(others, holm([res[x][1][3] for x in others])))
            hb = [r for r in sub if r["method"] == "H-max+B"]
            att_b = 100 * sum(min(p[1] for p in r["points"])
                              <= r["bracket"]["LB"] + 1e-6
                              for r in hb) / max(len(hb), 1)
            st = r"$^{\ast}$"
            # true setup optimum: LB of the bracket, which closes on every
            # E2 instance (not H-max's least setup, which may miss it)
            opt = {r["instance"]: r["bracket"]["LB"] for r in sub}
            cells = [f"{np.mean(list(opt.values())):.2f}"]
            att = {m: 100 * mean.loc[m].z2_opt.mean() for m in E2MAIN}
            cells += [f"{att['H-max']:.0f}", f"{att_b:.0f}",
                      f"{att['H-mix']:.0f}", f"{att['ALNS+R']:.0f}",
                      f"{att['[7]+R']:.0f}"]
            for x in ("H-mix", "ALNS+R", "[7]+R"):
                w, t, l, _, _ = res[x][0]
                ps = (w + 0.5 * t) / max(w + t + l, 1)
                cells.append(f"{ps:.2f}{st if pi[x] < 0.05 else ''}")
            for x in ("ALNS+R", "[7]+R"):
                v = -res[x][1][4]
                dz = "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$")
                cells.append(f"{dz}{st if pz[x] < 0.05 else ''}")
            first = (lab if j == 0 and i == 0 else "") + " & " + (
                f"{lam:g}" if i == 0 else "")
            reg = f" & {gname}" if by_regime else ""
            out.append(first + reg + " & " + " & ".join(cells) + " \\\\")
        if by_regime and not (k == 1 and j == 1):
            out.append("\\midrule" if j == 1 else "\\cmidrule(l){2-14}")
    print("\n".join(out))


def eddts_table(folders):
    """Supplementary Table S9: the version of the companion method without
    the annealing step of its first stage (FFD+EDDTS; FFD-EDDTS in the code),
    which that study recommends, beside the version with it (FFD+SA-EDDTS),
    by scale and regime: H's IGD+ wins/ties/losses against FFD+EDDTS (union
    of H, CPM, ALNS+R, ALNS and both versions), the median least-tardiness
    difference relative to H's (%), attainment of the setup optimum (%) and
    mean least setup (h); last column: median difference of the least
    tardiness of FFD+EDDTS relative to that of FFD+SA-EDDTS (%)."""
    ms = [H, "CPM", "ALNS+R", "ALNS", "FFD+SA-EDDTS", "FFD-EDDTS"]
    rows = [r for r in _rows(folders) if r["method"] in ms]
    out = []

    def fmt(v):
        return ("0.0" if abs(v) < 0.05 else f"{v:+.1f}").replace("-", "$-$")
    for k, (sc, lab) in enumerate(SCALE):
        runs, _, _ = runs_table([r for r in rows if r["scale"] == sc], ms)
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "z1_min", "z2_min", "z2_opt"]].mean()
        reg = runs.groupby("instance").excess.first().apply(np.sign)
        for j, (g, name) in enumerate(((-1, "$e<0$"), (0, "$e=0$"),
                                       (1, "$e>0$"))):
            ins = reg[reg == g].index
            h = mean.loc[H].loc[ins]
            cells = []
            for x in ("FFD-EDDTS",):          # FFD+SA-EDDTS: Table 8
                o = mean.loc[x].loc[ins]
                d = o.igd_plus - h.igd_plus
                w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
                dz = 100 * (o.z1_min - h.z1_min) / h.z1_min.where(h.z1_min > 1e-12)
                cells.append(f"{w}/{len(ins) - w - l}/{l} & {fmt(dz.median())} & "
                             f"{100 * o.z2_opt.mean():.0f} & {o.z2_min.mean():.2f}")
            a, b = mean.loc["FFD+SA-EDDTS"].loc[ins], mean.loc["FFD-EDDTS"].loc[ins]
            dd = 100 * (b.z1_min - a.z1_min) / a.z1_min.where(a.z1_min > 1e-12)
            out.append(f"{lab if j == 0 else ''} & {name} & "
                       + " & ".join(cells) + f" & {fmt(dd.median())} \\\\")
        out.append("\\midrule" if k < len(SCALE) - 1 else "\\bottomrule")
    print("\n".join(out))


def calib_table(folders):
    """Supplementary Tables S2 and S3: calibration on replications 2-3 (126
    instances), by the rule of tune.py: per instance, IGD+ ranks among the
    candidates; lowest mean rank, ties within 0.05 broken by mean HV. S2:
    the ten candidates of ALNS+R and of NSGA-II+R (L9 rows and the default);
    S3: the five hybrid candidates."""
    from params import FACTORS, row_params
    from tune import ORDER, TIE
    rows = _rows(folders)

    def table(wanted, require_z2=False):
        sub = [r for r in rows if r["method"] in wanted]
        frames = []
        for sc in ORDER:
            part = [r for r in sub if r["scale"] == sc]
            if part:
                frames.append(runs_table(part)[0])
        runs = pd.concat(frames)
        runs = runs[runs.method.isin(wanted)].copy()
        runs["igd_rank"] = runs.groupby("instance").igd_plus.rank()
        ov = runs.groupby("method")[["igd_rank", "hv_norm", "igd_plus",
                                     "z2_opt"]].mean().loc[wanted]
        ok = ov[ov.z2_opt >= 1 - 1e-9] if require_z2 else ov
        tied = ok[ok.igd_rank <= ok.igd_rank.min() + TIE]
        return ov, str(tied.hv_norm.idxmax()), runs.instance.nunique()

    for base in ("ALNS+R", "NSGA-II+R"):
        wanted = [f"{base}#{k}" for k in range(10)]
        ov, choice, n = table(wanted)
        print(f"% {base}: {n} instances; chosen {choice}")
        for k, m in enumerate(wanted):
            p = row_params(base, k)
            lv = " & ".join(f"{p[name]:g}" for name, _ in FACTORS[base])
            mark = "$\\checkmark$" if m == choice else ""
            print(f"{base if k == 0 else ''} & {k} & {lv} & "
                  f"{ov.loc[m, 'igd_rank']:.2f} & {ov.loc[m, 'hv_norm']:.3f} & "
                  f"{100 * ov.loc[m, 'z2_opt']:.0f} & {mark} \\\\")
        print("\\midrule")
    hyb = ["CPM+ALNS+R:union@0.02", "CPM+ALNS+R:union@0.05",
           "CPM+ALNS+R:union@0.1", "CPM+ALNS+R:union@0.2",
           "CPM+ALNS+R:anchor@0.05"]
    ov, choice, n = table(hyb, require_z2=True)
    print(f"% hybrid: {n} instances; chosen {choice}")
    for m in hyb:
        design, share = m.split(":")[1].split("@")
        mark = "$\\checkmark$" if m == choice else ""
        print(f"{design} & {share} & {ov.loc[m, 'igd_rank']:.2f} & "
              f"{ov.loc[m, 'hv_norm']:.3f} & {ov.loc[m, 'igd_plus']:.4f} & "
              f"{100 * ov.loc[m, 'z2_opt']:.0f} & {mark} \\\\")


def repro_table(folders):
    """Supplementary Table S10: reproduction check of the adapter for the
    companion method (check_comp7.py): per task and measure (least
    tardiness, least setup, front size), the range of the five archived
    runs, the two runs of the companion study's own runner on this machine
    and the two runs of this repository's adapter."""
    import os
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "comp7", "repro_reference.json")))
    recs = [json.loads(x) for f in folders
            for x in open(os.path.join(f, "results.jsonl")) if x.strip()]
    keys = list(ref["tasks"])
    for i, k in enumerate(keys):
        v = ref["tasks"][k]
        st, t = v["stored"], v["task"]
        name = f"{t['preset']} (seed {t['base_seed']})"
        for j, (col, key, fmt, lab) in enumerate((
                ("z1_min", "z1_min", ".2f", "least tardiness"),
                ("z2_min", "z2_min", ".3f", "least setup"),
                ("points", "n_points", "d", "points on the front"))):
            sv = [x[col] for x in st]
            own = sorted((r for r in recs if r["task"] == k
                          and r["path"] == "own"), key=lambda r: r["rep"])
            ad = sorted((r for r in recs if r["task"] == k
                         and r["path"] == "adapter"), key=lambda r: r["rep"])
            print(f"{name if j == 0 else ''} & {lab} & "
                  f"{min(sv):{fmt}}--{max(sv):{fmt}} & "
                  + ", ".join(format(r[key], fmt) for r in own) + " & "
                  + ", ".join(format(r[key], fmt) for r in ad) + " \\\\")
        print("\\midrule" if i < len(keys) - 1 else "\\bottomrule")


if __name__ == "__main__":
    kind, folders = sys.argv[1], sys.argv[2:]
    {"main": main_table, "comp7": comp7_table, "water": water_table,
     "budget": budget_table, "sens": sens_table, "scale": scale_table,
     "cpm": cpm_table, "perturb": perturb_summary, "e2": e2_table,
     "eddts": eddts_table, "e2regime": e2_regime_table,
     "calib": calib_table, "repro": repro_table}[kind](
        folders)
