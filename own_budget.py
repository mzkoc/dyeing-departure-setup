"""The companion method at the budget of its own campaign (check of 16 Nov 2026).

    python own_budget.py results_valid results_comp7 results_valid_extra results_comp7own

In the paper (Section 7.3, Table 8) SA-EDDTS of the companion study runs on
the equal clock: 3.29 s per instance at 30 jobs and 63.92 s at 60 jobs. Its
own campaign gave it 80 s at 30 jobs and 120-160 s at 50-80 jobs (133.3 s
at 60 jobs by linear interpolation). results_comp7own holds SA-EDDTS run
with those budgets on the validation instances (replications 4-9) at 30 and
60 jobs. The hybrid and every other method keep the equal clock.

For each scale and regime (and pooled) the report gives, as Table 8 does,
the hybrid's IGD+ wins/ties/losses against SA-EDDTS, the median difference
of SA-EDDTS's least tardiness relative to the hybrid's (%), the share of
instances on which SA-EDDTS attains the setup optimum and its mean least
setup, once for SA-EDDTS on the equal clock and once at its own budget.
IGD+ is normalised by the per-instance union of all the methods listed in
METHODS, the own-budget runs included; with OWN left out, the numbers of the
equal-clock rows reproduce Table 8 (printed first as a check).

Also written: own_budget_instances.csv (one row per instance and method).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_fair import load, runs_table

H = "CPM+ALNS+R:union@0.1"
BASE = "FFD+SA-EDDTS"
OWN = "FFD+SA-EDDTS@own"
TABLE8 = [H, "CPM", "ALNS+R", "ALNS", BASE, "FFD+SA-EDDTS+R"]
METHODS = TABLE8 + [OWN]
SCALES = [("small", 30), ("medium", 60)]
REGIMES = [(-1, "e<0"), (0, "e=0"), (1, "e>0"), (None, "all")]


def rows_from(folders, own_folder):
    rows = [r for f in folders for r in load(f)]
    for r in load(own_folder):
        r = dict(r)
        r["method"] = OWN
        rows.append(r)
    return rows


def compare(mean, ins, x):
    h = mean.loc[H].loc[ins]
    o = mean.loc[x].loc[ins]
    d = o.igd_plus - h.igd_plus                  # > 0: hybrid better
    w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    t = len(ins) - w - l
    p = (wilcoxon(h.igd_plus, o.igd_plus).pvalue
         if (d.abs() > 1e-9).any() else 1.0)
    dz = (100 * (o.z1_min - h.z1_min) / h.z1_min.where(h.z1_min > 1e-12))
    return {"W/T/L": f"{w}/{t}/{l}", "PS": (w + 0.5 * t) / len(ins),
            "p": p, "dZ1 median %": float(dz.median()),
            "Z2* %": 100 * o.z2_opt.mean(), "least Z2": o.z2_min.mean(),
            "Z2*": h.z2_min.mean()}


def report(rows, methods, title):
    print(f"\n######## {title}")
    per_inst = []
    for sc, n in SCALES:
        sub = [r for r in rows if r["scale"] == sc and r["method"] in methods]
        own_ins = {r["instance"] for r in sub if r["method"] == OWN}
        if OWN in methods and own_ins:
            if len(own_ins) < 54:
                print(f"WARNING: {n} jobs: own-budget runs on {len(own_ins)} "
                      f"of 54 instances; the report covers those only")
            sub = [r for r in sub if r["instance"] in own_ins]
        runs, _, got = runs_table(sub, methods)
        if not got:
            print(f"{n} jobs: no runs")
            continue
        runs["reg"] = np.sign(runs.excess).astype(int)
        clock = runs.groupby("method")[["clock", "seconds"]].mean()
        print(f"\n=== {n} jobs: {runs.instance.nunique()} instances; "
              f"clock / seconds used per method:")
        print(clock.round(2).to_string())
        mean = runs.groupby(["method", "instance"])[
            ["igd_plus", "hv_norm", "z1_min", "z2_min", "z2_opt"]].mean()
        reg = runs.groupby("instance").reg.first()
        out = []
        for g, gl in REGIMES:
            ins = reg.index if g is None else reg[reg == g].index
            for x in (BASE, OWN):
                if x not in got:
                    continue
                c = compare(mean, ins, x)
                out.append({"jobs": n, "regime": gl, "instances": len(ins),
                            "method": x, **c})
        df = pd.DataFrame(out)
        with pd.option_context("display.width", 200):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if OWN in got:
            b, o = mean.loc[BASE], mean.loc[OWN]
            idx = b.index.intersection(o.index)
            b, o = b.loc[idx], o.loc[idx]
            dz = 100 * (o.z1_min - b.z1_min) / b.z1_min.where(b.z1_min > 1e-12)
            print(f"own budget against equal clock ({len(idx)} instances): "
                  f"IGD+ lower on {int((o.igd_plus < b.igd_plus - 1e-9).sum())}, "
                  f"higher on {int((o.igd_plus > b.igd_plus + 1e-9).sum())}; "
                  f"least tardiness median {dz.median():+.1f}% "
                  f"(lower on {int((o.z1_min < b.z1_min - 1e-9).sum())}); "
                  f"setup optimum {100 * b.z2_opt.mean():.0f}% -> "
                  f"{100 * o.z2_opt.mean():.0f}%; least setup "
                  f"{b.z2_min.mean():.2f} -> {o.z2_min.mean():.2f} h")
        m = mean.reset_index()
        m["jobs"] = n
        m["regime"] = m.instance.map(reg)
        per_inst.append(m)
    return pd.concat(per_inst) if per_inst else None


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    *folders, own = sys.argv[1:]
    rows = rows_from(folders, own)
    report([r for r in rows if r["method"] != OWN], TABLE8,
           "check: equal clock only (must match Table 8, SA-EDDTS columns)")
    per = report(rows, METHODS, "with SA-EDDTS at its own budget")
    if per is not None:
        per.to_csv("own_budget_instances.csv", index=False)
        print("\nwritten: own_budget_instances.csv")


if __name__ == "__main__":
    main()
