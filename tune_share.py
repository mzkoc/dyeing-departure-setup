"""Choose the hybrid's settings on calibration instances.

The hybrid has two settings: the share of the clock given to CPM before
ALNS+R takes over, and the design (seed, anchor or union; see hybrid.py).
They are chosen here on instances that are not used for testing -
replications 2 and 3 of the equal wall-clock subset, whereas the comparison
reports replications 0 and 1 - and by a rule fixed before the calibration is
run:

  1. keep the variants that attain the Z2 lower bound on every calibration
     instance;
  2. among them, take the lowest mean IGD+ rank, the ranks being taken per
     instance among the compared variants only;
  3. break a tie (mean ranks within 0.05) by the higher mean normalised HV.

IGD+ and HV are computed as in analyze_fair.py, against the union of every
run in the folder on that instance. ALNS+R is there as a reference: the table
also reports each variant's least tardiness relative to it.

By default the compared variants are the shares of the seed design
(CPM+ALNS+R@<share>); --variants names any other set.

    python tune_share.py results_tune
    python tune_share.py results_tune --variants CPM+ALNS+R@0.05,CPM+ALNS+R:anchor@0.05,CPM+ALNS+R:union@0.05
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from analyze_fair import load, runs_table

TIE = 0.05
ORDER = {"small": 0, "medium": 1, "large": 2, "xlarge": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="results_tune")
    ap.add_argument("--variants", default="")
    a = ap.parse_args()

    rows = load(a.folder)
    frames = []
    for scale in ORDER:
        sub = [r for r in rows if r["scale"] == scale]
        if sub:
            frames.append(runs_table(sub)[0])
    runs = pd.concat(frames)
    if set(runs.replication) & {0, 1}:
        print("WARNING: replications 0 or 1 found - those are the test "
              "instances and should not be used for calibration.\n")

    if a.variants:
        wanted = [v for v in a.variants.split(",") if v]
        missing = sorted(set(wanted) - set(runs.method))
        if missing:
            raise SystemExit(f"not in {a.folder}: {missing}")
    else:
        wanted = sorted({m for m in runs.method
                         if m.startswith("CPM+ALNS+R@")},
                        key=lambda m: float(m.split("@", 1)[1]))
    hyb = runs[runs.method.isin(wanted)].copy()
    hyb["variant"] = pd.Categorical(hyb.method, categories=wanted,
                                    ordered=True)
    hyb["igd_rank"] = hyb.groupby("instance").igd_plus.rank()
    hyb["z1_rank"] = hyb.groupby("instance").z1_min.rank()

    ref = runs[runs.method == "ALNS+R"].set_index("instance").z1_min
    cols = ["igd_rank", "z1_rank", "hv_norm", "igd_plus", "z2_opt"]
    if len(ref):
        hyb["z1_vs_alnsr"] = 100 * (hyb.z1_min / hyb.instance.map(ref) - 1)
        cols.append("z1_vs_alnsr")

    by_scale = (hyb.groupby(["scale", "variant"], observed=True)[cols]
                .mean().reset_index()
                .assign(o=lambda t: t.scale.map(ORDER))
                .sort_values(["o", "variant"]).drop(columns="o")
                .set_index(["scale", "variant"]))
    print("Per scale (mean over calibration instances; z1_vs_alnsr is the "
          "least tardiness in % above ALNS+R):")
    print(by_scale.round(3).to_string())

    overall = hyb.groupby("variant", observed=True)[cols].mean()
    print("\nAll calibration instances:")
    print(overall.round(3).to_string())

    n_inst = hyb.instance.nunique()
    ok = overall[overall.z2_opt >= 1.0 - 1e-9]
    if ok.empty:
        print("\nNo variant attains the Z2 bound on every instance.")
        return
    best_rank = ok.igd_rank.min()
    tied = ok[ok.igd_rank <= best_rank + TIE]
    choice = tied.hv_norm.idxmax()
    print(f"\nRule: Z2 bound on all {n_inst} instances -> lowest mean IGD+ "
          f"rank (ties within {TIE}) -> highest HV")
    print(f"Chosen: {choice}")
    if len(tied) > 1:
        print(f"  (tied on rank: {[str(v) for v in tied.index]}; chosen by HV)")
    name = "design_calibration.csv" if a.variants else "share_calibration.csv"
    out = os.path.join(a.folder, name)
    by_scale.reset_index().to_csv(out, index=False)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
