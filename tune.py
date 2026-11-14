"""Choose among calibration variants by the rule fixed before the runs.

Candidates are given with --variants (comma-separated) or --prefix (every
method in the folder whose name starts with it, e.g. "ALNS+R#"). Indicators
are computed as in analyze_fair.py, with the normalisation and the IGD+
reference built from the candidates' runs only.

Rule (ANALYSIS_PLAN.md, section 3):
  1. with --require-z2, keep the candidates that attain the Z2 lower bound on
     every calibration instance (the hybrid; the benchmarks do not attain it
     and are not filtered);
  2. take the lowest mean IGD+ rank, ranks taken per instance among the
     candidates;
  3. break a tie (mean ranks within 0.05) by the higher mean normalised HV.

With --write FILE --key K the choice is stored in FILE (created if absent):
for ALNS+R#k / NSGA-II+R#k the parameters of row k under K, for the hybrid
the method name under K.

    python tune.py results_calib --prefix "ALNS+R#" --write frozen.json --key ALNS+R
    python tune.py results_calib --prefix "NSGA-II+R#" --write frozen.json --key NSGA-II+R
    python tune.py results_calib --variants CPM+ALNS+R:union@0.02,CPM+ALNS+R:union@0.05 \
        --require-z2 --write frozen.json --key hybrid
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from analyze_fair import load, runs_table
from params import parse_variant, row_params

TIE = 0.05
ORDER = {"verify": 0, "small": 1, "medium": 2, "large": 3, "xlarge": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--variants", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--require-z2", action="store_true")
    ap.add_argument("--write", default="")
    ap.add_argument("--key", default="")
    a = ap.parse_args()

    rows = load(a.folder)
    if a.variants:
        wanted = [v for v in a.variants.split(",") if v]
    elif a.prefix:
        wanted = sorted({r["method"] for r in rows
                         if r["method"].startswith(a.prefix)})
    else:
        raise SystemExit("give --variants or --prefix")
    present = {r["method"] for r in rows}
    missing = [v for v in wanted if v not in present]
    if missing:
        raise SystemExit(f"not in {a.folder}: {missing}")
    rows = [r for r in rows if r["method"] in wanted]
    if {r["replication"] for r in rows} - {2, 3}:
        print("WARNING: replications other than 2 and 3 found; the "
              "calibration set is replications 2 and 3.\n")

    frames = []
    for scale in ORDER:
        sub = [r for r in rows if r["scale"] == scale]
        if sub:
            frames.append(runs_table(sub)[0])
    runs = pd.concat(frames)
    runs = runs[runs.method.isin(wanted)].copy()
    runs["variant"] = pd.Categorical(runs.method, categories=wanted,
                                     ordered=True)
    runs["igd_rank"] = runs.groupby("instance").igd_plus.rank()
    cols = ["igd_rank", "hv_norm", "igd_plus", "z1_min", "z2_opt"]

    by_scale = (runs.groupby(["scale", "variant"], observed=True)[cols]
                .mean().reset_index()
                .assign(o=lambda t: t.scale.map(ORDER))
                .sort_values(["o", "variant"]).drop(columns="o")
                .set_index(["scale", "variant"]))
    print("Per scale (means over calibration instances):")
    print(by_scale.round(3).to_string())
    overall = runs.groupby("variant", observed=True)[cols].mean()
    print("\nAll calibration instances "
          f"({runs.instance.nunique()} instances):")
    print(overall.round(3).to_string())

    ok = overall
    if a.require_z2:
        ok = overall[overall.z2_opt >= 1.0 - 1e-9]
        if ok.empty:
            raise SystemExit("\nNo candidate attains the Z2 bound on every "
                             "instance.")
    best = ok.igd_rank.min()
    tied = ok[ok.igd_rank <= best + TIE]
    choice = str(tied.hv_norm.idxmax())
    print(f"\nRule: {'Z2 bound on every instance -> ' if a.require_z2 else ''}"
          f"lowest mean IGD+ rank (ties within {TIE}) -> highest HV")
    print(f"Chosen: {choice}")
    if len(tied) > 1:
        print(f"  (tied on rank: {[str(v) for v in tied.index]}; "
              f"chosen by HV)")

    tag = a.key or "choice"
    by_scale.reset_index().to_csv(
        os.path.join(a.folder, f"calibration_{tag}.csv".replace("+", "p")),
        index=False)
    if a.write:
        if not a.key:
            raise SystemExit("--write needs --key")
        frozen = {}
        if os.path.exists(a.write):
            with open(a.write) as fh:
                frozen = json.load(fh)
        v = parse_variant(choice)
        frozen[a.key] = row_params(*v) if v else choice
        with open(a.write, "w") as fh:
            json.dump(frozen, fh, indent=2)
        print(f"written to {a.write}: {a.key} = {frozen[a.key]}")


if __name__ == "__main__":
    main()
