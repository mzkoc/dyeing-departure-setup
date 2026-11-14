"""Measure the clock factor for fair_compare.py on this machine.

The clocks of the equal wall-clock check are the mean CPU times of CPM in the
main study, which ran on an Intel Core i9-13900H with four scales as four
concurrent processes. On another machine, or with another number of
concurrent processes, the same clock buys a different amount of work. This
script reruns CPM with the main-study budget (10 000 evaluations, which
reproduces exactly) on instances of one scale, with --workers processes at
once, and compares the time with the time stored in results/part_*/.

The median ratio is the --factor to pass to fair_compare.py with the same
number of workers. Z1 must also match the stored value exactly; if it does
not, the environment does not reproduce the study and the run stops there.

    python calibrate.py --scale medium --workers 12
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import statistics
import time
from concurrent.futures import ProcessPoolExecutor


def run(scale, ci, tau, rep):
    from cpm import cpm_report
    from instances import build_matrix, generate_instance
    cell = [c for c in build_matrix() if c.scale == scale][ci]
    inst = generate_instance(cell, tau, rep)
    t0 = time.time()
    front, _ = cpm_report(inst)
    return inst.name, time.time() - t0, min(p.z1 for p in front)


def main():
    import pandas as pd
    from instances import build_matrix

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0,
                    help="instances to run (default: one per worker)")
    a = ap.parse_args()

    stored = pd.concat(pd.read_csv(p) for p in
                       glob.glob(os.path.join("results", "part_*",
                                              "results.csv")))
    stored = stored[stored.method == "CPM"].set_index("instance")

    cells = [c for c in build_matrix() if c.scale == a.scale]
    jobs = [(a.scale, ci, tau, rep) for rep in range(10)
            for tau in (0.3, 0.5, 0.7) for ci in range(len(cells))]
    jobs = jobs[: (a.n or a.workers)]

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(run, *zip(*jobs)))

    ratios = []
    for name, secs, z1 in res:
        ref = stored.loc[name]
        same = abs(z1 - ref.z1_min) < 1e-6
        ratios.append(secs / ref.seconds)
        print(f"{name:<28} {secs:8.1f} s  stored {ref.seconds:8.1f} s  "
              f"ratio {secs / ref.seconds:5.2f}  Z1 {'matches' if same else 'DIFFERS'}")
        if not same:
            raise SystemExit("Z1 differs from the stored run: the environment "
                             "does not reproduce the study.")
    f = statistics.median(ratios)
    print(f"\nmedian ratio {f:.2f} -> use --factor {f:.2f} "
          f"with --workers {a.workers}")


if __name__ == "__main__":
    main()
