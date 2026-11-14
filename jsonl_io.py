"""Reading results.jsonl files safely, and the suspension check.

A power cut or a crash can leave the last line of a results file cut off
(or filled with zero bytes). A plain json.loads loop then stops every later
step, and the next record appended to the broken line is lost with it.

read_records(path)               valid records; broken lines are skipped and
                                 counted (warning on stderr)
read_records(path, repair=True)  as above, and the file is rewritten without
                                 the broken lines, which are kept in
                                 <path>.bad; records of suspended runs are
                                 moved to <path>.suspended, so that
                                 fair_compare.py runs them again

Suspended run (QC rule, ANALYSIS_PLAN.md, amendment 1): a run whose wall
time exceeds its clock by more than 25% plus 2 s. Every method stops on the
wall clock (measured: at most 0.5% over it), so such a run was interrupted
- the computer slept or the process was stopped - and the clock ran on while
it did not compute. The rule looks at times only, never at results. CPU time
is not used: with more workers than free cores it is below wall time for
every run alike.
"""

from __future__ import annotations

import json
import os
import sys

OVER_RATIO, OVER_SECONDS = 1.25, 2.0


def suspended(r: dict) -> bool:
    w, c = r.get("seconds"), r.get("clock")
    return w is not None and c is not None and w > OVER_RATIO * c + OVER_SECONDS


def read_records(path: str, repair: bool = False, quiet: bool = False):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fh:
        raw = fh.read()
    good, bad = [], []
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            good.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            bad.append(line)
    if bad and not quiet:
        print(f"WARNING: {path}: {len(bad)} broken line(s) skipped "
              f"(power cut or crash while writing)", file=sys.stderr)
    if not repair:
        return good
    susp = [r for r in good if suspended(r)]
    keep = [r for r in good if not suspended(r)]
    if bad:
        with open(path + ".bad", "ab") as fh:
            for line in bad:
                fh.write(line + b"\n")
    if susp:
        with open(path + ".suspended", "a") as fh:
            for r in susp:
                fh.write(json.dumps(r) + "\n")
        print(f"{path}: {len(susp)} suspended run(s) (wall time > "
              f"{OVER_RATIO:g} x clock + {OVER_SECONDS:g} s) moved to "
              f"{path}.suspended; they will be run again", flush=True)
    if bad or susp or (raw and not raw.endswith(b"\n")):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if bad:
            print(f"{path}: rewritten without the broken line(s); "
                  f"they are kept in {path}.bad", flush=True)
    return keep


def append_record(path: str, rec: dict) -> None:
    """Append one record and force it to disk, so that a power cut loses at
    most the line being written."""
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    # python jsonl_io.py results_calib/results.jsonl [--repair]
    p = sys.argv[1]
    rs = read_records(p, repair="--repair" in sys.argv)
    n_s = sum(suspended(r) for r in rs)
    print(f"{p}: {len(rs)} valid records, {n_s} suspended among them")
    for lab, num, den in (("wall/clock", "seconds", "clock"),
                          ("CPU/wall", "cpu_seconds", "seconds")):
        q = sorted(r[num] / r[den] for r in rs
                   if r.get(num) is not None and r.get(den) and r[den] > 1.0)
        if q:
            k = len(q)
            print(f"{lab:<10} (runs over 1 s): min {q[0]:.3f}  "
                  f"median {q[k // 2]:.3f}  99% {q[min(k - 1, 99 * k // 100)]:.3f}"
                  f"  max {q[-1]:.3f}")
