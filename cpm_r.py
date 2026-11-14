"""CPM+R: CPM with the same improvement step R that ALNS+R uses.

The pilot showed ALNS+R ahead of CPM. ALNS+R applies R (redistribution against
the schedule's own completion times, then exact assignment) to every archive
entrant, whereas CPM redistributes once per outer pass against the times of
its best-Z1 point only. To make the comparison symmetric - the ladder against
ALNS, both with R - CPM is run on most of the clock and R is then applied to
every point of its front within the rest. R keeps each point's timeline, so
the Z2 levels of the ladder, and the Z2 optimum among them, are unchanged.
"""

from __future__ import annotations

import time
from typing import List

from alns_r import R_ROUNDS, r_on_front
from benchmarks import Budget, Solution
from cpm import cpm_report

R_SHARE = 0.15      # share of the clock reserved for R on the front


def cpm_r(inst, seconds: float, rounds: int = R_ROUNDS,
          return_info: bool = False):
    """rounds: as ALNS+R's r_rounds (fair_compare.py passes the frozen value;
    the development-set runs of v6-v8 used the default, 3)."""
    t0 = time.time()
    front, diag = cpm_report(inst, variant="cpm", eval_budget=10 ** 9,
                             time_limit=seconds * (1 - R_SHARE),
                             safety_seconds=seconds * (1 - R_SHARE) * 1.5)
    left = max(seconds - (time.time() - t0), 0.01)
    b = Budget(10 ** 9, seconds=left)
    sols: List[Solution] = [Solution(p.batches, p.schedule, p.z1, p.z2)
                            for p in front]
    out, info = r_on_front(inst, sols, b, rounds)
    if return_info:
        return out, diag["evaluations"] + b.used, info
    return out, diag["evaluations"] + b.used
