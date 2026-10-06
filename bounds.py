"""
A combinatorial lower bound on Z1 (total tardiness).

Why it exists. CP-SAT proves optimality of Z1 only on very small instances
(21 of 45 at n = 12, 3 of 45 at n = 16, none at n = 20 in the verification
run), and by n = 30 its bound on Z1 is uninformative. This counting bound is
valid at every scale; it is loose, and it is reported as such.

THE ARGUMENT. Three facts hold in every feasible schedule, whatever its
batching, and none of them assume contiguity or a fixed batch count.

  (1) Depth.  A batch that is the d-th on its machine cannot finish before
      d * p_min. With m machines at most k*m batches sit at depth <= k, so
      ordering all batches by completion time, the k-th satisfies
          L_k = ceil(k / m) * p_min.

  (2) Capacity.  The first k batches hold at most k*Q kg between them. So the
      most jobs that can possibly complete within the first k batches is
          N(k) = max { t : sum of the t smallest job sizes <= k*Q }.
      Inverting, the i-th job to complete cannot be in a batch ranked earlier
      than K(i) = min { k : N(k) >= i }.

  (3) Matching.  Pairing the completion floors L_{K(i)} (ascending) with the
      due dates (ascending) minimises sum of max(0, L - d); the cost matrix is
      Monge, so the identity permutation is optimal. Any other pairing costs
      more, so the identity gives the floor.

Together:  Z1 >= sum_i max(0, L_{K(i)} - d_(i)).

TWO EARLIER ATTEMPTS FAILED, both caught by systematic checking rather than by
inspection, which is why `verify_validity` ships with the module:

  attempt 1  used one slot per job, inflating ranks from ~n/3 to n. The
             "bound" exceeded real solutions in 90 of 90 cases.
  attempt 2  fixed the slot count but filled slots greedily, advancing to the
             next slot whenever a job did not fit. Greedy solves the
             relaxation only approximately, and an approximate solution of a
             relaxation is an UPPER bound on it - which may sit above the true
             optimum. It exceeded real solutions in 2 of 90 cases. A bound
             that fails twice is not a bound.

The counting argument above avoids both traps: nothing is packed, so there is
no greedy step to be suboptimal, and the ranks come from a capacity count that
cannot overstate them.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from core import Instance


def completion_floor(inst: Instance, n_ranks: int) -> List[float]:
    """L_k for k = 1..n_ranks: the earliest the k-th batch can possibly end."""
    p_min = min(inst.proc.values())
    return [math.ceil(k / inst.machines) * p_min
            for k in range(1, n_ranks + 1)]


def min_rank_per_job(inst: Instance) -> List[int]:
    """K(i) for i = 1..n: the earliest batch rank the i-th job can occupy.

    Derived from capacity alone: the first k batches carry at most k*Q kg, so
    only the jobs whose sizes fit in that budget can be among them.
    """
    sizes = sorted(j.size for j in inst.jobs)
    n = len(sizes)
    prefix = [0.0]
    for s in sizes:
        prefix.append(prefix[-1] + s)

    ranks: List[int] = []
    k = 1
    for i in range(1, n + 1):
        # smallest k with prefix[i] <= k*Q
        k = max(k, math.ceil(prefix[i] / inst.capacity - 1e-12))
        ranks.append(max(1, k))
    return ranks


def z1_lower_bound(inst: Instance) -> float:
    """Lower bound on total tardiness. Valid for the unrestricted problem."""
    ranks = min_rank_per_job(inst)
    L = completion_floor(inst, max(ranks))
    dues = sorted(j.due for j in inst.jobs)
    total = 0.0
    for i, d in enumerate(dues):
        total += max(0.0, L[ranks[i] - 1] - d)
    return round(total, 6)


def gap_to_bound(z1: float, lb: float) -> float:
    """How far a solution sits above the bound, as a percentage."""
    if lb <= 1e-9:
        return float("nan")
    return round(100.0 * (z1 - lb) / lb, 2)


# --------------------------------------------------------------------------


def verify_validity(inst: Instance, z1_values: Sequence[float]
                    ) -> Tuple[bool, float]:
    """The bound must not exceed ANY feasible solution's Z1. One violation
    invalidates it; this is checked, never assumed."""
    lb = z1_lower_bound(inst)
    return (all(lb <= z + 1e-6 for z in z1_values), lb)
