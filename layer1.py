"""
CPM Layer 1: batching.

Design note (revised). An earlier version restricted batching to CONTIGUOUS
runs of the EDD-ordered job list, on the argument that EDD-contiguous slicing
is optimal for Z1 when completion times are known. Measurement killed it:

    packing gap   FFD vs ceil(sum kg / Q)          4-10%
    contiguity    EDD-contiguous vs FFD            7-19%   <-- pure loss

Contiguity at the BATCHING level costs 7-19% extra batches, and extra batches
mean extra processing, later completions and higher Z1. It bought nothing:
P1's closed form for Z2 rests on BLOCK contiguity on a machine, which has
nothing to do with how jobs are grouped into batches. So batching is now
unrestricted and contiguity survives only where it earns its place, in L2/L3.

The layer therefore does two things:

  1. Batch count. Pack each (colour, customer) group into as few batches as
     possible - FFD and BFD, take the better. The count is then FIXED, because
     under block contiguity it does not affect Z2 at all (P1, Corollary 5) but
     it does set how many positions each colour needs from L2-L4.

  2. Redistribution. Holding that count, move jobs between the group's batches
     to reduce tardiness against the completion times seen in the previous
     outer iteration. Jobs sharing a batch share its completion time, so the
     point is to put tight-due jobs together in the batches that finish early.

The L1 ablation is variant "blind": identical packing, no redistribution. Same
batch count, same capacity feasibility, same determinism - the only thing that
changes is whether due dates inform the grouping. One variable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from core import Batch, Instance, Job

INF = float("inf")

Group = Tuple[str, str]          # (family, customer)

MAX_LOCAL_SEARCH_PASSES = 30


# --------------------------------------------------------------------------
# Step 1: batch count
# --------------------------------------------------------------------------


def _first_fit_decreasing(jobs: Sequence[Job],
                          capacity: float) -> List[List[Job]]:
    bins: List[List[Job]] = []
    loads: List[float] = []
    for j in sorted(jobs, key=lambda x: (-x.size, x.id)):
        for b, load in enumerate(loads):
            if load + j.size <= capacity + 1e-9:
                bins[b].append(j)
                loads[b] += j.size
                break
        else:
            bins.append([j])
            loads.append(j.size)
    return bins


def _best_fit_decreasing(jobs: Sequence[Job],
                         capacity: float) -> List[List[Job]]:
    bins: List[List[Job]] = []
    loads: List[float] = []
    for j in sorted(jobs, key=lambda x: (-x.size, x.id)):
        best, best_slack = -1, INF
        for b, load in enumerate(loads):
            slack = capacity - load - j.size
            if slack >= -1e-9 and slack < best_slack:
                best, best_slack = b, slack
        if best >= 0:
            bins[best].append(j)
            loads[best] += j.size
        else:
            bins.append([j])
            loads.append(j.size)
    return bins


def pack_group(jobs: Sequence[Job], capacity: float) -> List[List[Job]]:
    """Fewest batches we can find for one group. FFD and BFD, better wins."""
    if any(j.size > capacity + 1e-9 for j in jobs):
        raise ValueError("a job exceeds machine capacity")
    ffd = _first_fit_decreasing(jobs, capacity)
    bfd = _best_fit_decreasing(jobs, capacity)
    return ffd if len(ffd) <= len(bfd) else bfd


def packing_lower_bound(jobs: Sequence[Job], capacity: float) -> int:
    return max(1, math.ceil(sum(j.size for j in jobs) / capacity))


# --------------------------------------------------------------------------
# Step 2: due-date-aware redistribution
# --------------------------------------------------------------------------


def _order_batches_for_times(bins: Sequence[Sequence[Job]]) -> List[int]:
    """Which batch should take which completion time.

    Completion times are handed out in ascending order. Cost of batch B at
    time C is sum(max(0, C - d_j)), which falls faster for batches holding
    tighter due dates, so the batch with the earliest mean due date takes the
    earliest slot. L4 later does this exactly with the Hungarian algorithm;
    here it only has to be a good proxy.
    """
    keys = []
    for idx, b in enumerate(bins):
        mean_due = sum(j.due for j in b) / len(b) if b else INF
        keys.append((mean_due, min((j.due for j in b), default=INF), idx))
    return [idx for _, _, idx in sorted(keys)]


def _assignment_cost(bins: Sequence[Sequence[Job]],
                     est_completion: Sequence[float]) -> float:
    order = _order_batches_for_times(bins)
    total = 0.0
    for slot, bidx in enumerate(order):
        c = est_completion[slot] if slot < len(est_completion) \
            else est_completion[-1]
        for j in bins[bidx]:
            total += max(0.0, c - j.due)
    return total


def redistribute(bins: List[List[Job]], capacity: float,
                 est_completion: Sequence[float]) -> List[List[Job]]:
    """Best-improvement local search over moves and swaps. Deterministic.

    The batch count never changes: a move that would empty a batch is
    rejected, because the number of positions L2-L4 must supply is fixed.
    """
    if not est_completion or len(bins) <= 1:
        return bins

    work = [list(b) for b in bins]
    loads = [sum(j.size for j in b) for b in work]
    best_cost = _assignment_cost(work, est_completion)

    for _ in range(MAX_LOCAL_SEARCH_PASSES):
        best_delta = -1e-9
        best_action = None

        # moves
        for a in range(len(work)):
            if len(work[a]) <= 1:
                continue                      # would empty the batch
            for ji, j in enumerate(work[a]):
                for b in range(len(work)):
                    if a == b or loads[b] + j.size > capacity + 1e-9:
                        continue
                    work[a].pop(ji)
                    work[b].append(j)
                    cost = _assignment_cost(work, est_completion)
                    work[b].pop()
                    work[a].insert(ji, j)
                    delta = cost - best_cost
                    if delta < best_delta:
                        best_delta, best_action = delta, ("move", a, ji, b)

        # swaps
        for a in range(len(work)):
            for b in range(a + 1, len(work)):
                for ia, ja in enumerate(work[a]):
                    for ib, jb in enumerate(work[b]):
                        if ja.size == jb.size:
                            continue          # no capacity change, symmetric
                        if loads[a] - ja.size + jb.size > capacity + 1e-9:
                            continue
                        if loads[b] - jb.size + ja.size > capacity + 1e-9:
                            continue
                        work[a][ia], work[b][ib] = jb, ja
                        cost = _assignment_cost(work, est_completion)
                        work[a][ia], work[b][ib] = ja, jb
                        delta = cost - best_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_action = ("swap", a, ia, b, ib)

        if best_action is None:
            break

        if best_action[0] == "move":
            _, a, ji, b = best_action
            j = work[a].pop(ji)
            work[b].append(j)
            loads[a] -= j.size
            loads[b] += j.size
        else:
            _, a, ia, b, ib = best_action
            ja, jb = work[a][ia], work[b][ib]
            work[a][ia], work[b][ib] = jb, ja
            loads[a] += jb.size - ja.size
            loads[b] += ja.size - jb.size

        best_cost += best_delta

    return work


# --------------------------------------------------------------------------
# Instance-level driver
# --------------------------------------------------------------------------


def group_jobs(inst: Instance) -> Dict[Group, List[Job]]:
    groups: Dict[Group, List[Job]] = defaultdict(list)
    for j in inst.jobs:
        groups[(j.family, j.customer)].append(j)
    return {k: sorted(v, key=lambda j: j.id) for k, v in groups.items()}


def build_batches(inst: Instance,
                  est_completion: Dict[Group, List[float]] | None = None,
                  *, variant: str = "cpm") -> List[Batch]:
    """Run L1 over the whole instance.

    variant "cpm"   packing + due-date-aware redistribution (proposed)
    variant "blind" packing only, no redistribution (L1 ablation)
    """
    batches: List[Batch] = []
    bid = 0
    for (fam, cust), jobs in sorted(group_jobs(inst).items()):
        bins = pack_group(jobs, inst.capacity)
        if variant == "cpm" and est_completion:
            est = est_completion.get((fam, cust))
            if est:
                bins = redistribute(bins, inst.capacity, est)
        for run in bins:
            if not run:
                continue
            batches.append(Batch(id=bid,
                                 jobs=tuple(sorted(j.id for j in run)),
                                 family=fam, customer=cust))
            bid += 1
    return batches


def observed_completions(inst: Instance, batches: Sequence[Batch],
                         completion: Dict[int, float]
                         ) -> Dict[Group, List[float]]:
    """Per group, the completion times its batches received, ascending.

    Feeds the next outer iteration. Sorted rather than matched to specific
    batches, because redistribution will change which jobs sit in which batch;
    what carries over is the SET of times the group can expect.
    """
    per_group: Dict[Group, List[float]] = defaultdict(list)
    for b in batches:
        if b.id in completion:
            per_group[(b.family, b.customer)].append(completion[b.id])
    return {g: sorted(v) for g, v in per_group.items()}


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def packing_quality(inst: Instance) -> Dict[str, float]:
    """How close the packing gets to the lower bound, per instance."""
    lb = got = 0
    for jobs in group_jobs(inst).values():
        lb += packing_lower_bound(jobs, inst.capacity)
        got += len(pack_group(jobs, inst.capacity))
    return {
        "batches_lower_bound": lb,
        "batches_packed": got,
        "pct_above_lb": round(100.0 * (got - lb) / lb, 2) if lb else 0.0,
    }
