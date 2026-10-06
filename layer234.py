"""
CPM Layers 2-4: the scheduling half.

  L2a  colour -> machine SUPPORT      drives Z2 (closed form, P1)
  L2b  colour -> machine COUNTS       drives Z1
  L3   block order per machine        drives Z1
  L4   batch -> position assignment   drives Z1, solved EXACTLY (P3)

Two refinements that came out of testing, both recorded here because the
design doc claimed otherwise:

1. Block order and Z2. Z2 on a machine depends ONLY on which colour is last,
   not on the order of the rest:

       order   ABC   ACB   BAC   BCA   CAB   CBA       sigma A=.2 B=.5 C=.35
       Z2       .70   .55   .70   .85   .55   .85

   Fixing the last block to argmax(sigma) attains P1's closed form, and every
   permutation of the remaining blocks is then free of Z2 consequences. So L3
   permutes only the non-last blocks and the decomposition stays clean.

2. Support enumeration is NOT exhaustive in general. The doc said "full
   enumeration for |F| <= 8, m <= 12"; assigning |F| colours to m machines is
   m^|F| before splits, i.e. 12^8 = 4.3e8 for the largest cell. Exhaustive
   enumeration is kept only where it is cheap, and it is used to VALIDATE the
   guided search rather than to replace it - the same pattern as the P1
   brute-force check.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import Batch, Instance, Schedule, evaluate_schedule

Colour = str
MachineIdx = int

# colour -> machines it occupies
Support = Dict[Colour, Tuple[MachineIdx, ...]]
# (colour, machine) -> how many batches of that colour sit there
Counts = Dict[Tuple[Colour, MachineIdx], int]
# machine -> the colours in block order
BlockOrder = Dict[MachineIdx, Tuple[Colour, ...]]

MAX_BLOCK_PERM = 6          # permute exhaustively up to this many free blocks
MAX_SUPPORT_ENUM = 200_000  # refuse exhaustive enumeration beyond this
MAX_JOINT_ORDERS = 64       # take block orders exactly below this, descend above
INF = float("inf")


# --------------------------------------------------------------------------
# Z2 from a support (P1 closed form)
# --------------------------------------------------------------------------


def machines_of(support: Support) -> Dict[MachineIdx, List[Colour]]:
    out: Dict[MachineIdx, List[Colour]] = defaultdict(list)
    for f, ms in support.items():
        for m in ms:
            out[m].append(f)
    return {m: sorted(v) for m, v in out.items()}


def z2_of_support(inst: Instance, support: Support) -> float:
    """P1: sum over machines of (sum of sigma over its colours - max sigma)."""
    total = 0.0
    for colours in machines_of(support).values():
        vals = [inst.sigma[f] for f in colours]
        total += sum(vals) - max(vals)
    return round(total, 6)


def last_block_colour(inst: Instance, colours: Sequence[Colour]) -> Colour:
    """The colour that must finish the machine to attain the closed form."""
    return max(colours, key=lambda f: (inst.sigma[f], f))


# --------------------------------------------------------------------------
# L2b: batch counts over the support
# --------------------------------------------------------------------------


def initial_counts(inst: Instance, support: Support,
                   n_batches: Dict[Colour, int]) -> Counts:
    """Spread each colour's batches over its machines, levelling machine load."""
    counts: Counts = {}
    load = {m: 0.0 for m in range(inst.machines)}
    for f, ms in support.items():
        k = n_batches[f]
        for m in ms:
            counts[(f, m)] = 1            # every support machine needs >= 1
            load[m] += inst.proc[f]
        for _ in range(k - len(ms)):
            m = min(ms, key=lambda x: (load[x], x))
            counts[(f, m)] += 1
            load[m] += inst.proc[f]
    return counts


def counts_neighbours(support: Support, counts: Counts
                      ) -> List[Counts]:
    """Move one batch of one colour to another of that colour's machines."""
    out: List[Counts] = []
    for f, ms in support.items():
        if len(ms) < 2:
            continue
        for a in ms:
            if counts[(f, a)] <= 1:
                continue                  # would vacate a support machine
            for b in ms:
                if a == b:
                    continue
                nc = dict(counts)
                nc[(f, a)] -= 1
                nc[(f, b)] += 1
                out.append(nc)
    return out


# --------------------------------------------------------------------------
# L3 + timing: positions
# --------------------------------------------------------------------------


def block_orders(inst: Instance, support: Support) -> Dict[MachineIdx,
                                                           List[Tuple[Colour, ...]]]:
    """Candidate block orders per machine, last block pinned to argmax sigma."""
    out: Dict[MachineIdx, List[Tuple[Colour, ...]]] = {}
    for m, colours in machines_of(support).items():
        last = last_block_colour(inst, colours)
        free = [f for f in colours if f != last]
        if len(free) <= MAX_BLOCK_PERM:
            perms = [tuple(p) + (last,) for p in itertools.permutations(free)]
        else:
            # too many blocks to enumerate: order by the colour's tightest due
            perms = [tuple(sorted(free)) + (last,)]
        out[m] = perms
    return out


def positions_of(inst: Instance, counts: Counts,
                 order: BlockOrder) -> Tuple[Dict[Colour,
                                                  List[Tuple[MachineIdx, int, float]]],
                                             float]:
    """Completion time of every position, plus the Z2 actually incurred.

    A position is (machine, slot index on that machine, completion time).
    Timing is fully determined by counts and block order, because every batch
    of a colour has the same processing time (A2). This is P3: by the time L4
    runs, the times are fixed and only the matching is open.
    """
    pools: Dict[Colour, List[Tuple[MachineIdx, int, float]]] = defaultdict(list)
    z2 = 0.0
    for m, colours in order.items():
        t = 0.0
        prev: Colour | None = None
        slot = 0
        for f in colours:
            if prev is not None and prev != f:
                s = inst.sigma[prev]
                t += s
                z2 += s
            for _ in range(counts.get((f, m), 0)):
                t += inst.proc[f]
                pools[f].append((m, slot, t))
                slot += 1
            prev = f
    return dict(pools), round(z2, 6)


# --------------------------------------------------------------------------
# L4: exact batch -> position assignment (P3)
# --------------------------------------------------------------------------


_PAD = 1e18          # padded due date: relu(C - PAD) is always 0


class Assigner:
    """Vectorised, memoised L4.

    The per-colour due dates never change while the support search runs, so
    they are padded into one array per colour up front. Cost of batch i at a
    position completing at C is sum_j relu(C - d_ij), which is then one
    broadcast instead of a Python double loop - profiling put 56 of 60 seconds
    in that loop, against 1.4 seconds for the Hungarian solver itself.

    Results are memoised on the tuple of completion times, because the support
    and counts searches revisit the same colour timings constantly.
    """

    def __init__(self, inst: Instance, batches: Sequence[Batch]):
        self.inst = inst
        self.batch_ids: Dict[Colour, List[int]] = {}
        self.dues: Dict[Colour, np.ndarray] = {}
        by_colour: Dict[Colour, List[Batch]] = defaultdict(list)
        for b in batches:
            by_colour[b.family].append(b)
        for f, bs in by_colour.items():
            width = max(len(b.jobs) for b in bs)
            arr = np.full((len(bs), width), _PAD)
            for i, b in enumerate(bs):
                for j, jid in enumerate(b.jobs):
                    arr[i, j] = inst.job(jid).due
            self.batch_ids[f] = [b.id for b in bs]
            self.dues[f] = arr
        self._memo: Dict[Tuple[Colour, Tuple[float, ...]],
                         Tuple[Tuple[int, ...], float]] = {}

    def _solve_colour(self, f: Colour,
                      times: Tuple[float, ...]) -> Tuple[Tuple[int, ...], float]:
        key = (f, times)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        D = self.dues[f]                                  # (nb, width)
        C = np.asarray(times)                             # (npos,)
        cost = np.maximum(0.0, C[None, :, None] - D[:, None, :]).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        z1 = float(cost[rows, cols].sum())
        mapping = tuple(int(c) for c in cols[np.argsort(rows)])
        self._memo[key] = (mapping, z1)
        return mapping, z1

    def assign(self, pools: Dict[Colour,
                                 List[Tuple[MachineIdx, int, float]]]
               ) -> Tuple[Schedule, float]:
        placed: Dict[MachineIdx, List[Tuple[int, int]]] = defaultdict(list)
        z1 = 0.0
        for f, ids in self.batch_ids.items():
            pool = pools.get(f, [])
            if len(pool) != len(ids):
                raise ValueError(
                    f"colour {f}: {len(ids)} batches but {len(pool)} positions")
            times = tuple(round(c, 9) for _, _, c in pool)
            mapping, cost = self._solve_colour(f, times)
            z1 += cost
            for i, k in enumerate(mapping):
                m, slot, _ = pool[k]
                placed[m].append((slot, ids[i]))
        schedule: Schedule = {m: [] for m in range(self.inst.machines)}
        for m, items in placed.items():
            schedule[m] = [bid for _, bid in sorted(items)]
        return schedule, round(z1, 6)


def assign_batches(inst: Instance, batches: Sequence[Batch],
                   pools: Dict[Colour, List[Tuple[MachineIdx, int, float]]]
                   ) -> Tuple[Schedule, float]:
    """Convenience wrapper. Prefer reusing one Assigner across a search."""
    return Assigner(inst, batches).assign(pools)


# --------------------------------------------------------------------------
# Putting L2b + L3 + L4 together for a fixed support
# --------------------------------------------------------------------------


def solve_support(inst: Instance, batches: Sequence[Batch],
                  support: Support, *, counts_passes: int = 12,
                  order_rounds: int = 2,
                  assigner: "Assigner | None" = None
                  ) -> Tuple[Schedule, float, float]:
    """Best (Z1, Z2) we can reach for this colour->machine support.

    Z2 is already pinned by the support (P1); the search is entirely over Z1:
    batch counts (L2b), block order (L3), then the exact matching (L4).

    Block orders are optimised by COORDINATE DESCENT, one machine at a time,
    rather than over the cartesian product across machines. The product is
    prod_m (k_m - 1)! and blows up with the machine count; machines are only
    weakly coupled - through a colour whose positions span several of them -
    so descending one machine at a time costs sum_m instead of prod_m and
    loses very little.
    """
    n_batches: Dict[Colour, int] = defaultdict(int)
    for b in batches:
        n_batches[b.family] += 1

    for f, ms in support.items():
        if n_batches[f] < len(ms):
            raise ValueError(f"colour {f} has fewer batches than machines")

    if assigner is None:
        assigner = Assigner(inst, batches)

    orders = block_orders(inst, support)
    machines = sorted(orders)
    counts = initial_counts(inst, support, n_batches)

    def evaluate(c: Counts, order: BlockOrder) -> Tuple[Schedule, float, float]:
        pools, z2 = positions_of(inst, c, order)
        sched, z1 = assigner.assign(pools)
        return sched, z1, z2

    joint = 1
    for m in machines:
        joint *= len(orders[m])

    def best_for_counts(c: Counts) -> Tuple[Schedule, float, float]:
        if joint <= MAX_JOINT_ORDERS:
            # Small enough to take exactly: no coupling is missed.
            best: Tuple[Schedule | None, float, float] = (None, INF, INF)
            for combo in itertools.product(*(orders[m] for m in machines)):
                s2, v1, v2 = evaluate(c, dict(zip(machines, combo)))
                if v1 < best[1]:
                    best = (s2, v1, v2)
            return best                                   # type: ignore

        order: BlockOrder = {m: orders[m][0] for m in machines}
        sched, z1, z2 = evaluate(c, order)
        for _ in range(order_rounds):
            improved = False
            for m in machines:
                if len(orders[m]) < 2:
                    continue
                keep = order[m]
                for cand in orders[m]:
                    if cand == keep:
                        continue
                    order[m] = cand
                    s2, v1, v2 = evaluate(c, order)
                    if v1 < z1 - 1e-9:
                        keep, sched, z1, z2 = cand, s2, v1, v2
                        improved = True
                order[m] = keep
            if not improved:
                break
        return sched, z1, z2

    sched, z1, z2 = best_for_counts(counts)
    for _ in range(counts_passes):
        # Best improvement, not first: the evaluator is now cheap enough that
        # scanning the whole neighbourhood pays for itself in solution quality.
        best_nc, best_res, best_z1 = None, None, z1
        for nc in counts_neighbours(support, counts):
            res = best_for_counts(nc)
            if res[1] < best_z1 - 1e-9:
                best_nc, best_res, best_z1 = nc, res, res[1]
        if best_nc is None:
            break
        counts = best_nc
        sched, z1, z2 = best_res                          # type: ignore
    return sched, z1, z2


# --------------------------------------------------------------------------
# L2a: supports
# --------------------------------------------------------------------------


def minimal_support(inst: Instance, colours: Sequence[Colour]) -> Support:
    """The Z2-optimal support: every colour whole, on its own machine where
    possible, with the m largest-sigma colours occupying distinct machines
    (P1, Corollary 3).

    `colours` must be the colours that actually have batches, not the whole
    palette: a colour with no jobs has no batches to place, and asking the
    layers to schedule it raises.
    """
    ordered = sorted(colours, key=lambda f: (-inst.sigma[f], f))
    support: Support = {}
    for i, f in enumerate(ordered):
        support[f] = (i % inst.machines,)
    return support


def balanced_support(inst: Instance, colours: Sequence[Colour],
                     n_batches: Dict[Colour, int]) -> Support:
    """The Z1 end of the front: spread every colour as widely as its batch
    count allows. Maximum Z2, minimum load imbalance. Seeding the Pareto
    search from both ends stops it converging into one corner."""
    support: Support = {}
    for f in colours:
        k = min(inst.machines, n_batches.get(f, 1))
        support[f] = tuple(range(k))
    return support


def enumerate_supports(inst: Instance, colours: Sequence[Colour],
                       max_machines_per_colour: int = 1) -> List[Support]:
    """Every assignment of colours to machines. Only for small cells.

    Cost is m^|F| with no splits, more with them; refuses above
    MAX_SUPPORT_ENUM. Used to validate the guided search, not to replace it.
    """
    m = inst.machines
    if max_machines_per_colour == 1:
        size = m ** len(colours)
        if size > MAX_SUPPORT_ENUM:
            raise ValueError(f"{size} supports is too many to enumerate")
        return [dict(zip(colours, [(x,) for x in combo]))
                for combo in itertools.product(range(m), repeat=len(colours))]

    options: List[List[Tuple[int, ...]]] = []
    for _ in colours:
        opts: List[Tuple[int, ...]] = []
        for k in range(1, max_machines_per_colour + 1):
            opts.extend(itertools.combinations(range(m), k))
        options.append(opts)
    size = 1
    for o in options:
        size *= len(o)
        if size > MAX_SUPPORT_ENUM:
            raise ValueError(f"{size}+ supports is too many to enumerate")
    return [dict(zip(colours, combo)) for combo in itertools.product(*options)]


def support_neighbours(inst: Instance, support: Support,
                       n_batches: Dict[Colour, int]) -> List[Support]:
    """Move a colour to another machine, split it onto one more, or merge it
    back. Splitting is what buys Z1 at a cost of exactly +sigma_f (P1,
    Corollary 4), so it must be in the neighbourhood."""
    out: List[Support] = []
    for f, ms in support.items():
        for m in range(inst.machines):
            if m in ms:
                continue
            if len(ms) == 1:                        # move
                nb = dict(support)
                nb[f] = (m,)
                out.append(nb)
            if len(ms) < n_batches[f]:              # split
                nb = dict(support)
                nb[f] = tuple(sorted(ms + (m,)))
                out.append(nb)
        if len(ms) > 1:                             # merge
            for drop in ms:
                nb = dict(support)
                nb[f] = tuple(x for x in ms if x != drop)
                out.append(nb)
    return out
