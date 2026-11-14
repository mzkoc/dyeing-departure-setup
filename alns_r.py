"""
ALNS+R: the ALNS benchmark given the batch-content step that CPM has and the
benchmarks of the original study did not.

Reviewer 2 (comment 4) asked for a comparison in which the metaheuristics are
not denied what the proposed method uses. In the original study NSGA-II and
ALNS work on the fixed FFD/BFD batches of CPM's first pass and never change
which jobs share a batch, whereas CPM redistributes jobs between the batches
of a group against observed completion times (L1) and assigns batches to
positions exactly (L4). This module adds both to ALNS as an improvement step
R, so that the remaining difference between CPM and ALNS+R is the search over
supports and sequences: the ladder against ALNS.

R acts on one schedule and keeps its timeline. The family on every position
of every machine stays where it is, so by A2 and A3 every completion time and
every setup stays where it is, and Z2 cannot change. Within that timeline:

  1. each (family, customer) group redistributes its jobs among its own
     batches, holding the batch count, against the sorted completion times
     its batches have in this schedule - the same `layer1.redistribute` CPM
     calls, with the schedule's own times instead of the previous pass's;
  2. the batches of each family are assigned to that family's positions by
     the Hungarian method - the same `layer234.assign_batches` CPM calls.

Both the redistributed batches and the unchanged ones are assigned, and the
better result is kept, so R never worsens Z1. The two steps are repeated
while Z1 improves, up to R_ROUNDS times, which mirrors CPM's outer feedback
loop. Every result goes through `core.evaluate_schedule` and is counted as an
evaluation.

R is applied to every candidate that enters the Pareto archive, before it is
added, and the search continues from the improved solution with its batches.
Nothing else in ALNS is changed: operators, weights, acceptance and segment
restarts are those of `benchmarks.alns`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from benchmarks import (DESTROY, REPAIR, Budget, Solution, _insert, _remove,
                        dominates, edd_key, evaluate, filter_front,
                        greedy_schedule, initial_batches)
from core import Batch, Instance, Schedule, evaluate_schedule
from layer1 import redistribute
from layer234 import assign_batches

R_ROUNDS = 3


def _pools(inst: Instance, batches: List[Batch], sched: Schedule,
           completion: Dict[int, float]
           ) -> Dict[str, List[Tuple[int, int, float]]]:
    """Positions of each family in the schedule: (machine, slot, completion)."""
    by_id = {b.id: b for b in batches}
    pools: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    for m, seq in sched.items():
        for slot, bid in enumerate(seq):
            pools[by_id[bid].family].append((m, slot, completion[bid]))
    return pools


def _redistributed(inst: Instance, batches: List[Batch],
                   completion: Dict[int, float]) -> List[Batch]:
    groups: Dict[Tuple[str, str], List[Batch]] = defaultdict(list)
    for b in batches:
        groups[(b.family, b.customer)].append(b)
    out: List[Batch] = []
    for (fam, cust), bs in sorted(groups.items()):
        bins = [[inst.job(j) for j in b.jobs] for b in bs]
        est = sorted(completion[b.id] for b in bs)
        new = redistribute(bins, inst.capacity, est)
        if len(new) != len(bs) or any(not run for run in new):
            raise RuntimeError("redistribution changed the batch count")
        for b, run in zip(bs, new):
            out.append(Batch(id=b.id, jobs=tuple(sorted(j.id for j in run)),
                             family=fam, customer=cust))
    return out


def improve(inst: Instance, sol: Solution, budget: Budget,
            rounds: int = R_ROUNDS) -> Solution:
    """R: redistribution plus exact assignment on the schedule's timeline."""
    best = sol
    for _ in range(rounds):
        ev = evaluate_schedule(inst, best.batches, best.schedule,
                               strict=False)
        pools = _pools(inst, best.batches, best.schedule, ev.completion)
        cands: List[Solution] = []
        for bset in (best.batches,
                     _redistributed(inst, best.batches, ev.completion)):
            sched, _ = assign_batches(inst, bset, pools)
            s = evaluate(inst, bset, sched, budget)
            if s is not None:
                cands.append(s)
        if not cands:
            break
        c = min(cands, key=lambda s: s.z1)
        if c.z1 < best.z1 - 1e-9:
            if abs(c.z2 - best.z2) > 1e-6:
                raise RuntimeError("R changed Z2; the timeline was not kept")
            best = c
        else:
            break
    return best


def r_on_front(inst: Instance, front: List[Solution], budget: Budget,
               rounds: int = R_ROUNDS):
    """R on every point of a front, least tardiness first (if the budget
    runs out, the points left unimproved are the high-tardiness ones).
    Used by CPM+R and [7]+R. Returns (front, info)."""
    out, left = [], 0
    for s in sorted(front, key=lambda s: s.z1):
        if budget.exhausted:
            out.append(s)
            left += 1
        else:
            out.append(improve(inst, s, budget, rounds=rounds))
    return filter_front(out), {"r_rounds": rounds,
                               "r_exhausted": bool(budget.exhausted),
                               "r_unimproved": left}


def alns_r(inst: Instance, budget: Budget, *, seed: int = 0,
           segment: int = 25, decay: float = 0.8,
           init: List[Solution] | None = None,
           start_from_init: bool = True,
           dmax: float = 0.35, r_rounds: int = R_ROUNDS) -> List[Solution]:
    """`benchmarks.alns` with R applied to every archive entrant.

    `init` (used by the hybrid in hybrid.py) adds solutions to the starting
    archive: each is improved by R first. With `start_from_init` the search
    starts from the one with the least tardiness; without it, it starts as
    `benchmarks.alns` does, from the greedy EDD schedule on FFD/BFD batches,
    and the added solutions only sit in the archive.

    `dmax` is the largest share of batches a destroy move removes (the
    smallest is 0.1), `r_rounds` the number of rounds of R; with the defaults
    the run is that of the study before calibration.
    """
    rng = random.Random(seed)
    batches = initial_batches(inst, rng)
    sched = greedy_schedule(inst, batches, edd_key(inst))
    cur = evaluate(inst, batches, sched, budget)
    if cur is None:
        return list(init or [])
    cur = improve(inst, cur, budget, r_rounds)
    archive: List[Solution] = [cur]
    if init:
        seeded = [improve(inst, s, budget, r_rounds) for s in init]
        archive = filter_front(archive + seeded)
        if start_from_init:
            cur = min(seeded, key=lambda s: s.z1)

    wd = {d: 1.0 for d in DESTROY}
    wr = {r: 1.0 for r in REPAIR}
    sd = {d: [0.0, 1] for d in DESTROY}
    sr = {r: [0.0, 1] for r in REPAIR}
    n_placed = max(1, len(batches))
    it = 0

    while not budget.exhausted:
        it += 1
        d = rng.choices(DESTROY, weights=[wd[x] for x in DESTROY])[0]
        r = rng.choices(REPAIR, weights=[wr[x] for x in REPAIR])[0]
        n_placed = max(1, len(cur.batches))
        k = max(1, int(n_placed * rng.uniform(0.1, dmax)))

        trial = {m: list(seq) for m, seq in cur.schedule.items()}
        trial, removed = _remove(inst, trial, cur.batches, d, k, rng)
        if not removed:
            continue
        trial = _insert(inst, trial, cur.batches, removed, r, rng)
        cand = evaluate(inst, cur.batches, trial, budget)
        if cand is None:
            continue

        score = 0.0
        if any(dominates(cand, a) for a in archive):
            score = 4.0
        elif not any(dominates(a, cand) for a in archive):
            score = 2.0
        if score:
            cand = improve(inst, cand, budget, r_rounds)  # the only addition
            archive = filter_front(archive + [cand])

        w = rng.random()

        def sc(s: Solution) -> float:
            return w * s.z1 + (1 - w) * s.z2 * 100.0
        if score or sc(cand) < sc(cur):
            cur = cand
            if not score:
                score = 1.0

        sd[d][0] += score
        sd[d][1] += 1
        sr[r][0] += score
        sr[r][1] += 1
        if it % segment == 0:
            for x in DESTROY:
                wd[x] = decay * wd[x] + (1 - decay) * (sd[x][0] / sd[x][1])
                wd[x] = max(wd[x], 0.05)
                sd[x] = [0.0, 1]
            for x in REPAIR:
                wr[x] = decay * wr[x] + (1 - decay) * (sr[x][0] / sr[x][1])
                wr[x] = max(wr[x], 0.05)
                sr[x] = [0.0, 1]
            cur = rng.choice(archive)

    return filter_front(archive)
