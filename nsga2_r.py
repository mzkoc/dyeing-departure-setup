"""
NSGA-II+R: the NSGA-II benchmark given the batch-content step R of alns_r.

NSGA-II decodes a permutation of batch ids against one shared batch set, so R
(which changes batch contents per solution) cannot be written back into a
genome. Two additions make the step available to it, mirroring ALNS+R and
CPM's outer loop:

  1. External archive. Every individual that is not dominated by the archive
     is improved by R and the improved solution enters the archive. The
     population itself evolves as in `benchmarks.nsga2`.
  2. Batch refresh. Every REFRESH generations the shared batch set is
     replaced by the batches of the archive's least-tardiness solution, and
     the population is re-evaluated with them. R keeps the batch ids and the
     batch count of every group, so every genome stays a valid permutation.
     This is the counterpart of CPM's outer loop, which feeds the times of its
     best-Z1 point back into batching.

Selection, operators, the decoder and the lambda gene are those of
`benchmarks.nsga2`; nothing else is changed. The returned front is the
non-dominated set of the external archive and the population archive.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from alns_r import improve
from benchmarks import (PERM_LAMBDA_MAX, PERM_LAMBDA_MIN, Budget, Solution,
                        _crowding, _fast_nondominated_sort, _greedy_decode,
                        _inversion, _log_uniform, _order_crossover, dominates,
                        evaluate, filter_front, initial_batches)
from core import Instance

REFRESH = 10


def nsga2_r(inst: Instance, budget: Budget, *, seed: int = 0,
            pop_size: int = 60, p_cross: float = 0.9,
            p_mut: float = 0.2, refresh: int = REFRESH,
            r_rounds: int = 3) -> List[Solution]:
    rng = random.Random(seed)
    state = {"batches": initial_batches(inst, rng)}
    ids = [b.id for b in state["batches"]]

    def make(perm: List[int], lam: float) -> Solution | None:
        bs = state["batches"]
        return evaluate(inst, bs, _greedy_decode(inst, bs, perm, lam), budget)

    ext: List[Solution] = []

    def offer(s: Solution) -> None:
        nonlocal ext
        if any(dominates(a, s) or (abs(a.z1 - s.z1) < 1e-9
                                   and abs(a.z2 - s.z2) < 1e-9) for a in ext):
            return
        ext = filter_front(ext + [improve(inst, s, budget, r_rounds)])

    genes: List[Tuple[List[int], float]] = []
    pop: List[Solution] = []
    while len(pop) < pop_size and not budget.exhausted:
        perm = ids[:]
        rng.shuffle(perm)
        lam = _log_uniform(rng)
        s = make(perm, lam)
        if s is not None:
            pop.append(s)
            genes.append((perm, lam))
            offer(s)
    if not pop:
        return []

    archive = filter_front(pop)
    gen = 0
    while not budget.exhausted:
        gen += 1
        fronts = _fast_nondominated_sort(pop)
        rank = {i: r for r, f in enumerate(fronts) for i in f}
        crowd: Dict[int, float] = {}
        for f in fronts:
            crowd.update(_crowding(pop, f))

        def pick() -> int:
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if crowd.get(a, 0.0) > crowd.get(b, 0.0) else b

        children: List[Solution] = []
        child_genes: List[Tuple[List[int], float]] = []
        while len(children) < pop_size and not budget.exhausted:
            (pa, la), (pb, lb) = genes[pick()], genes[pick()]
            if rng.random() < p_cross:
                perm = _order_crossover(pa, pb, rng)
                u = rng.random()
                lam = math.exp(math.log(la) + u * (math.log(lb) - math.log(la)))
            else:
                perm, lam = list(pa), la
            if rng.random() < p_mut:
                perm = _inversion(perm, rng)
            if rng.random() < p_mut:
                lam = min(PERM_LAMBDA_MAX,
                          max(PERM_LAMBDA_MIN,
                              lam * math.exp(rng.gauss(0.0, 0.7))))
            s = make(perm, lam)
            if s is not None:
                children.append(s)
                child_genes.append((perm, lam))
                offer(s)

        merged = pop + children
        merged_genes = genes + child_genes
        fronts = _fast_nondominated_sort(merged)
        new_pop: List[Solution] = []
        new_genes: List[Tuple[List[int], float]] = []
        for f in fronts:
            if len(new_pop) + len(f) <= pop_size:
                for i in f:
                    new_pop.append(merged[i])
                    new_genes.append(merged_genes[i])
            else:
                d = _crowding(merged, f)
                for i in sorted(f, key=lambda x: -d.get(x, 0.0)):
                    if len(new_pop) >= pop_size:
                        break
                    new_pop.append(merged[i])
                    new_genes.append(merged_genes[i])
                break
        pop, genes = new_pop, new_genes
        archive = filter_front(archive + pop)

        if refresh and gen % refresh == 0 and ext and not budget.exhausted:
            state["batches"] = min(ext, key=lambda s: s.z1).batches
            re_pop, re_genes = [], []
            for (perm, lam) in genes:
                s = make(perm, lam)
                if s is None:
                    break
                re_pop.append(s)
                re_genes.append((perm, lam))
            if len(re_pop) == len(genes):
                pop, genes = re_pop, re_genes
                archive = filter_front(archive + pop)

    return filter_front(ext + archive)
