"""
Benchmark methods: ALNS and NSGA-II.

Both are wired through `core.evaluate_schedule`. Neither computes its own Z1
or Z2, so no two methods in this study can disagree about the value of the
same schedule - that was the single biggest weakness of the old notebook,
where every algorithm reported its own numbers.

Both search the UNRESTRICTED space. No block contiguity, no last-block
pinning, no fixed batch count. CPM restricts itself; its competitors do not,
which is what keeps the comparison honest.

BUDGET. Effort is counted in objective evaluations, not seconds - the same
reason CPM moved off wall-clock: identical runs disagreed depending on machine
load. One `Budget` object is handed to every method so the counts mean the
same thing.

A caveat worth stating in the paper rather than hiding: an evaluation is not
equally expensive across methods. CPM's unit covers one support, inside which
a Hungarian solve runs per colour; ALNS and NSGA-II evaluate one schedule.
Wall-clock is therefore reported alongside, and the two together give a fair
picture where either alone would mislead.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from core import Batch, Instance, Schedule, evaluate_schedule
from layer1 import group_jobs, pack_group


# --------------------------------------------------------------------------


class Budget:
    """Shared effort counter, by evaluations or by wall clock.

    Evaluations are the default because they reproduce exactly: with a time
    limit the same instance returned different fronts depending on machine
    load. But one evaluation is not equally expensive across methods - the
    proposed method's covers a whole support, with a Hungarian solve per
    family inside it, where a metaheuristic's covers one schedule - so a
    comparison at equal evaluations can be read as favouring it.

    `seconds` switches the same object to a wall-clock limit, so the equal
    time comparison runs through the identical code path. Runs under it do
    not reproduce exactly; that is the point of the check, and the results
    from it are reported separately from the main study.
    """

    __slots__ = ("max_evals", "used", "seconds", "_t0")

    def __init__(self, max_evals: int, seconds: float | None = None):
        self.max_evals = max_evals
        self.seconds = seconds
        self.used = 0
        self._t0 = time.time() if seconds else 0.0

    def spend(self) -> bool:
        if self.exhausted:
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        if self.seconds is not None:
            return time.time() - self._t0 >= self.seconds
        return self.used >= self.max_evals


class Solution:
    __slots__ = ("batches", "schedule", "z1", "z2")

    def __init__(self, batches: List[Batch], schedule: Schedule,
                 z1: float, z2: float):
        self.batches, self.schedule = batches, schedule
        self.z1, self.z2 = z1, z2

    def objectives(self) -> Tuple[float, float]:
        return (self.z1, self.z2)


def evaluate(inst: Instance, batches: List[Batch], schedule: Schedule,
             budget: Budget) -> Solution | None:
    """The only path to an objective value in this module."""
    if not budget.spend():
        return None
    ev = evaluate_schedule(inst, batches, schedule, strict=False)
    if not ev.feasible:
        return None
    return Solution(batches, schedule, ev.z1, ev.z2)


def dominates(a: Solution, b: Solution) -> bool:
    return (a.z1 <= b.z1 + 1e-9 and a.z2 <= b.z2 + 1e-9
            and (a.z1 < b.z1 - 1e-9 or a.z2 < b.z2 - 1e-9))


def filter_front(sols: Sequence[Solution]) -> List[Solution]:
    out: List[Solution] = []
    for s in sols:
        if any(dominates(t, s) for t in sols if t is not s):
            continue
        if any(abs(t.z1 - s.z1) < 1e-9 and abs(t.z2 - s.z2) < 1e-9
               for t in out):
            continue
        out.append(s)
    return sorted(out, key=lambda s: (s.z2, s.z1))


# --------------------------------------------------------------------------
# Shared construction
# --------------------------------------------------------------------------


def initial_batches(inst: Instance, rng: random.Random,
                    shuffle: bool = False) -> List[Batch]:
    batches: List[Batch] = []
    bid = 0
    for (fam, cust), jobs in sorted(group_jobs(inst).items()):
        js = list(jobs)
        if shuffle:
            rng.shuffle(js)
            bins: List[List] = []
            loads: List[float] = []
            for j in js:
                placed = False
                for i, load in enumerate(loads):
                    if load + j.size <= inst.capacity + 1e-9:
                        bins[i].append(j)
                        loads[i] += j.size
                        placed = True
                        break
                if not placed:
                    bins.append([j])
                    loads.append(j.size)
        else:
            bins = pack_group(js, inst.capacity)
        for run in bins:
            batches.append(Batch(id=bid, jobs=tuple(sorted(j.id for j in run)),
                                 family=fam, customer=cust))
            bid += 1
    return batches


def greedy_schedule(inst: Instance, batches: Sequence[Batch],
                    key) -> Schedule:
    """Place batches on the least-loaded machine in `key` order."""
    sched: Schedule = {m: [] for m in range(inst.machines)}
    load = [0.0] * inst.machines
    last = [None] * inst.machines
    for b in sorted(batches, key=key):
        best, best_cost = 0, float("inf")
        for m in range(inst.machines):
            setup = 0.0 if last[m] in (None, b.family) else inst.sigma[last[m]]
            cost = load[m] + setup
            if cost < best_cost:
                best, best_cost = m, cost
        sched[best].append(b.id)
        load[best] = best_cost + inst.proc[b.family]
        last[best] = b.family
    return sched


def edd_key(inst: Instance):
    return lambda b: (min(inst.job(j).due for j in b.jobs), b.family, b.id)


# --------------------------------------------------------------------------
# ALNS
# --------------------------------------------------------------------------

DESTROY = ("random", "worst", "colour", "machine")
REPAIR = ("greedy", "edd", "least_load")


def _remove(inst: Instance, sched: Schedule, batches: Sequence[Batch],
            how: str, k: int, rng: random.Random) -> Tuple[Schedule, List[int]]:
    by_id = {b.id: b for b in batches}
    placed = [(m, bid) for m, seq in sched.items() for bid in seq]
    if not placed:
        return sched, []

    if how == "random":
        chosen = rng.sample(placed, min(k, len(placed)))
    elif how == "worst":
        ev = evaluate_schedule(inst, batches, sched, strict=False)
        scored = sorted(
            placed,
            key=lambda t: -sum(max(0.0, ev.completion.get(t[1], 0.0)
                                   - inst.job(j).due)
                               for j in by_id[t[1]].jobs))
        chosen = scored[:k]
    elif how == "colour":
        fam = rng.choice(sorted(inst.proc))
        chosen = [t for t in placed if by_id[t[1]].family == fam][:k]
    else:                                             # machine
        m = rng.randrange(inst.machines)
        chosen = [t for t in placed if t[0] == m][:k]

    removed = [bid for _, bid in chosen]
    new: Schedule = {m: [b for b in seq if b not in removed]
                     for m, seq in sched.items()}
    return new, removed


def _insert(inst: Instance, sched: Schedule, batches: Sequence[Batch],
            removed: Sequence[int], how: str,
            rng: random.Random) -> Schedule:
    by_id = {b.id: b for b in batches}
    order = list(removed)
    if how == "edd":
        order.sort(key=lambda bid: min(inst.job(j).due
                                       for j in by_id[bid].jobs))
    elif how == "greedy":
        rng.shuffle(order)

    # "greedy" is objective-aware: it inserts where the machine's tardiness
    # plus a randomly weighted setup term grows least, so the repair itself
    # can move along the front. "edd" keeps the makespan-type insertion and
    # "least_load" appends to the lightest machine, which gives the operator
    # pool three distinct behaviours for the adaptive weights to choose among.
    a = rng.random()
    dues = {x: [inst.job(j).due for j in by_id[x].jobs] for x in by_id}
    for bid in order:
        b = by_id[bid]
        best = (0, 0, float("inf"))
        for m, seq in sched.items():
            if how == "least_load":
                score = sum(inst.proc[by_id[x].family] for x in seq)
                if score < best[2]:
                    best = (m, len(seq), score)
                continue
            for pos in range(len(seq) + 1):
                trial = list(seq)
                trial.insert(pos, bid)
                prev = None
                t = 0.0
                tard = 0.0
                setup = 0.0
                for x in trial:
                    fx = by_id[x].family
                    if prev is not None and prev != fx:
                        t += inst.sigma[prev]
                        setup += inst.sigma[prev]
                    t += inst.proc[fx]
                    if how == "greedy":
                        tard += sum(max(0.0, t - d) for d in dues[x])
                    prev = fx
                if how == "greedy":
                    cost = (1.0 - a) * tard + a * setup * 100.0
                else:
                    cost = t
                if cost < best[2]:
                    best = (m, pos, cost)
        m, pos, _ = best
        sched[m].insert(pos, bid)
    return sched


def alns(inst: Instance, budget: Budget, *, seed: int = 0,
         segment: int = 25, decay: float = 0.8) -> List[Solution]:
    """Adaptive large neighbourhood search with a Pareto archive.

    Operator weights are adapted the usual way (Ropke & Pisinger). Acceptance
    is by dominance against the archive, with a random scalarising weight to
    steer each segment toward a different part of the front.
    """
    rng = random.Random(seed)
    batches = initial_batches(inst, rng)
    sched = greedy_schedule(inst, batches, edd_key(inst))
    cur = evaluate(inst, batches, sched, budget)
    if cur is None:
        return []
    archive: List[Solution] = [cur]

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
        k = max(1, int(n_placed * rng.uniform(0.1, 0.35)))

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
            archive = filter_front(archive + [cand])

        # random scalarisation keeps the walk from collapsing into one corner
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


# --------------------------------------------------------------------------
# NSGA-II
# --------------------------------------------------------------------------


def _fast_nondominated_sort(pop: Sequence[Solution]) -> List[List[int]]:
    S, n_dom = [[] for _ in pop], [0] * len(pop)
    fronts: List[List[int]] = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(pop: Sequence[Solution], front: Sequence[int]) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    for obj in (lambda s: s.z1, lambda s: s.z2):
        order = sorted(front, key=lambda i: obj(pop[i]))
        dist[order[0]] = dist[order[-1]] = float("inf")
        lo, hi = obj(pop[order[0]]), obj(pop[order[-1]])
        if hi - lo < 1e-12:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += (obj(pop[order[k + 1]])
                               - obj(pop[order[k - 1]])) / (hi - lo)
    return dist


PERM_LAMBDA_MIN = 1e-2     # weakest setup aversion (practically pure EC)
PERM_LAMBDA_MAX = 200.0    # strong enough to keep a colour on its machine
                           # against a load difference of 100+ hours: an
                           # earlier cap of 6 could not, so the decoder could
                           # not express a zero-setup schedule even where one
                           # trivially exists (|F| <= m)


def _log_uniform(rng: random.Random) -> float:
    """Setup-aversion gene, log-uniform over [LAMBDA_MIN, LAMBDA_MAX]."""
    lo, hi = math.log(PERM_LAMBDA_MIN), math.log(PERM_LAMBDA_MAX)
    return math.exp(rng.uniform(lo, hi))


def _greedy_decode(inst: Instance, batches: Sequence[Batch],
                   perm: Sequence[int], lam: float) -> Schedule:
    """Place batches in `perm` order, each on the machine that finishes it
    earliest under a setup-aversion weight.

    Two things here were wrong in the first version of this benchmark and are
    worth recording, because they made NSGA-II look far weaker than it is.

    1. Machine choice used to be read straight off a random key
       (`m = int(key * machines)`), which is blind to both setup and load: a
       batch could land on a machine carrying a different colour and already
       full. ALNS in this same module has always chosen machines with setups
       in view, and it beat NSGA-II at every scale - that gap was the decoder,
       not the algorithm. The greedy rule below is the standard permutation
       decoder for parallel machines.

    2. A pure earliest-completion rule is a Z1 rule. With nothing to trade
       against it, the search could not reach the low-Z2 region at all, so
       the comparison on the second objective was decided before it began.
       `lam` is an evolved gene: at 0 the rule is pure earliest completion,
       and as it grows the decoder increasingly refuses to break a colour
       run. The population therefore spreads along the Z2 axis by itself.
    """
    sched: Schedule = {m: [] for m in range(inst.machines)}
    time = [0.0] * inst.machines
    last: List[str | None] = [None] * inst.machines
    by_id = {b.id: b for b in batches}

    for idx in perm:
        b = by_id[idx]
        best, best_score = 0, float("inf")
        for m in range(inst.machines):
            setup = 0.0 if last[m] in (None, b.family) else inst.sigma[last[m]]
            end = time[m] + setup + inst.proc[b.family]
            score = end + lam * setup
            if score < best_score:
                best, best_score = m, score
        setup = (0.0 if last[best] in (None, b.family)
                 else inst.sigma[last[best]])
        sched[best].append(b.id)
        time[best] += setup + inst.proc[b.family]
        last[best] = b.family
    return sched


def _order_crossover(a: Sequence[int], b: Sequence[int],
                     rng: random.Random) -> List[int]:
    """Order crossover (OX): keeps a slice of `a`, fills the rest in `b`'s
    order. The standard permutation operator - a one-point cut on a raw key
    vector, which is what this used before, mixes unrelated decision blocks
    and destroys the ordering the genome encodes."""
    n = len(a)
    if n < 3:
        return list(a)
    i, j = sorted(rng.sample(range(n), 2))
    child: List[int | None] = [None] * n
    child[i:j + 1] = list(a[i:j + 1])
    taken = set(a[i:j + 1])
    fill = (x for x in b if x not in taken)
    for k in range(n):
        if child[k] is None:
            child[k] = next(fill)
    return child                                           # type: ignore


def _inversion(perm: List[int], rng: random.Random) -> List[int]:
    """Inversion mutation: reverse a segment. Standard for permutations."""
    n = len(perm)
    if n < 2:
        return perm
    i, j = sorted(rng.sample(range(n), 2))
    return perm[:i] + perm[i:j + 1][::-1] + perm[j + 1:]


def nsga2(inst: Instance, budget: Budget, *, seed: int = 0,
          pop_size: int = 60, p_cross: float = 0.9,
          p_mut: float = 0.2) -> List[Solution]:
    """Permutation NSGA-II with a setup-aware greedy decoder.

    Genome: a permutation of the batches plus one real gene `lam` steering
    the decoder's setup aversion. Operators: order crossover and inversion.
    Selection: fast non-dominated sorting with crowding distance and binary
    tournament - textbook NSGA-II.

    Batching is the same FFD/BFD packing CPM starts from, before L1's
    redistribution. Fixing it matches how decomposition benchmarks are
    normally run and, if anything, hands NSGA-II a good batching for free.
    """
    rng = random.Random(seed)
    batches = initial_batches(inst, rng)
    ids = [b.id for b in batches]

    def make(perm: List[int], lam: float) -> Solution | None:
        return evaluate(inst, batches,
                        _greedy_decode(inst, batches, perm, lam), budget)

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
    if not pop:
        return []

    archive = filter_front(pop)
    while not budget.exhausted:
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
                # blend on the log scale, where the gene is sampled
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

    return filter_front(archive)


# --------------------------------------------------------------------------
# LPT + EDD dispatching baseline
# --------------------------------------------------------------------------


def lpt_edd(inst: Instance, budget: Budget) -> List[Solution]:
    """Dispatching baseline: pack, then order by EDD, then by colour-then-EDD.

    Two rules rather than one, so the baseline has at least a chance of
    showing a trade-off instead of a single point.
    """
    rng = random.Random(0)
    batches = initial_batches(inst, rng)
    out: List[Solution] = []
    for key in (edd_key(inst),
                lambda b: (b.family, min(inst.job(j).due for j in b.jobs)),
                lambda b: (-inst.proc[b.family],
                           min(inst.job(j).due for j in b.jobs))):
        s = evaluate(inst, batches, greedy_schedule(inst, batches, key), budget)
        if s is not None:
            out.append(s)
    return filter_front(out)
