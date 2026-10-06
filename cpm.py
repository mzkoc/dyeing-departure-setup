"""
CPM driver: the outer loop and the Pareto front.

Two nested ideas.

Outer loop (L1 feedback). L1 cannot see completion times on its first pass, so
it packs and stops. From the second pass it redistributes jobs between a
group's batches against the times the previous pass produced. Three or four
passes reach a fixed point.

Pareto front. Z2 is a function of the colour->machine support alone (P1), and
it moves in discrete steps: putting a colour on one more machine costs exactly
+sigma_f (Corollary 4). So instead of an epsilon grid we run a Pareto local
search over supports, keeping an archive of non-dominated (Z1, Z2) pairs. Each
archive member is a real schedule, and the Z2 axis is exact rather than
sampled - no payoff table, no grid resolution, no fallback constants.

The front this produces is an APPROXIMATION of the front of the restricted
space (block contiguity, last block pinned to argmax sigma): the sweep walks
one path up the ladder of split budgets and does not enumerate every support.
CP-SAT searches a larger space, which is how the restriction's price gets
measured.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from core import Batch, Evaluation, Instance, Schedule, evaluate_schedule
from layer1 import build_batches, observed_completions
from layer234 import (
    Assigner,
    Support,
    minimal_support,
    solve_support,
    z2_of_support,
)

OUTER_PASSES = 4
FIRST_PASS_SHARE = 0.7  # share of the budget given to the first (full) sweep
STALL_LEVELS = 5       # levels of no Z1 gain before the sweep stops
EVAL_BUDGET = 10000    # support evaluations per instance (the real limit)
SAFETY_SECONDS = 600.0  # wall-clock net, not the control


class Budget:
    """Deterministic evaluation budget.

    The search used to be cut off by wall-clock, which made results depend on
    machine load: the same instance at the same 30 s limit returned Z1 = 105.0
    run alone and Z1 = 136.0 inside a batch of six. A paper cannot report
    numbers that move with the load on the box, so the control is now a count
    of support evaluations. The clock survives only as a safety net, and when
    it fires the run is flagged rather than silently truncated.
    """

    __slots__ = ("max_evals", "used", "deadline", "timed_out")

    def __init__(self, max_evals: int, time_limit: float | None = None):
        self.max_evals = max_evals
        self.used = 0
        self.deadline = ((time.time() + time_limit)
                         if time_limit is not None else None)
        self.timed_out = False

    def spend(self) -> bool:
        if self.used >= self.max_evals:
            return False
        if self.deadline is not None and time.time() > self.deadline:
            self.timed_out = True
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_evals or self.timed_out
PARETO_PASSES = 40


# --------------------------------------------------------------------------


class Point:
    __slots__ = ("z1", "z2", "support", "schedule", "batches")

    def __init__(self, z1: float, z2: float, support: Support,
                 schedule: Schedule, batches: List[Batch]):
        self.z1, self.z2 = z1, z2
        self.support, self.schedule, self.batches = support, schedule, batches

    def key(self) -> Tuple[float, float]:
        return (self.z1, self.z2)

    def __repr__(self) -> str:
        return f"Point(Z1={self.z1:.1f}, Z2={self.z2:.2f})"


def dominates(a: Point, b: Point) -> bool:
    return (a.z1 <= b.z1 + 1e-9 and a.z2 <= b.z2 + 1e-9
            and (a.z1 < b.z1 - 1e-9 or a.z2 < b.z2 - 1e-9))


def filter_front(points: Sequence[Point]) -> List[Point]:
    out: List[Point] = []
    for p in points:
        if any(dominates(q, p) for q in points if q is not p):
            continue
        if any(abs(q.z1 - p.z1) < 1e-9 and abs(q.z2 - p.z2) < 1e-9
               for q in out):
            continue
        out.append(p)
    return sorted(out, key=lambda p: (p.z2, p.z1))


# --------------------------------------------------------------------------


def _n_batches(batches: Sequence[Batch]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for b in batches:
        out[b.family] += 1
    return dict(out)


def _evaluate(inst: Instance, batches: List[Batch],
              support: Support,
              assigner: Assigner | None = None,
              *, quick: bool = False,
              budget: "Budget | None" = None) -> Point | None:
    """Evaluate a support. `quick` trims the counts and block-order search.

    The sweep looks at hundreds of supports per level and keeps one, so paying
    full price for every candidate wastes the budget on options that get
    discarded. Candidates are screened cheaply; whatever the level settles on
    is then polished at full strength.
    """
    if budget is not None and not budget.spend():
        return None
    kw = dict(counts_passes=2, order_rounds=1) if quick else {}
    try:
        sched, z1, z2 = solve_support(inst, batches, support,
                                      assigner=assigner, **kw)
    except ValueError:
        return None
    return Point(z1, z2, support, sched, batches)


def _budget(support: Support) -> int:
    """Split budget: how many machine slots beyond one per colour."""
    return sum(len(ms) for ms in support.values()) - len(support)


def _budget_neighbours(inst: Instance, support: Support,
                       nb: Dict[str, int]) -> List[Support]:
    """Neighbours that keep the split budget, so Z2 stays in its level band.

    Two moves: RELOCATE a colour to a different machine, and TRANSFER one
    split from one colour to another. Both preserve sum(|M_f|), which is what
    pins the level.
    """
    out: List[Support] = []
    colours = list(support)

    for f in colours:
        ms = support[f]
        for old in ms:
            for new in range(inst.machines):
                if new in ms:
                    continue
                cand = dict(support)
                cand[f] = tuple(sorted([x for x in ms if x != old] + [new]))
                out.append(cand)

    for f in colours:
        if len(support[f]) <= 1:
            continue
        for g in colours:
            if g == f or len(support[g]) >= min(inst.machines, nb.get(g, 1)):
                continue
            for drop in support[f]:
                for add in range(inst.machines):
                    if add in support[g]:
                        continue
                    cand = dict(support)
                    cand[f] = tuple(x for x in support[f] if x != drop)
                    cand[g] = tuple(sorted(support[g] + (add,)))
                    out.append(cand)
    return out


def _cheapest_split(inst: Instance, support: Support,
                    nb: Dict[str, int]) -> Support | None:
    """Step up one level: add one machine slot at the smallest Z2 cost.

    Corollary 4 says a split costs exactly +sigma_f, so the cheapest colour to
    split is the one with the smallest sigma that still has room - but the
    machine it lands on matters too, because it may become that machine's last
    block. The closed form settles it without scheduling anything.
    """
    best, best_z2 = None, float("inf")
    for f, ms in support.items():
        if len(ms) >= min(inst.machines, nb.get(f, 1)):
            continue
        for m in range(inst.machines):
            if m in ms:
                continue
            cand = dict(support)
            cand[f] = tuple(sorted(ms + (m,)))
            z2 = z2_of_support(inst, cand)
            if z2 < best_z2 - 1e-12:
                best, best_z2 = cand, z2
    return best


def solve_once(inst: Instance, batches: List[Batch], *,
               max_passes: int = PARETO_PASSES,
               budget: "Budget | None" = None) -> List[Point]:
    """Level-based sweep over Z2, for a fixed batch set.

    Z2 is a closed-form function of the support (P1) and moves in discrete
    steps of +sigma_f, so the front's second axis is known before any
    scheduling happens. The sweep walks it explicitly: at each split budget it
    minimises Z1 with the budget held fixed, then takes the cheapest step up.

    This replaces an earlier generic Pareto local search, which had to
    discover the Z2 axis by exploration and left large gaps in it - one front
    covered only {0.00, 0.20} of the reachable levels, another jumped straight
    from 0.00 to 4.05. Sweeping the levels we can already compute removes the
    guesswork, and it is what makes an epsilon grid unnecessary.
    """
    if budget is None:
        budget = Budget(EVAL_BUDGET, SAFETY_SECONDS)
    nb = _n_batches(batches)
    assigner = Assigner(inst, batches)

    # Only colours that actually produced batches. A colour can be present in
    # the palette yet receive no jobs at all - in one xlarge instance the
    # generator gave "Light" zero jobs - and seeding the support from the
    # palette then asks the layers to schedule a colour with no batches.
    # solve_support raised, the seed came back None, and the instance was
    # silently dropped from the results: two of 90 xlarge rows were missing
    # for exactly this reason.
    active = sorted({b.family for b in batches})
    support = minimal_support(inst, active)
    points: List[Point] = []
    stalled = 0
    best_z1 = float("inf")

    max_levels = 1 + sum(min(inst.machines, nb.get(f, 1)) - 1
                         for f in active)
    max_levels = min(max_levels, max_passes)

    for _ in range(max_levels):
        if budget.exhausted:
            break

        # minimise Z1 with the level held fixed
        current = _evaluate(inst, batches, support, assigner, quick=True,
                            budget=budget)
        if current is None:
            break
        # Keep the level's entry point. The local search below may accept a
        # relocation that improves Z1 at a HIGHER Z2 within the same split
        # budget; without this line the level-0 support, which attains the
        # bound of Corollary 3, could be discarded and the returned front would
        # miss the provably optimal setup time it started from.
        points.append(current)
        improved = True
        while improved and not budget.exhausted:
            improved = False
            for cand in _budget_neighbours(inst, current.support, nb):
                pt = _evaluate(inst, batches, cand, assigner, quick=True,
                               budget=budget)
                if pt is None:
                    if budget.exhausted:
                        break
                    continue
                if pt.z1 < current.z1 - 1e-9:
                    current = pt
                    improved = True
                    break
        polished = _evaluate(inst, batches, current.support, assigner,
                             budget=budget)
        if polished is not None and polished.z1 <= current.z1 + 1e-9:
            current = polished
        points.append(current)

        if current.z1 < best_z1 - 1e-9:
            best_z1 = current.z1
            stalled = 0
        else:
            stalled += 1
            if stalled >= STALL_LEVELS:
                break                     # more Z2 is buying no more Z1

        nxt = _cheapest_split(inst, current.support, nb)
        if nxt is None:
            break
        support = nxt

    return filter_front(points)


def cpm(inst: Instance, *, variant: str = "cpm",
        outer_passes: int = OUTER_PASSES,
        eval_budget: int = EVAL_BUDGET,
        safety_seconds: float = SAFETY_SECONDS) -> List[Point]:
    """Full CPM: the L1 feedback loop wrapped around the level sweep.

    The budget is a count of support evaluations, split evenly across the
    outer passes so the sweep cannot spend everything in the first one and
    starve the feedback loop - that loop is worth about 10% of Z1 on its own.
    Unspent evaluations roll forward.

    Returns the front; `cpm_report` gives the same thing with diagnostics.
    """
    front, _ = cpm_report(inst, variant=variant, outer_passes=outer_passes,
                          eval_budget=eval_budget,
                          safety_seconds=safety_seconds)
    return front


def _child_budget(parent: Budget, evals: int) -> Budget:
    """A sub-budget that shares the parent's deadline."""
    left = None
    if parent.deadline is not None:
        left = max(parent.deadline - time.time(), 0.0)
    return Budget(max(1, min(evals, parent.max_evals - parent.used)), left)


def refine_front(inst: Instance, batches: List[Batch],
                 supports: Sequence[Support], budget: Budget) -> List[Point]:
    """Re-solve known supports for a new batch set (outer passes >= 2).

    The first pass walks the ladder of split budgets. Later passes only change
    the batches (L1 redistribution), and a batch change leaves Z2 untouched
    (Corollary 5), so there is no reason to walk the ladder again from the
    bottom: each support already on the front is re-evaluated with the new
    batches and improved by the same level-preserving local search. Earlier
    versions restarted the sweep in every pass with a quarter of the budget,
    which on many-machine instances left each sweep too short to leave the
    bottom of the ladder.
    """
    assigner = Assigner(inst, batches)
    nb = _n_batches(batches)
    pts: List[Point] = []
    todo = list(supports)
    for i, sup in enumerate(todo):
        if budget.exhausted:
            break
        sub = _child_budget(budget,
                            (budget.max_evals - budget.used) // (len(todo) - i))
        cur = _evaluate(inst, batches, sup, assigner, quick=True, budget=sub)
        if cur is not None:
            pts.append(cur)
            improved = True
            while improved and not sub.exhausted:
                improved = False
                for cand in _budget_neighbours(inst, cur.support, nb):
                    pt = _evaluate(inst, batches, cand, assigner, quick=True,
                                   budget=sub)
                    if pt is None:
                        if sub.exhausted:
                            break
                        continue
                    if pt.z1 < cur.z1 - 1e-9:
                        cur, improved = pt, True
                        break
            pol = _evaluate(inst, batches, cur.support, assigner, budget=sub)
            if pol is not None and pol.z1 <= cur.z1 + 1e-9:
                cur = pol
            pts.append(cur)
        budget.used += sub.used
        budget.timed_out = budget.timed_out or sub.timed_out
    return filter_front(pts)


def cpm_report(inst: Instance, *, variant: str = "cpm",
               outer_passes: int = OUTER_PASSES,
               eval_budget: int = EVAL_BUDGET,
               safety_seconds: float = SAFETY_SECONDS,
               time_limit: float | None = None,
               first_share: float = FIRST_PASS_SHARE
               ) -> Tuple[List[Point], Dict[str, object]]:
    """CPM plus a diagnostics dict.

    Budget. The first pass receives `first_share` of the evaluation budget and
    walks the whole ladder (solve_once). Each later pass re-solves the supports
    of the current front for the redistributed batches (refine_front) and
    shares what is left evenly, unspent evaluations rolling forward; any part
    of a later pass that refinement leaves unspent goes to a fresh sweep.

    `time_limit` turns the same split into a wall-clock split, for the equal
    wall-clock check; `safety_seconds` is only a net and a run that hits it is
    flagged.

    The ablation ("blind") runs the IDENTICAL pass structure and budget; its
    batches simply never change between passes. One variable.
    """
    start = time.time()
    est = None
    best: List[Point] = []
    used = 0
    passes_run = 0
    timed_out = False
    total_time = time_limit if time_limit is not None else safety_seconds

    for p in range(outer_passes):
        batches = build_batches(inst, est, variant=variant)
        remaining = eval_budget - used
        elapsed = time.time() - start
        t_left = total_time - elapsed
        if remaining <= 0 or t_left <= 0:
            break
        if p == 0:
            frac = first_share if outer_passes > 1 else 1.0
        else:
            frac = 1.0 / (outer_passes - p)
        evals = max(1, int(remaining * frac))
        secs = t_left * frac if time_limit is not None else t_left
        budget = Budget(min(evals, remaining), secs)

        if p == 0:
            front = solve_once(inst, batches, budget=budget)
        else:
            # Re-solve what is already known first, then spend whatever the
            # pass has left on a fresh sweep with the new batches: the ladder
            # path taken by the cheapest-split rule can differ once the
            # batches change.
            front = refine_front(inst, batches,
                                 [q.support for q in best], budget)
            if not budget.exhausted:
                front = filter_front(front + solve_once(inst, batches,
                                                        budget=budget))
        used += budget.used
        if time_limit is None:
            timed_out = timed_out or budget.timed_out
        passes_run += 1
        if not front:
            break
        best = filter_front(best + front)

        anchor = min(front, key=lambda q: q.z1)
        ev = evaluate_schedule(inst, anchor.batches, anchor.schedule,
                               strict=False)
        new_est = observed_completions(inst, anchor.batches, ev.completion)
        if new_est == est:
            break                           # fixed point reached
        est = new_est

    return best, {
        "evaluations": used,
        "outer_passes": passes_run,
        "timed_out": timed_out,
        "seconds": round(time.time() - start, 3),
    }


def verify_front(inst: Instance, front: Sequence[Point]) -> List[str]:
    """Re-check every point through the evaluator. Nothing is trusted."""
    problems: List[str] = []
    for p in front:
        ev: Evaluation = evaluate_schedule(inst, p.batches, p.schedule,
                                           strict=False)
        if not ev.feasible:
            problems.append(f"{p}: infeasible - {ev.violations[:2]}")
        if abs(ev.z1 - p.z1) > 1e-6:
            problems.append(f"{p}: Z1 mismatch, evaluator says {ev.z1}")
        if abs(ev.z2 - p.z2) > 1e-6:
            problems.append(f"{p}: Z2 mismatch, evaluator says {ev.z2}")
    return problems
