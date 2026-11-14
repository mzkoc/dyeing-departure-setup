"""CPM+ALNS+R: the ladder supplies the setup end, ALNS+R the tardiness end.

The hybrid runs, within one clock:

  1. the bound-attaining schedule B of Corollary 3: the minimal support,
     solved once by L2b-L4 on the packed batches. It is computed first and
     without interruption, and its time is deducted from the clock, so every
     run contains a Z2-optimal schedule by construction (CPM's own first rung would be lost
     if the clock ran out while CPM was still building its batches);
  2. CPM on a share of the clock (its ladder);
  3. CPM-A: every CPM point and B are adapted to their support - solve,
     redistribute the batches against that schedule's completion times, solve
     again, while Z1 improves. The layer diagnosis showed that CPM's batches,
     built for one timeline and reused for every support, are where it loses
     tardiness; adaptation recovers part of that loss at negligible cost;
  4. ALNS+R on the rest of the clock.

Designs (how much of steps 1-3 reaches step 4):
  seed    every CPM-A point enters ALNS+R's starting archive and the search
          starts from the least-tardiness one;
  anchor  only B enters the archive; the search starts from its own greedy
          schedule;
  union   no interaction: ALNS+R runs unseeded (random stream seed + 500, so
          that it does not repeat the ALNS+R benchmark's run) and the fronts
          are merged.

All time spent in steps 1-3, including B and the adaptation, is taken from
the clock: ALNS+R receives what is left.

The returned `parts` give, for the union design, ALNS+R's own front, the
CPM-A front and B, so that the contribution of the ladder can be measured
afterwards: union against (its own ALNS+R part + B), and union against (the
ALNS+R benchmark + B).

Method names:  CPM+ALNS+R               seed, share SHARE
               CPM+ALNS+R@0.1           seed, share 0.1
               CPM+ALNS+R:anchor@0.05   anchor, share 0.05
               CPM+ALNS+R:union@0.05    union, share 0.05
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from alns_r import _redistributed, alns_r
from benchmarks import Budget, Solution, filter_front
from core import evaluate_schedule
from cpm import _evaluate, cpm_report
from layer1 import build_batches
from layer234 import minimal_support, solve_support

SHARE = 0.30        # share of the clock given to CPM in "CPM+ALNS+R"
NAME = "CPM+ALNS+R"
MODES = ("seed", "anchor", "union")
UNION_SEED_OFFSET = 500
ADAPT_ROUNDS = 6


def parse_hybrid(method: str) -> Optional[Tuple[str, float]]:
    """(mode, share) encoded in a method name, or None if not the hybrid."""
    if method == NAME:
        return "seed", SHARE
    if not method.startswith(NAME):
        return None
    rest = method[len(NAME):]
    mode, share = "seed", SHARE
    if rest.startswith(":"):
        rest = rest[1:]
        mode, _, tail = rest.partition("@")
        rest = "@" + tail if tail else ""
    if rest.startswith("@"):
        share = float(rest[1:])
    elif rest:
        raise ValueError(f"cannot read hybrid name: {method}")
    if mode not in MODES:
        raise ValueError(f"unknown hybrid mode in {method}; use {MODES}")
    if not 0.0 < share < 1.0:
        raise ValueError(f"share must lie in (0, 1): {method}")
    return mode, share


def hybrid_share(method: str):
    p = parse_hybrid(method)
    return None if p is None else p[1]


def bound_point(inst) -> Tuple[Solution, dict]:
    """Corollary 3's schedule: minimal support, packed batches, full L2b-L4."""
    batches = build_batches(inst, None, variant="cpm")
    colours = sorted({b.family for b in batches})
    support = minimal_support(inst, colours)
    p = _evaluate(inst, batches, support)
    if p is None:
        raise RuntimeError(f"{inst.name}: the minimal support has no schedule")
    return Solution(p.batches, p.schedule, p.z1, p.z2), support


def adapt(inst, support, sol: Solution,
          rounds: int = ADAPT_ROUNDS) -> Solution:
    """CPM-A for one point: re-solve its support with batches redistributed
    against the schedule's own completion times, while Z1 improves. Never
    returns a worse point than it was given."""
    best = sol
    batches = list(sol.batches)
    prev = None
    for _ in range(rounds):
        sched, z1, z2 = solve_support(inst, batches, support)
        if z1 < best.z1 - 1e-9:
            best = Solution(batches, sched, z1, z2)
        if prev is not None and z1 >= prev - 1e-9:
            break
        prev = z1
        ev = evaluate_schedule(inst, batches, sched, strict=False)
        batches = _redistributed(inst, batches, ev.completion)
    return best


def cpm_alns_r(inst, seconds: float, *, seed: int = 0, share: float = SHARE,
               mode: str = "seed", alns_params: Optional[dict] = None,
               return_parts: bool = False, search_inst=None):
    """`search_inst` (E2, union mode only): the instance the ALNS+R part
    searches on, while B, CPM and CPM-A work on `inst`. The returned front
    then mixes objective values of two instances; the caller re-evaluates
    the parts (e2.py). None: both work on `inst`, as everywhere else."""
    if search_inst is not None and mode != "union":
        raise ValueError("search_inst needs the union design")
    ap = dict(alns_params or {})
    t0 = time.time()

    b_sol, b_support = bound_point(inst)
    evals = 1
    t_bound = time.time() - t0

    pts: List[Tuple[Solution, dict]] = []
    t_cpm = seconds * share - (time.time() - t0)
    t1 = time.time()
    if t_cpm > 0:
        front, diag = cpm_report(inst, variant="cpm", eval_budget=10 ** 9,
                                 time_limit=t_cpm, safety_seconds=t_cpm * 1.5)
        evals += diag["evaluations"]
        pts = [(Solution(p.batches, p.schedule, p.z1, p.z2), p.support)
               for p in front]

    t2 = time.time()
    b_ad = adapt(inst, b_support, b_sol)
    cpm_part = filter_front([adapt(inst, sup, s) for s, sup in pts] + [b_ad])
    if not any(abs(s.z2 - b_ad.z2) < 1e-9 for s in cpm_part):
        raise RuntimeError("the Z2-optimal schedule left the CPM part")

    t3 = time.time()
    left = max(seconds - (time.time() - t0), 0.01)
    b = Budget(10 ** 9, seconds=left)
    if mode == "seed":
        out = alns_r(inst, b, seed=seed, init=cpm_part, **ap)
        alns_part = []
    elif mode == "anchor":
        out = alns_r(inst, b, seed=seed, init=[b_ad], start_from_init=False,
                     **ap)
        alns_part = []
    elif mode == "union":
        alns_part = alns_r(search_inst if search_inst is not None else inst,
                           b, seed=seed + UNION_SEED_OFFSET, **ap)
        out = filter_front(alns_part + cpm_part)
    else:
        raise ValueError(mode)
    evals += b.used

    if not return_parts:
        return out, evals
    origin = []
    for s in out:
        if any(s is c for c in cpm_part):
            origin.append("cpm")
        elif any(s is a for a in alns_part):
            origin.append("alns")
        else:
            origin.append("search")
    # time of each component (logging only; nothing above depends on it)
    timing = {"bound": round(t_bound, 4), "cpm": round(t2 - t1, 4),
              "adapt": round(t3 - t2, 4),
              "alns": round(time.time() - t3, 4)}
    parts = {"alns": alns_part, "cpm": cpm_part, "bound": b_ad,
             "origin": origin, "seconds": timing}
    return out, evals, parts
