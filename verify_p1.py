"""
Brute-force verification of P1 (the Z2 closed form and its lower bound).

The proof says: under contiguity, Z2(m) = sum(sigma_f over F_m) - max(sigma_f),
and globally min Z2 = sum of the e = |F| - m smallest sigma values (0 when
e <= 0). This script enumerates EVERY assignment of batches to machines and
EVERY sequence on each machine - contiguous or not - and checks that the
observed minimum matches the closed form.

This is the code-side counterpart of the proof. If it fails, either the proof
or the evaluator is wrong, and we want to know that before building CPM on it.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, Iterator, List

from core import (
    Batch,
    Instance,
    Job,
    Schedule,
    evaluate_schedule,
    z2_lower_bound,
    z2_machine_closed_form,
)


# --------------------------------------------------------------------------
# Exhaustive enumeration of schedules
# --------------------------------------------------------------------------


def all_schedules(batch_ids: List[int], m: int) -> Iterator[Schedule]:
    """Every way to place the batches into m ordered sequences.

    Built by inserting one batch at a time into any machine at any position,
    which enumerates each ordered arrangement exactly once.
    """
    def rec(i: int, current: List[List[int]]) -> Iterator[Schedule]:
        if i == len(batch_ids):
            yield {mi: list(seq) for mi, seq in enumerate(current)}
            return
        bid = batch_ids[i]
        for mi in range(m):
            for pos in range(len(current[mi]) + 1):
                current[mi].insert(pos, bid)
                yield from rec(i + 1, current)
                current[mi].pop(pos)

    yield from rec(0, [[] for _ in range(m)])


def count_schedules(n_batches: int, m: int) -> int:
    total = 1
    for i in range(n_batches):
        total *= m + i
    return total


# --------------------------------------------------------------------------
# Random small instances
# --------------------------------------------------------------------------


def random_small_instance(rng: random.Random, n_f: int, m: int,
                          n_batches: int) -> tuple[Instance, List[Batch]]:
    families = [f"F{i}" for i in range(n_f)]
    proc = {f: rng.choice([2.0, 3.0, 4.0, 5.0, 6.0]) for f in families}
    # sigma is randomised here ON PURPOSE. The production instances take it
    # from a dyehouse matrix, but P1 must hold for ANY from-colour-only setup
    # function, not for one particular table - so the verification should not
    # be able to pass by accident on the numbers we happen to ship.
    sigma = {f: round(proc[f] * rng.choice([0.1, 0.15, 0.2, 0.25]), 3)
             for f in families}

    # Every colour must appear at least once, so the |F| > m regime is real.
    colours = families + [rng.choice(families)
                          for _ in range(n_batches - n_f)]
    rng.shuffle(colours)

    jobs: List[Job] = []
    batches: List[Batch] = []
    jid = 0
    for bid, f in enumerate(colours):
        cust = f"C{rng.randint(0, 1)}"
        k = rng.randint(1, 2)
        ids = []
        for _ in range(k):
            jobs.append(Job(id=jid, size=rng.choice([80.0, 120.0, 160.0]),
                            family=f, customer=cust,
                            due=float(rng.randint(5, 30))))
            ids.append(jid)
            jid += 1
        batches.append(Batch(id=bid, jobs=tuple(ids), family=f, customer=cust))

    inst = Instance(jobs=jobs, machines=m, capacity=400.0,
                    proc=proc, sigma=sigma,
                    name=f"verify_F{n_f}_m{m}_B{n_batches}")
    return inst, batches


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_instance(inst: Instance, batches: List[Batch],
                   verbose: bool = False) -> dict:
    bids = [b.id for b in batches]
    fam = {b.id: b.family for b in batches}

    best_z2 = float("inf")
    best_sched: Schedule | None = None
    contiguous_min = float("inf")
    closed_form_mismatch = 0
    checked = 0

    for sched in all_schedules(bids, inst.machines):
        ev = evaluate_schedule(inst, batches, sched, strict=False)
        checked += 1
        if ev.z2 < best_z2 - 1e-9:
            best_z2 = ev.z2
            best_sched = {k: list(v) for k, v in sched.items()}

        # Is this schedule contiguous? If so, P1's per-machine formula must
        # reproduce its Z2 exactly.
        if is_contiguous(sched, fam):
            predicted = sum(
                z2_machine_closed_form(inst, [fam[b] for b in seq])
                for seq in sched.values() if seq
            )
            # The formula assumes the largest-sigma colour is the last block;
            # a contiguous schedule need not do that, so the formula is a
            # lower bound for this particular arrangement.
            if predicted > ev.z2 + 1e-9:
                closed_form_mismatch += 1
            contiguous_min = min(contiguous_min, ev.z2)

    lb = z2_lower_bound(inst)
    return {
        "name": inst.name,
        "families": len(inst.proc),
        "machines": inst.machines,
        "batches": len(batches),
        "excess": inst.excess,
        "schedules_checked": checked,
        "brute_force_min_z2": round(best_z2, 6),
        "contiguous_min_z2": round(contiguous_min, 6),
        "closed_form_lb": round(lb, 6),
        "lb_matches": abs(best_z2 - lb) < 1e-6,
        "contiguity_optimal": abs(contiguous_min - best_z2) < 1e-6,
        "per_machine_formula_violations": closed_form_mismatch,
        "best_schedule": best_sched,
    }


def is_contiguous(sched: Schedule, fam: Dict[int, str]) -> bool:
    """True if every colour forms at most one block on every machine."""
    for seq in sched.values():
        seen = set()
        prev = None
        for bid in seq:
            f = fam[bid]
            if f != prev:
                if f in seen:
                    return False
                seen.add(f)
                prev = f
    return True


# --------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(20260923)

    # (|F|, m, |B|) triples covering all three Z2 regimes:
    #   e < 0  Z2 trivially zero
    #   e = 0  boundary
    #   e > 0  setup binding
    configs = [
        (2, 4, 4),   # e = -2
        (3, 5, 5),   # e = -2
        (3, 3, 5),   # e =  0
        (4, 4, 5),   # e =  0
        (4, 2, 5),   # e = +2
        (5, 2, 6),   # e = +3
        (5, 3, 6),   # e = +2
        (4, 1, 5),   # e = +3, single machine
        (6, 2, 6),   # e = +4
    ]

    print(f"{'instance':>22} {'e':>3} {'#sched':>9} {'BF min':>8} "
          f"{'LB':>8} {'LB=BF':>6} {'contig':>7} {'viol':>5}")
    print("-" * 78)

    all_ok = True
    for n_f, m, n_b in configs:
        if count_schedules(n_b, m) > 400_000:
            print(f"  skipping F{n_f}/m{m}/B{n_b}: too many schedules")
            continue
        inst, batches = random_small_instance(rng, n_f, m, n_b)
        r = check_instance(inst, batches)
        ok = r["lb_matches"] and r["contiguity_optimal"] \
            and r["per_machine_formula_violations"] == 0
        all_ok &= ok
        print(f"{r['name']:>22} {r['excess']:>3} {r['schedules_checked']:>9} "
              f"{r['brute_force_min_z2']:>8.3f} {r['closed_form_lb']:>8.3f} "
              f"{str(r['lb_matches']):>6} {str(r['contiguity_optimal']):>7} "
              f"{r['per_machine_formula_violations']:>5}")

    print("-" * 78)
    print("P1 VERIFIED" if all_ok else "P1 FAILED - investigate before proceeding")


if __name__ == "__main__":
    main()
