"""The method of the companion study [7] on the instances of this paper.

[7] solves the same batching problem under an asymmetric, pair-dependent
setup matrix: Stage 1 forms batches (FFD, then simulated annealing), Stage 2
sequences them by a tabu search (EDDTS) coupled to an epsilon-constraint
bound, and the front is sampled on the AUGMECON2 grid (two payoff runs and
GRID_POINTS - 2 grid points). Its code is used unchanged (comp7/, from the
archive of [7], version 1.3.0; see comp7/SOURCE.md), with the correction
switches and tuned parameters of the revised [7] (`rerun.load_params`).

Only the input and the output are translated.

  in    jobs: id, size, colour = family, customer, due date, p_f of the family;
        m vessels of capacity Q; the setup matrix of [7] filled with
        sigma(f, g) = sigma_f for f != g and 0 for f = g, i.e. assumption A3.
  out   every schedule of the returned front is rebuilt as batches and
        machine sequences of this repository and evaluated by
        core.evaluate_schedule; a point whose feasibility or objective values
        disagree with what [7]'s code reports is dropped and counted.

Effort. [7] gives each of its GRID_POINTS subproblems the same wall-clock
budget (HEURISTIC_TIME_BUDGET) and restarts the tabu search until it is
spent; Stage 1 runs once, untimed. Here the clock C of the comparison is
split the same way: once Stage 1 has finished, the per-subproblem budget is
set to (C - time used so far) / GRID_POINTS. The wrapper that does this
records the time and changes nothing else. GRID_POINTS = 10 as in the
campaign of [7].

Variants: "FFD+SA-EDDTS" is the method of [7]; "FFD-EDDTS" omits the
annealing step (the ablation of [7], Section 6.5, found the annealing step
does not help on its instances). "FFD+SA-EDDTS+R" gives [7] the step R that
ALNS+R, NSGA-II+R and CPM+R have: [7] runs on 85% of the clock and R is
applied to every point of its front within the rest, exactly as in cpm_r.py.
R keeps each point's timeline, so its setup value is unchanged; [7]'s front
before R is logged with the run.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import sys
import time

import numpy as np

from benchmarks import Solution, filter_front
from core import Batch, Instance, evaluate_schedule, setup_time

HERE = os.path.dirname(os.path.abspath(__file__))
COMP_DIR = os.environ.get("COMP7_DIR", os.path.join(HERE, "comp7"))
GRID_POINTS = 10
METHODS = {"FFD+SA-EDDTS": True, "FFD-EDDTS": False,   # name -> Stage 1 SA
           "FFD+SA-EDDTS+R": True}
WITH_R = {"FFD+SA-EDDTS+R"}
R_SHARE = 0.15        # as cpm_r.R_SHARE
MIN_BUDGET = 0.05     # [7] treats time_limit = 0 as "no limit"

_DC = None
_PARAMS = None


def _load():
    global _DC, _PARAMS
    if _DC is None:
        if COMP_DIR not in sys.path:
            sys.path.append(COMP_DIR)      # appended: never shadows our modules
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import dyeing_core
                from rerun import load_params
        except ModuleNotFoundError as e:
            raise SystemExit(
                f"the code of [7] needs the package '{e.name}': "
                f"pip install pyomo==6.9.5") from e
        _DC = dyeing_core
        _PARAMS = load_params(os.path.join(COMP_DIR, "tuned_params.json"))
    return _DC, _PARAMS


def to_comp7(inst: Instance):
    """Jobs, vessels and setup matrix in the format of [7]."""
    jobs = [{"id": j.id, "color": j.family, "customer": j.customer,
             "size": float(j.size), "due_date": float(j.due),
             "proc_time": float(inst.proc[j.family])} for j in inst.jobs]
    machines = [{"id": i + 1, "capacity": float(inst.capacity)}
                for i in range(inst.machines)]
    fams = sorted(inst.proc)
    # A3 (sigma_f), or the instance's own pair-dependent matrix in E2
    matrix = {(a, b): float(setup_time(inst, a, b))
              for a in fams for b in fams}
    return jobs, machines, matrix


def from_comp7(schedule):
    """Batches and machine sequences of this repository from a schedule of
    [7] (a list of vessels, each with its ordered batches)."""
    batches, sched, k = [], {}, 0
    for mi, vessel in enumerate(schedule):
        seq = []
        for b in vessel["batches"]:
            ids = tuple(j["id"] if isinstance(j, dict) else j
                        for j in b.get("jobs", []))
            batches.append(Batch(k, ids, b["family"], b["customer"]))
            seq.append(k)
            k += 1
        sched[mi] = seq
    return batches, sched


def _run(dc, params, jobs, machines, matrix, clock, t0, *, seed, use_sa,
         grid_points, run_seed=None):
    """[7]'s generate_pareto_front_epsilon on input in [7]'s format, with
    the clock split as described above; returns [7]'s raw output and info.
    `run_seed` sets [7]'s RUN_SEED (default: the seed), so that the seeding
    of [7]'s own runner can be reproduced (check_comp7.py)."""
    for k, v in params.items():
        setattr(dc, k, v)
    dc.STAGE1_USE_SA = bool(use_sa)
    dc.SETUP_MATRIX = matrix
    dc.MAX_SIGMA = max(matrix.values())
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    dc.RUN_SEED = seed if run_seed is None else run_seed
    dc.HEURISTIC_TIME_BUDGET = max(clock / grid_points, MIN_BUDGET)
    info = {"grid_points": grid_points, "stage1_seconds": 0.0}

    orig = dc.improved_simulated_annealing

    def timed_stage1(*a, **kw):
        out = orig(*a, **kw)
        used = time.time() - t0
        info["stage1_seconds"] = round(used, 3)
        dc.HEURISTIC_TIME_BUDGET = max((clock - used) / grid_points,
                                       MIN_BUDGET)
        return out

    dc.improved_simulated_annealing = timed_stage1
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = dc.generate_pareto_front_epsilon(
                jobs, machines, approach="FFD+SA-EDDTS",
                num_points=grid_points, setup_matrix=matrix)
    finally:
        dc.improved_simulated_annealing = orig
    info["subproblem_budget"] = round(dc.HEURISTIC_TIME_BUDGET, 4)
    stage1 = getattr(dc, "LAST_STAGE1_BATCHES", None)
    info["stage1_batches"] = len(stage1) if stage1 is not None else None
    return out, info


def comp7_front(inst: Instance, clock: float, *, seed: int = 0,
                use_sa: bool = True, grid_points: int = GRID_POINTS):
    """Run [7]'s method for `clock` seconds; return (front, info)."""
    dc, params = _load()
    t0 = time.time()
    jobs, machines, matrix = to_comp7(inst)
    out, info = _run(dc, params, jobs, machines, matrix, clock, t0,
                     seed=seed, use_sa=use_sa, grid_points=grid_points)
    front = out[0] if isinstance(out, tuple) else out

    sols, rejected = [], 0
    for s in front or []:
        batches, sched = from_comp7(s["schedule"])
        ev = evaluate_schedule(inst, batches, sched, strict=False)
        t, u = float(s["tardiness"]), float(s["setup_time"])
        if (not ev.feasible
                or abs(ev.z1 - t) > 1e-6 * max(1.0, abs(t))
                or abs(ev.z2 - u) > 1e-6 * max(1.0, abs(u))):
            rejected += 1
            continue
        sols.append(Solution(batches, sched, ev.z1, ev.z2))
    info["returned"] = len(front or [])
    info["rejected"] = rejected
    return filter_front(sols), info


def comp7_r_front(inst: Instance, clock: float, *, seed: int = 0,
                  use_sa: bool = True, rounds: int = 3):
    """[7] on (1 - R_SHARE) of the clock, then R on its front, exactly as
    CPM+R (alns_r.r_on_front; rounds = ALNS+R's frozen r_rounds)."""
    from alns_r import r_on_front
    from benchmarks import Budget
    t0 = time.time()
    front, info = comp7_front(inst, clock * (1 - R_SHARE), seed=seed,
                              use_sa=use_sa)
    left = max(clock - (time.time() - t0), 0.01)
    b = Budget(10 ** 9, seconds=left)
    out, rinfo = r_on_front(inst, front, b, rounds)
    info.update(rinfo)
    info["pre_r"] = [[s.z1, s.z2] for s in front]
    info["r_evaluations"] = b.used
    return out, info


if __name__ == "__main__":
    # quick check: python comp7.py
    from instances import build_matrix, generate_instance
    cells = [c for c in build_matrix(include_verify=True)
             if c.scale == "small"]
    inst = generate_instance(cells[1], 0.5, 4)
    for name, sa in METHODS.items():
        t = time.time()
        f, info = comp7_front(inst, 3.0, seed=4, use_sa=sa)
        print(f"{inst.name} {name}: {len(f)} points in "
              f"{time.time() - t:.1f} s, {info}")
        print("   ", [(round(s.z1, 2), round(s.z2, 3)) for s in f])
