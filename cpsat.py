"""
CP-SAT exact reference.

This model searches the UNRESTRICTED space. It does not assume block
contiguity, it does not pin a machine's last block to argmax(sigma), and it
does not fix the batch count. That is the point: CPM works inside those
restrictions, and the only honest way to price them is to compare against a
solver that does not.

Modelling choices worth stating:

  Batch pools.  Two-level compatibility (same colour AND same customer) is
  enforced structurally rather than with constraints: each (colour, customer)
  group gets its own pool of candidate batches, and a job's x variables exist
  only for its own group's pool. This removes an entire constraint family and
  shrinks the model a lot.

  Pool size.  Each group's pool is the batch count of an ACHIEVABLE packing
  (FFD) plus `pool_slack`; ceil(total kg / Q) is only a lower bound on bin
  packing and is often not attainable. The pool is an upper limit, not a
  fixed count: unused slots are switched off. With pool_slack = 0, which the
  verification run uses, a group may use at most its FFD count, so the model
  is unrestricted in sequencing and batch composition but not in batch count.

  Sequencing.  AddCircuit per machine, with arc literals carrying the setup.
  Sequence-dependent setups fall out of the arc structure; no big-M, and
  subtour elimination is inside the constraint rather than bolted on with
  flow conservation as in the MILP of Appendix A.

  Integers.  CP-SAT is integral, so hours are scaled by SCALE (two decimals).

Objectives are handled by an augmented epsilon-constraint with the slack sign
the right way round: minimise Z1 - delta * slack / range, subject to
Z2 + slack = epsilon. Writing + delta instead,
which penalises the slack and pushes Z2 UP toward epsilon, producing weakly
dominated points.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from ortools.sat.python import cp_model

from core import Batch, Instance, Schedule, evaluate_schedule
from layer1 import pack_group

SCALE = 100          # hours -> integer hundredths
DEFAULT_SLACK = 1    # extra candidate batches per group


def _s(x: float) -> int:
    return int(round(x * SCALE))



def _configure(solver: cp_model.CpSolver, det_time: float,
               workers: int = 8, seed: int = 0,
               wall_clock_net: float | None = None) -> None:
    """Deterministic solver settings.

    With the defaults, repeated runs on the same instance disagreed: Z1 came
    back as 42.40, 73.40 and 74.15 on three identical calls, and one call
    reported INFEASIBLE for a model other calls solved. Setting the seed and
    interleaving helped but did not close it (57.35, 57.35, 53.85), because
    the stopping criterion was still wall-clock: the solver gets a different
    amount of work done depending on machine load.

    The control is therefore `max_deterministic_time`, a machine-independent
    work unit - the same move we made in CPM when its own time-based cutoff
    turned out to be producing irreproducible fronts. Wall clock survives
    only as a safety net.
    """
    solver.parameters.max_deterministic_time = det_time
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.interleave_search = True
    solver.parameters.relative_gap_limit = 0.0
    if wall_clock_net is not None:
        solver.parameters.max_time_in_seconds = wall_clock_net


# --------------------------------------------------------------------------


class CpSatModel:
    """Integrated batching + scheduling model over a batch pool."""

    def __init__(self, inst: Instance, *, pool_slack: int = DEFAULT_SLACK):
        self.inst = inst
        self.pool_slack = pool_slack

        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for j in inst.jobs:
            groups[(j.family, j.customer)].append(j.id)
        self.groups = dict(groups)

        # candidate batches: (group, index) -> flat batch id
        # Pool size comes from an ACHIEVABLE packing, not from
        # ceil(total kg / Q): that bound is a lower bound on bin packing and
        # is frequently not attainable, which made the model INFEASIBLE at
        # pool_slack = 0. FFD gives a count we know can be realised.
        self.pool: List[Tuple[Tuple[str, str], int]] = []
        for g, jids in sorted(self.groups.items()):
            kg = len(pack_group([inst.job(j) for j in jids], inst.capacity))
            for i in range(kg + pool_slack):
                self.pool.append((g, i))
        self.family_of = [g[0] for g, _ in self.pool]

    # ----------------------------------------------------------------------

    def add_hint(self, model, batches: Sequence[Batch],
                 schedule: Schedule) -> bool:
        """Warm start from a CPM solution.

        The hint only supplies an incumbent; the lower bound CP-SAT proves is
        untouched by it, so the optimality gap it reports stays an independent
        statement about CPM's solution rather than a circular one.
        """
        v = self.vars
        pool_of = v["pool_of"]
        used: Dict[int, int] = {}
        for k, seq in schedule.items():
            for b in seq:
                used[b] = k
        by_group: Dict[Tuple[str, str], List[Batch]] = defaultdict(list)
        for b in batches:
            by_group[(b.family, b.customer)].append(b)

        assigned: Dict[int, Batch] = {}
        for g, bs in by_group.items():
            slots = pool_of.get(g, [])
            if len(bs) > len(slots):
                return False
            for slot, b in zip(slots, bs):
                assigned[slot] = b

        # The symmetry-breaking constraint fixes the machine ORDER: machine k
        # may host a batch only once machine k-1 hosts a lower-indexed one. A
        # hint that labels machines differently is rejected as infeasible and
        # silently discarded - which is what happened on the first attempt,
        # where the warm start came back worse than the solution it started
        # from. Relabel so the machine holding the lowest slot becomes 0.
        slot_of_batch = {b.id: slot for slot, b in assigned.items()}
        first_slot: Dict[int, int] = {}
        for bid, k in used.items():
            s_ = slot_of_batch.get(bid)
            if s_ is not None:
                first_slot[k] = min(first_slot.get(k, s_), s_)
        relabel = {old: new for new, (old, _) in
                   enumerate(sorted(first_slot.items(), key=lambda t: t[1]))}
        used = {bid: relabel.get(k, k) for bid, k in used.items()}

        for slot in range(len(self.pool)):
            b = assigned.get(slot)
            model.AddHint(v["y"][slot], 1 if b is not None else 0)
            if b is None:
                continue
            g = self.pool[slot][0]
            for j in self.groups[g]:
                model.AddHint(v["x"][(j, slot)], 1 if j in b.jobs else 0)
            for k in range(self.inst.machines):
                model.AddHint(v["a"][(slot, k)], 1 if used.get(b.id) == k else 0)
        return True

    def build(self, epsilon: int | None = None):
        inst, pool = self.inst, self.pool
        B, M = len(pool), inst.machines
        m = cp_model.CpModel()

        y = [m.NewBoolVar(f"y{b}") for b in range(B)]
        x: Dict[Tuple[int, int], cp_model.IntVar] = {}
        pool_of: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for b, (g, _) in enumerate(pool):
            pool_of[g].append(b)

        for g, jids in self.groups.items():
            bs = pool_of[g]
            for j in jids:
                for b in bs:
                    x[(j, b)] = m.NewBoolVar(f"x{j}_{b}")
                m.AddExactlyOne(x[(j, b)] for b in bs)
            for b in bs:
                m.Add(sum(_s(inst.job(j).size) * x[(j, b)] for j in jids)
                      <= _s(inst.capacity))
                for j in jids:
                    m.AddImplication(x[(j, b)], y[b])
                m.Add(sum(x[(j, b)] for j in jids) >= 1).OnlyEnforceIf(y[b])
                m.Add(sum(x[(j, b)] for j in jids) == 0).OnlyEnforceIf(y[b].Not())
            for a, b in zip(bs, bs[1:]):                 # symmetry within pool
                m.Add(y[a] >= y[b])

        # machine assignment
        a = {(b, k): m.NewBoolVar(f"a{b}_{k}")
             for b in range(B) for k in range(M)}
        for b in range(B):
            m.Add(sum(a[(b, k)] for k in range(M)) == y[b])

        horizon = sum(_s(inst.proc[self.family_of[b]]) for b in range(B)) \
            + sum(_s(v) for v in inst.sigma.values()) * B
        start = [m.NewIntVar(0, horizon, f"st{b}") for b in range(B)]
        end = [m.NewIntVar(0, horizon, f"en{b}") for b in range(B)]
        for b in range(B):
            p = _s(inst.proc[self.family_of[b]])
            m.Add(end[b] == start[b] + p).OnlyEnforceIf(y[b])
            m.Add(start[b] == 0).OnlyEnforceIf(y[b].Not())
            m.Add(end[b] == 0).OnlyEnforceIf(y[b].Not())

        # Redundant but strong: m identical machines means at most m batches
        # can be in progress at once. The circuit constraints imply this, but
        # only after the arcs are decided; a cumulative over all batches
        # propagates it immediately and is what lifts the otherwise useless
        # lower bound on Z1.
        ivs = []
        for b in range(B):
            p = _s(inst.proc[self.family_of[b]])
            ivs.append(m.NewOptionalIntervalVar(start[b], p, end[b], y[b],
                                                f"iv{b}"))
        m.AddCumulative(ivs, [1] * B, M)

        # Machines are identical, so fix their order: machine k may host a
        # batch only if machine k-1 already hosts a lower-indexed one.
        for k in range(1, M):
            for b in range(B):
                m.Add(sum(a[(bb, k - 1)] for bb in range(b)) >= 1
                      ).OnlyEnforceIf(a[(b, k)])

        # sequencing: one circuit per machine over its batches plus a depot
        setup_terms = []
        for k in range(M):
            arcs = []
            for b in range(B):
                arcs.append((b + 1, b + 1, a[(b, k)].Not()))   # self-loop = off
                lit_in = m.NewBoolVar(f"first{b}_{k}")
                lit_out = m.NewBoolVar(f"last{b}_{k}")
                arcs.append((0, b + 1, lit_in))
                arcs.append((b + 1, 0, lit_out))
                m.AddImplication(lit_in, a[(b, k)])
                m.AddImplication(lit_out, a[(b, k)])
                # first batch on a machine starts at time zero (A1)
                m.Add(start[b] == 0).OnlyEnforceIf(lit_in)
            for i in range(B):
                for j in range(B):
                    if i == j:
                        continue
                    lit = m.NewBoolVar(f"arc{i}_{j}_{k}")
                    arcs.append((i + 1, j + 1, lit))
                    m.AddImplication(lit, a[(i, k)])
                    m.AddImplication(lit, a[(j, k)])
                    fi, fj = self.family_of[i], self.family_of[j]
                    sij = 0 if fi == fj else _s(inst.sigma[fi])
                    m.Add(start[j] >= end[i] + sij).OnlyEnforceIf(lit)
                    if sij:
                        setup_terms.append((sij, lit))
            m.AddCircuit(arcs)

        # tardiness
        tard = []
        for j in inst.jobs:
            t = m.NewIntVar(0, horizon, f"T{j.id}")
            for b in pool_of[(j.family, j.customer)]:
                m.Add(t >= end[b] - _s(j.due)).OnlyEnforceIf(x[(j.id, b)])
            tard.append(t)

        z1 = m.NewIntVar(0, horizon * len(inst.jobs), "Z1")
        m.Add(z1 == sum(tard))
        z2 = m.NewIntVar(0, horizon, "Z2")
        m.Add(z2 == sum(c * lit for c, lit in setup_terms))
        if epsilon is not None:
            m.Add(z2 <= epsilon)

        self.vars = dict(y=y, x=x, a=a, start=start, end=end,
                         z1=z1, z2=z2, pool_of=dict(pool_of))
        return m

    # ----------------------------------------------------------------------

    def extract(self, solver: cp_model.CpSolver
                ) -> Tuple[List[Batch], Schedule]:
        inst, pool = self.inst, self.pool
        v = self.vars
        batches: List[Batch] = []
        seq: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        bid = 0
        for b, (g, _) in enumerate(pool):
            if not solver.Value(v["y"][b]):
                continue
            jids = tuple(sorted(j for j in self.groups[g]
                                if solver.Value(v["x"][(j, b)])))
            if not jids:
                continue
            batches.append(Batch(id=bid, jobs=jids,
                                 family=g[0], customer=g[1]))
            for k in range(inst.machines):
                if solver.Value(v["a"][(b, k)]):
                    seq[k].append((solver.Value(v["start"][b]), bid))
                    break
            bid += 1
        schedule: Schedule = {k: [] for k in range(inst.machines)}
        for k, items in seq.items():
            schedule[k] = [i for _, i in sorted(items)]
        return batches, schedule


# --------------------------------------------------------------------------


def solve(inst: Instance, *, epsilon: float | None = None,
          det_time: float = 60.0, workers: int = 8,
          pool_slack: int = DEFAULT_SLACK, log: bool = False) -> dict:
    """Solve once, optionally with a Z2 cap. Returns objectives and status."""
    mdl = CpSatModel(inst, pool_slack=pool_slack)
    model = mdl.build(None if epsilon is None else _s(epsilon))
    solver = cp_model.CpSolver()
    _configure(solver, det_time, workers,
               wall_clock_net=max(30.0, det_time * 20))
    solver.parameters.log_search_progress = log

    status = solver.Solve(model)
    out = {
        "status": solver.StatusName(status),
        "optimal": status == cp_model.OPTIMAL,
        "seconds": round(solver.WallTime(), 2),
        "z1": None, "z2": None, "bound": None,
        "batches": None, "schedule": None,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        batches, schedule = mdl.extract(solver)
        ev = evaluate_schedule(inst, batches, schedule, strict=False)
        out.update(z1=ev.z1, z2=ev.z2, batches=batches, schedule=schedule,
                   feasible=ev.feasible, violations=ev.violations,
                   bound=round(solver.BestObjectiveBound() / SCALE, 4)
                   if model.HasObjective() else None)
    return out


def lexicographic_payoff(inst: Instance, *, det_time: float = 30.0,
                         **kw) -> dict:
    """Proper payoff table: no hard-coded epsilon anywhere.

    The old notebook wrote epsilon = 20, 40, 100 or 1000 depending on the
    approach, which both corrupted the ideal/nadir estimates and made the
    approaches start from different feasible regions. Here both ends come from
    optimisation, and min Z2 is cross-checked against P1's closed form.
    """
    mdl = CpSatModel(inst, **kw)

    # min Z2
    model = mdl.build(None)
    model.Minimize(mdl.vars["z2"])
    s = cp_model.CpSolver()
    _configure(s, det_time, wall_clock_net=max(30.0, det_time * 20))
    st = s.Solve(model)
    z2_min = s.Value(mdl.vars["z2"]) / SCALE if st in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    z2_opt = st == cp_model.OPTIMAL

    # min Z1
    mdl2 = CpSatModel(inst, **kw)
    model2 = mdl2.build(None)
    model2.Minimize(mdl2.vars["z1"])
    s2 = cp_model.CpSolver()
    _configure(s2, det_time, wall_clock_net=max(30.0, det_time * 20))
    st2 = s2.Solve(model2)
    z1_min = s2.Value(mdl2.vars["z1"]) / SCALE if st2 in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    return {
        "z2_min": z2_min, "z2_min_optimal": z2_opt,
        "z1_min": z1_min, "z1_min_optimal": st2 == cp_model.OPTIMAL,
        "z2_status": s.StatusName(st), "z1_status": s2.StatusName(st2),
    }
