"""
Core data structures and the single source of truth for objective evaluation.

Every solution method in this project - CPM, CP-SAT, MILP, ALNS, NSGA-II,
LPT+EDD - must report its objectives through `evaluate_schedule`. No method
computes its own Z1 or Z2. Two algorithms must never be able to disagree
about the value of the same schedule.

Model assumptions (see design doc):
  A1  m identical machines, continuously available, clean at t = 0.
      No setup before the first batch on a machine.
  A2  Batch processing time = p_f. Independent of batch contents/fill.
  A3  Setup depends only on the departing (FROM) colour:
      sigma(f1, f2) = 0 if f1 == f2, else sigma[f1].
  A4  WCE_total = kappa * Z2.
  A5  Deterministic due dates and processing times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Machine = int
BatchId = int
JobId = int


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    id: JobId
    size: float           # kg
    family: str           # colour family
    customer: str
    due: float            # hours


@dataclass
class Instance:
    jobs: List[Job]
    machines: int
    capacity: float                 # Q, kg
    proc: Dict[str, float]          # p_f, hours, per colour family
    sigma: Dict[str, float]         # setup when LEAVING colour f, hours
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def families(self) -> List[str]:
        return sorted(self.proc)

    @property
    def customers(self) -> List[str]:
        return sorted({j.customer for j in self.jobs})

    def job(self, jid: JobId) -> Job:
        return self._index[jid]

    def __post_init__(self) -> None:
        self._index = {j.id: j for j in self.jobs}

    @property
    def excess(self) -> int:
        """e = |F| - m. Drives the Z2 regime (see P1, Corollaries 2 and 3)."""
        return len(self.proc) - self.machines


@dataclass
class Batch:
    id: BatchId
    jobs: Tuple[JobId, ...]
    family: str
    customer: str

    def size(self, inst: Instance) -> float:
        return sum(inst.job(j).size for j in self.jobs)

    def proc(self, inst: Instance) -> float:
        return inst.proc[self.family]        # A2


# A schedule is machine -> ordered list of batch ids.
Schedule = Dict[Machine, List[BatchId]]


@dataclass
class Evaluation:
    z1: float                       # total tardiness
    z2: float                       # total setup time
    makespan: float
    feasible: bool
    violations: List[str]
    completion: Dict[BatchId, float]
    start: Dict[BatchId, float]
    setups: List[Tuple[Machine, BatchId, BatchId, float]]

    def objectives(self) -> Tuple[float, float]:
        return (self.z1, self.z2)


# --------------------------------------------------------------------------
# Setup rule (A3)
# --------------------------------------------------------------------------


def setup_time(inst: Instance, from_family: str, to_family: str) -> float:
    """Cleaning depends on what is in the vessel, not on what comes next."""
    if from_family == to_family:
        return 0.0
    return inst.sigma[from_family]


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


def evaluate_schedule(
    inst: Instance,
    batches: Sequence[Batch],
    schedule: Schedule,
    *,
    strict: bool = True,
) -> Evaluation:
    """Evaluate a schedule. The only place Z1 and Z2 are ever computed.

    `strict=True` raises on infeasibility; `strict=False` reports it in the
    returned Evaluation so callers can inspect broken solutions.
    """
    by_id = {b.id: b for b in batches}
    violations: List[str] = []

    # ---- structural checks -------------------------------------------------
    scheduled: List[BatchId] = []
    for m, seq in schedule.items():
        if not (0 <= m < inst.machines):
            violations.append(f"machine index {m} out of range")
        scheduled.extend(seq)

    if len(scheduled) != len(set(scheduled)):
        violations.append("a batch appears on more than one machine or twice")

    missing = set(by_id) - set(scheduled)
    if missing:
        violations.append(f"batches not scheduled: {sorted(missing)}")

    unknown = set(scheduled) - set(by_id)
    if unknown:
        violations.append(f"unknown batch ids in schedule: {sorted(unknown)}")

    # ---- batch validity: coverage, capacity, two-level compatibility -------
    covered: List[JobId] = []
    for b in batches:
        covered.extend(b.jobs)
        if not b.jobs:
            violations.append(f"batch {b.id} is empty")
            continue
        if b.size(inst) > inst.capacity + 1e-9:
            violations.append(
                f"batch {b.id} exceeds capacity: "
                f"{b.size(inst):.1f} > {inst.capacity:.1f}"
            )
        for jid in b.jobs:
            j = inst.job(jid)
            if j.family != b.family:
                violations.append(f"job {jid} colour != batch {b.id} colour")
            if j.customer != b.customer:
                violations.append(f"job {jid} customer != batch {b.id} customer")

    if sorted(covered) != sorted(j.id for j in inst.jobs):
        violations.append("jobs are not partitioned exactly once into batches")

    if violations and strict:
        raise ValueError("infeasible schedule:\n  " + "\n  ".join(violations))

    # ---- timing ------------------------------------------------------------
    start: Dict[BatchId, float] = {}
    completion: Dict[BatchId, float] = {}
    setups: List[Tuple[Machine, BatchId, BatchId, float]] = []
    z2 = 0.0

    for m, seq in schedule.items():
        t = 0.0
        prev_family = None
        for bid in seq:
            b = by_id.get(bid)
            if b is None:
                continue
            if prev_family is not None:               # A1: none before first
                s = setup_time(inst, prev_family, b.family)
                if s > 0.0:
                    setups.append((m, seq[seq.index(bid) - 1], bid, s))
                    z2 += s
                t += s
            start[bid] = t
            t += b.proc(inst)
            completion[bid] = t
            prev_family = b.family

    # ---- objectives --------------------------------------------------------
    z1 = 0.0
    for b in batches:
        c = completion.get(b.id)
        if c is None:
            continue
        for jid in b.jobs:
            z1 += max(0.0, c - inst.job(jid).due)

    makespan = max(completion.values(), default=0.0)

    return Evaluation(
        z1=round(z1, 6),
        z2=round(z2, 6),
        makespan=round(makespan, 6),
        feasible=not violations,
        violations=violations,
        completion=completion,
        start=start,
        setups=setups,
    )


# --------------------------------------------------------------------------
# P1: closed form and bounds
# --------------------------------------------------------------------------


def z2_machine_closed_form(inst: Instance, colours: Sequence[str]) -> float:
    """P1: under contiguity, Z2(m) = sum(sigma_f) - max(sigma_f) over F_m."""
    if not colours:
        return 0.0
    vals = [inst.sigma[f] for f in set(colours)]
    return sum(vals) - max(vals)


def z2_lower_bound(inst: Instance) -> float:
    """P1, Corollaries 2 and 3.

    |F| <= m  ->  0 (every colour gets its own machine, and is its own last
                  block, so no transition is ever paid).
    |F| >  m  ->  sum of the e = |F| - m smallest sigma values. Each colour is
                  kept whole on one machine; the m largest-sigma colours become
                  the m machines' last blocks.
    This bound is attainable, so it is the exact optimum of Z2.
    """
    sig = sorted(inst.sigma[f] for f in inst.families)
    e = len(sig) - inst.machines
    if e <= 0:
        return 0.0
    return sum(sig[:e])
