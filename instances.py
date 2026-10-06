"""
Instance generator for the experimental matrix.

Design choices, each of which determines whether a factor is exercised:

  due dates        Fisher construction with tau in {0.3, 0.5, 0.7}, R = 0.4,
                   the makespan proxy estimated at batch level (A2)
  colour excess    the machine count is the experimental arm, so e = |F| - m
                   sweeps negative, zero and positive within a scale
  batching         capacity 400 kg against job sizes U[50, 300] kg, so most
                   (colour, customer) groups need more than one batch

Group density. Each customer draws its jobs from a preferred subset of about
half the colours, so the number of non-empty (colour, customer) groups is
smaller than |F| * |C| and the realised density is higher than n / (|F| |C|).
`Cell.group_density` reports the nominal figure; `realised_density` below
reports the one the instances actually have.

Every replication gets its own random.Random instance seeded from a stable
digest of (cell, tau, replication), so instances are identical across
sessions and independent of the order in which methods run.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
import random
from dataclasses import dataclass
from typing import Dict, List

from core import Instance, Job

# --------------------------------------------------------------------------
# Fixed model parameters
# --------------------------------------------------------------------------

CAPACITY = 400.0                 # Q, kg
JOB_SIZE_RANGE = (50.0, 300.0)   # kg, uniform
DUE_DATE_RANGE_R = 0.4           # Fisher R
LIQUOR_RATIO = 8.0               # L of bath per kg of fabric (1:8)

# Colour families are tone groups, not individual shades - incompatibility in
# a dyehouse works at this level. Ordered light to dark.
#
# Processing and cleaning times come from dyehouse practice, NOT from a
# formula. An earlier version set sigma_f = p_f / 10, which made the two
# objectives two functions of a single parameter - a reviewer would rightly
# ask whether they are independent at all. Here they are separate physical
# quantities: p_f is a dyeing cycle, sigma_f is a vessel wash-out.
#
# The sigma/p ratio is not constant: 0.063 for the lightest tone, 0.093 for
# black. Deeper shades take disproportionately longer to wash out, which the
# old fixed ratio could not express.
#
#                 p_f (h)  sigma_f (h)   sigma (min)   sigma/p
FAMILY_POOL = [
    ("Ecru",   4.0, 15 / 60),        #  15          0.063
    ("Pastel", 4.5, 20 / 60),        #  20          0.074
    ("Light",  5.0, 25 / 60),        #  25          0.083
    ("Medium", 5.5, 30 / 60),        #  30          0.091
    ("Brown",  6.0, 35 / 60),        #  35          0.097
    ("Dark",   7.0, 40 / 60),        #  40          0.095
    ("Navy",   8.0, 45 / 60),        #  45          0.094
    ("Black",  9.0, 50 / 60),        #  50          0.093
]

# Cleaning water per hour of setup.
#
#   bath volume        V = Q * LIQUOR_RATIO = 400 * 8 = 3200 L
#   lightest colour    sigma_min = 15 min, taken as one full bath exchange
#   coefficient        kappa = V / sigma_min = 3200 / 0.25 h = 12800 L/h
#
# No free parameter: capacity from the model, liquor ratio from the
# literature (1:6 to 1:15 is the usual exhaust-dyeing range), and the
# reference point from the dyehouse's own shortest wash-out. Water is then
# strictly proportional to Z2, so minimising setup hours and minimising
# cleaning water are the SAME problem - the conversion is a reporting step,
# not a second objective, and P1 carries over untouched.
WATER_PER_SETUP_HOUR = (CAPACITY * LIQUOR_RATIO) / (15 / 60)   # L per hour


def cleaning_water_litres(z2_hours: float) -> float:
    """Convert a setup-time objective into cleaning water.

    Note on scope: this is NOT total dyeing water. Process water - roughly
    100-180 L per kg of fabric - is fixed by the order book and no schedule
    changes it. Cleaning water is the one stream scheduling controls; under
    this model it is a small fraction of the total.
    """
    return z2_hours * WATER_PER_SETUP_HOUR


# --------------------------------------------------------------------------
# Experimental matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    scale: str
    n: int
    n_families: int
    n_customers: int
    machines: int

    @property
    def excess(self) -> int:
        return self.n_families - self.machines

    @property
    def group_density(self) -> float:
        return self.n / (self.n_families * self.n_customers)

    @property
    def label(self) -> str:
        return (f"{self.scale[:2]}_n{self.n}_m{self.machines}"
                f"_f{self.n_families}_c{self.n_customers}")


def build_matrix(include_verify: bool = False) -> List[Cell]:
    """The approved matrix: 4 scales x 3 machine counts.

    `include_verify` prepends a verification scale. CP-SAT proves optimality
    up to about 18 jobs and is useless by 30 - the smallest cell of the main
    matrix - so without these cells there is nowhere the method's distance
    from optimality can actually be measured. They exist to answer "is CPM
    right", not "does CPM scale", and they are reported separately.
    """
    spec = [
        # scale,        n,   |F|, |C|, machine arm
        ("small",       30,  4,   2,   (2, 4, 6)),
        ("medium",      60,  5,   3,   (3, 5, 7)),
        ("large",      100,  6,   3,   (3, 6, 9)),
        ("xlarge",     150,  8,   3,   (4, 8, 12)),
    ]
    if include_verify:
        spec = [
            ("verify",  12,  2,   2,   (1, 2, 3)),
            ("verify",  16,  3,   2,   (2, 3, 4)),
            ("verify",  20,  3,   2,   (2, 3, 4)),
        ] + spec
    cells: List[Cell] = []
    for scale, n, nf, nc, arms in spec:
        for m in arms:
            cells.append(Cell(scale, n, nf, nc, m))
    return cells


# tau = 0.8 with R = 0.4 makes the Fisher base P(1 - tau - R/2) exactly zero:
# due dates collapse into [0, 0.4P], the max(1.0, .) floor starts truncating,
# and 89-95% of jobs are late whatever the schedule - Z1 saturates and
# differences between methods compress. Measured late fractions:
#   tau  0.3   0.4   0.5   0.6   0.7   0.8
#        46%   52%   64%   74%   84%   92%
# {0.3, 0.5, 0.7} keeps the base positive (0.5P, 0.3P, 0.1P) and spreads the
# regimes more evenly while still binding at the loose end.
TAU_LEVELS = (0.3, 0.5, 0.7)
REPLICATIONS = 10


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _due_dates_fisher(rng: random.Random, jobs_meta: List[dict],
                      proc: Dict[str, float], machines: int,
                      tau: float, R: float = DUE_DATE_RANGE_R) -> List[float]:
    """Fisher due dates: d_j = P(1 - tau - R/2) + P*R*U(0,1).

    P is the makespan proxy. It is estimated at BATCH level, not job level:
    processing time is per batch (A2), so summing p_f over jobs would
    overstate the horizon badly and make every due date slack.
    """
    # Estimate batch count per (family, customer) group from total kg.
    groups: Dict[tuple, float] = {}
    for jm in jobs_meta:
        key = (jm["family"], jm["customer"])
        groups[key] = groups.get(key, 0.0) + jm["size"]

    total_proc = 0.0
    for (f, _), kg in groups.items():
        n_batches = max(1, math.ceil(kg / CAPACITY))
        total_proc += n_batches * proc[f]

    P = total_proc / machines
    base = P * (1.0 - tau - R / 2.0)
    span = P * R
    return [max(1.0, round(base + span * rng.random(), 1))
            for _ in jobs_meta]


def generate_instance(cell: Cell, tau: float, replication: int,
                      seed_base: int = 20260923) -> Instance:
    """Generate one instance. Deterministic in (cell, tau, replication).

    The seed uses a stable digest, NOT Python's built-in hash(): string
    hashing is salted per process unless PYTHONHASHSEED is set, so an earlier
    version silently produced a different instance in every session. Runs that
    looked like they disagreed about an algorithm were in fact solving
    different problems.
    """
    tag = f"{cell.label}|{tau}|{replication}|{seed_base}"
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    pool = FAMILY_POOL[:cell.n_families]
    families = [name for name, _, _ in pool]
    proc = {name: p for name, p, _ in pool}
    sigma = {name: round(s, 4) for name, _, s in pool}
    customers = [f"C{i+1}" for i in range(cell.n_customers)]

    # Customers have colour preferences, but every colour must appear so the
    # e = |F| - m regime is the one the cell intends.
    prefs = {c: rng.sample(families, k=max(2, len(families) // 2))
             for c in customers}

    lo, hi = JOB_SIZE_RANGE
    meta: List[dict] = []
    for i in range(cell.n):
        cust = rng.choice(customers)
        fam = rng.choice(prefs[cust])
        meta.append({
            "id": i,
            "size": round(rng.uniform(lo, hi), 1),
            "family": fam,
            "customer": cust,
        })

    # Force every colour to be present. The experimental factor is
    # e = |F| - m, so a colour that receives no jobs silently shifts a cell
    # off its intended regime - one xlarge instance labelled e = -4 was
    # really e = -5. An earlier version reassigned a RANDOM job per missing
    # colour, which could pick the same job twice and undo its own fix; the
    # victims are now drawn without replacement and from the most common
    # colours, so no other colour is emptied in the process.
    for _ in range(len(families)):
        counts = Counter(m_["family"] for m_ in meta)
        missing = [f for f in families if counts.get(f, 0) == 0]
        if not missing:
            break
        for f in missing:
            donor = max(counts, key=lambda c: (counts[c], c))
            if counts[donor] <= 1:
                break
            idx = next(i for i, m_ in enumerate(meta)
                       if m_["family"] == donor)
            meta[idx]["family"] = f
            counts[donor] -= 1
            counts[f] = 1
            owner = meta[idx]["customer"]
            if f not in prefs[owner]:
                prefs[owner].append(f)

    dues = _due_dates_fisher(rng, meta, proc, cell.machines, tau)
    jobs = [Job(id=m_["id"], size=m_["size"], family=m_["family"],
                customer=m_["customer"], due=d)
            for m_, d in zip(meta, dues)]

    return Instance(
        jobs=jobs,
        machines=cell.machines,
        capacity=CAPACITY,
        proc=proc,
        sigma=sigma,
        name=f"{cell.label}_t{tau}_r{replication}",
        meta={
            "scale": cell.scale,
            "tau": tau,
            "replication": replication,
            "excess": cell.excess,
            "group_density": round(cell.group_density, 2),
            "seed": seed,
        },
    )


def realised_density(inst: Instance) -> float:
    """Jobs per NON-EMPTY (colour, customer) group."""
    groups = {(j.family, j.customer) for j in inst.jobs}
    return len(inst.jobs) / len(groups)


def generate_all(cells: List[Cell] | None = None) -> List[Instance]:
    cells = cells or build_matrix()
    out: List[Instance] = []
    for cell in cells:
        for tau in TAU_LEVELS:
            for rep in range(REPLICATIONS):
                out.append(generate_instance(cell, tau, rep))
    return out


# --------------------------------------------------------------------------


def describe_matrix() -> None:
    cells = build_matrix()
    print(f"{'cell':>22} {'n':>4} {'|F|':>4} {'|C|':>4} {'m':>3} "
          f"{'e':>3} {'grp/job':>8}")
    print("-" * 56)
    for c in cells:
        print(f"{c.label:>22} {c.n:>4} {c.n_families:>4} {c.n_customers:>4} "
              f"{c.machines:>3} {c.excess:>+3} {c.group_density:>8.2f}")
    print("-" * 56)
    total = len(cells) * len(TAU_LEVELS) * REPLICATIONS
    print(f"{len(cells)} cells x {len(TAU_LEVELS)} tau x "
          f"{REPLICATIONS} reps = {total} instances")


if __name__ == "__main__":
    describe_matrix()
