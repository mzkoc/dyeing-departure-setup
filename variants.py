"""Instance variants for the sensitivity runs (ANALYSIS_PLAN.md, section 9,
amendments 3 and 4) and the larger scales for the scalability runs.

One factor at a time, on the cells of the validation set with colour excess
e > 0 at 60 jobs (m = 3) and 150 jobs (m = 4), all three tightness levels,
replications 4-9 (18 instances per scale). Every variant keeps the seed of
the base instance (common random numbers), so that it differs from the base
only through the factor named:

  perturb   the base instance; after the run every schedule of every front
            is evaluated on five perturbed instances, each family's p_f and
            sigma_f multiplied by U[0.8, 1.2] (draws seeded by the instance
            name and the draw number). Methods plan on the nominal data.
            The nominal fronts serve as the base of the comparisons.
  sigx2     every sigma_f doubled (setup/processing ratio 0.13-0.19)
  sigx4     every sigma_f times four (0.25-0.39)
  skew      customer shares proportional to 1/k (k = 1..|C|) instead of equal
  cluster   due dates in two clusters, at 0.25 and 0.75 of the Fisher window,
            each job in either at random and jittered by +/-0.05 of the
            window (same mean as the Fisher draw)
  c5        five customers instead of three
  q300      vessel capacity Q = 300 kg (job sizes unchanged, U[50, 300])
  size      job sizes U[100, 350] kg (Q = 400 kg)

Scales x200 and x300: 200 and 300 jobs, |F| = 8, |C| = 3, m in {4, 8, 12};
run at tau = 0.5.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from core import Instance
from instances import Cell, build_matrix, generate_instance

VARIANTS = ("perturb", "sigx2", "sigx4", "skew", "cluster", "c5", "q300",
            "size")
PERTURB_DRAWS = 5
PERTURB_RANGE = (0.8, 1.2)
CLUSTERS = ((0.25, 0.75), 0.05)
EXTRA_SCALES = {"x200": (200, 8, 3, (4, 8, 12)),
                "x300": (300, 8, 3, (4, 8, 12))}


def cells(scale: str):
    """Cells of a scale, the extra scales included."""
    if scale in EXTRA_SCALES:
        n, nf, nc, arms = EXTRA_SCALES[scale]
        return [Cell(scale, n, nf, nc, m) for m in arms]
    return [c for c in build_matrix(include_verify=True) if c.scale == scale]


def make_instance(cell: Cell, tau: float, rep: int,
                  variant: str = "") -> Instance:
    if not variant:
        return generate_instance(cell, tau, rep)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if variant == "c5":
        inst = generate_instance(replace(cell, n_customers=5), tau, rep)
    elif variant == "skew":
        k = cell.n_customers
        inst = generate_instance(cell, tau, rep, customer_weights=[
            1.0 / (i + 1) for i in range(k)])
    elif variant == "cluster":
        inst = generate_instance(cell, tau, rep, due_clusters=CLUSTERS)
    elif variant == "q300":
        inst = generate_instance(cell, tau, rep, capacity=300.0)
    elif variant == "size":
        inst = generate_instance(cell, tau, rep, size_range=(100.0, 350.0))
    else:
        inst = generate_instance(cell, tau, rep)
        if variant in ("sigx2", "sigx4"):
            f = 2.0 if variant == "sigx2" else 4.0
            inst = Instance(jobs=inst.jobs, machines=inst.machines,
                            capacity=inst.capacity, proc=dict(inst.proc),
                            sigma={a: round(s * f, 4)
                                   for a, s in inst.sigma.items()},
                            name=inst.name, meta=dict(inst.meta))
    inst.name = f"{inst.name}_{variant}"
    inst.meta = dict(inst.meta, variant=variant)
    return inst


def perturbed(inst: Instance, draw: int) -> Instance:
    """The instance with each family's p_f and sigma_f multiplied by an
    independent U[0.8, 1.2] factor; seeded by (name, draw)."""
    seed = int(hashlib.sha256(f"{inst.name}|perturb|{draw}".encode())
               .hexdigest()[:8], 16)
    rng = random.Random(seed)
    lo, hi = PERTURB_RANGE
    proc, sigma = {}, {}
    for f in sorted(inst.proc):
        proc[f] = inst.proc[f] * rng.uniform(lo, hi)
        sigma[f] = inst.sigma[f] * rng.uniform(lo, hi)
    return Instance(jobs=inst.jobs, machines=inst.machines,
                    capacity=inst.capacity, proc=proc, sigma=sigma,
                    name=f"{inst.name}|d{draw}", meta=dict(inst.meta))


def perturbed_points(inst: Instance, front):
    """[[ [z1, z2] for each schedule of the front ] for each draw]: the
    planned sequences, started as early as possible on the perturbed data."""
    from core import evaluate_schedule
    out = []
    for d in range(PERTURB_DRAWS):
        pi = perturbed(inst, d)
        row = []
        for s in front:
            ev = evaluate_schedule(pi, s.batches, s.schedule, strict=False)
            row.append([round(ev.z1, 6), round(ev.z2, 6)])
        out.append(row)
    return out
