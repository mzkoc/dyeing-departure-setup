"""Local, parallel runner for the revision experiments.

Every experiment is a list of independent tasks (one instance, one
replication, one or more methods). Tasks run in separate worker processes,
each pinned to one CPU core, and each finished task is written to its own JSON
file, so an interrupted run resumes where it stopped.

    python rerun.py cores                       # show the detected core types
    python rerun.py smoke                       # 1-2 minute end-to-end test
    python rerun.py verify --archive ../zenodo  # legacy mode vs the archive
    python rerun.py taguchi                     # parameter screening
    python analyse.py taguchi                   # -> tuned_params.json
    python rerun.py campaign --params tuned_params.json --reps 5
    python rerun.py sensitivity --params tuned_params.json --cores e
    python rerun.py groups      --params tuned_params.json --cores e
    python rerun.py grid        --params tuned_params.json
    python rerun.py ablation    --params tuned_params.json

Add --dry-run to any experiment to see the task count and time estimate.
"""
import os

# One thread per worker: numpy/BLAS must not spawn threads of its own, or the
# workers compete for cores and the wall-clock budgets stop being equal.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import platform
import sys
import time
import traceback
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# The values the archived campaign was configured with (notebook cell 11).
ARCHIVE_PARAMS = {
    "SA_COOLING_RATE": 0.95, "TABU_TENURE": 20, "EDDTS_PENALTY": 25.0,
    "ALNS_DECAY": 0.9, "ALNS_MAX_ITERATIONS": 2000,
    "ALNS_PENALTY_START": 500.0, "ALNS_PENALTY_END": 5000.0,
    "NSGA2_POP_SIZE": 100, "NSGA2_CROSSOVER_PROB": 0.7, "NSGA2_ETA_C": 10,
    "NSGA2_ETA_M": 20,
}
LEGACY_FLAGS = {
    "FIX_ALNS_DECAY": False, "FIX_NSGA2_PARAMS": False,
    "NSGA2_TIME_MATCHED": False, "FIX_ALNS_PAYOFF_EPS": False,
    "FIX_EQUAL_PAYOFF_BUDGET": False, "RUN_SEED": 0,
    "EDDTS_RESTARTS": True, "EDDTS_MAX_ITER_OVERRIDE": None,
    "STAGE1_USE_SA": True, "EDDTS_RANDOM_FAMILY_ORDER": False,
}
REVISION_FLAGS = dict(LEGACY_FLAGS, FIX_ALNS_DECAY=True, FIX_NSGA2_PARAMS=True,
                      NSGA2_TIME_MATCHED=True, FIX_ALNS_PAYOFF_EPS=True,
                      FIX_EQUAL_PAYOFF_BUDGET=True,
                      EDDTS_RANDOM_FAMILY_ORDER=True)

MACHINES = {30: 3, 50: 4, 80: 5, 100: 6}
BUDGET = {30: 8.0, 50: 12.0, 80: 16.0, 100: 20.0}
PRESETS = ["LOOSE", "MEDIUM", "TIGHT"]
SIZES = [30, 50, 80, 100]
SEEDS = [11, 22, 33, 44, 55]
METHODS = ["FFD+SA-EDDTS", "ALNS", "NSGA-II", "LPT+EDD"]
GRID_POINTS = 10

L9 = [[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 1], [1, 1, 2],
      [1, 2, 0], [2, 0, 2], [2, 1, 0], [2, 2, 1]]
TAGUCHI_FACTORS = {
    "FFD+SA-EDDTS": [("SA_COOLING_RATE", [0.90, 0.95, 0.99]),
                     ("TABU_TENURE", [10, 15, 20]),
                     ("EDDTS_PENALTY", [1.0, 5.0, 25.0])],
    "ALNS": [("ALNS_DECAY", [0.7, 0.8, 0.9]),
             ("ALNS_MAX_ITERATIONS", [1000, 2000, 4000]),
             ("ALNS_PENALTY_START", [20.0, 100.0, 500.0])],
    "NSGA-II": [("NSGA2_POP_SIZE", [50, 100, 200]),
                ("NSGA2_CROSSOVER_PROB", [0.7, 0.8, 0.9]),
                ("NSGA2_ETA_C", [10, 20, 30])],
}
# Tuning instances: seeds disjoint from the test seeds 11-55. n = 80 is added
# so that the settings are not chosen on small instances only.
TAGUCHI_INSTANCES = [("MEDIUM", 30, 101), ("MEDIUM", 50, 202),
                     ("TIGHT", 30, 303), ("TIGHT", 50, 101),
                     ("LOOSE", 30, 202), ("LOOSE", 50, 303),
                     ("LOOSE", 80, 404), ("MEDIUM", 80, 505),
                     ("TIGHT", 80, 606)]

SENS_CONFIGS = [(0.0, 1.0), (0.0, 1.8), (0.5, 1.8), (1.0, 1.0),
                (1.0, 1.4), (1.0, 1.8), (2.0, 1.8)]


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def load_params(path, legacy=False):
    params = dict(ARCHIVE_PARAMS)
    params.update(LEGACY_FLAGS if legacy else REVISION_FLAGS)
    if path:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        params.update(user.get("globals", user))
    return params


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def make_task(exp, variant, preset, size, seed, run, rep, methods, params,
              *, customers=3, colors=8, machines=None, budget=None,
              grid_points=GRID_POINTS, lam=1.0, rho_h=1.8, overrides=None,
              seed_mode="run"):
    g = dict(params)
    g.update(overrides or {})
    m = machines or MACHINES[size]
    b = budget if budget is not None else BUDGET[size]
    key = f"{variant}__{preset}_{size}_{seed}_r{run}_rep{rep}".replace("+", "p")
    return {
        "exp": exp, "variant": variant, "key": key,
        "preset": preset, "size": size, "base_seed": seed, "run": run,
        "gen_seed": seed * 1000 + run, "rep": rep, "methods": list(methods),
        "customers": customers, "colors": colors, "machines": m,
        "capacity": 600, "budget": b, "grid_points": grid_points,
        "lam": lam, "rho_h": rho_h, "globals": g, "seed_mode": seed_mode,
    }


def est_seconds(task):
    per = task["grid_points"] * task["budget"]
    return sum(per if m != "LPT+EDD" else 0.2 for m in task["methods"])


def build_experiment(name, args):
    params = load_params(args.params, legacy=(name == "verify"))
    reps = args.reps
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    tasks = []
    if name == "smoke":
        tasks.append(make_task("smoke", "smoke", "MEDIUM", 30, 11, 0, 0,
                               METHODS, params, budget=1.0, grid_points=4))
    elif name == "verify":
        for key in args.instances.split(","):
            preset, size, seed, run = key.split("_")
            tasks.append(make_task("verify", "legacy", preset, int(size),
                                   int(seed), int(run), 0, METHODS, params,
                                   seed_mode="legacy"))
    elif name == "campaign":
        for preset, size, seed, run, rep in itertools.product(
                PRESETS, SIZES, seeds, (0, 1), range(reps)):
            tasks.append(make_task("campaign", "main", preset, size, seed,
                                   run, rep, METHODS, params))
    elif name == "nosa":
        # Second review round: the main campaign repeated for the proposed
        # method without the annealing step of Stage 1 (FFD+EDDTS), on the same
        # instances, with the same replications, run seeds and budgets.
        for preset, size, seed, run, rep in itertools.product(
                PRESETS, SIZES, seeds, (0, 1), range(reps)):
            tasks.append(make_task("nosa", "FFD-EDDTS", preset, size, seed,
                                   run, rep, ["FFD+SA-EDDTS"], params,
                                   overrides={"STAGE1_USE_SA": False}))
    elif name == "taguchi":
        for method, factors in TAGUCHI_FACTORS.items():
            for row, levels in enumerate(L9):
                ov = {g: vals[li] for (g, vals), li in zip(factors, levels)}
                if "ALNS_PENALTY_START" in ov:
                    ov["ALNS_PENALTY_END"] = 10.0 * ov["ALNS_PENALTY_START"]
                for preset, size, seed in TAGUCHI_INSTANCES:
                    for run in (0, 1):
                        t = make_task("taguchi", f"{method}_L{row}", preset,
                                      size, seed, run, 0, [method], params,
                                      overrides=ov)
                        t["taguchi"] = {"method": method, "row": row,
                                        "levels": levels, "values": ov}
                        tasks.append(t)
    elif name == "sensitivity":
        for lam, rho in SENS_CONFIGS:
            for size, seed, run, rep in itertools.product(
                    (30, 50), (11, 22, 33), (0, 1), range(reps)):
                tasks.append(make_task("sensitivity", f"L{lam}_R{rho}",
                                       "MEDIUM", size, seed, run, rep,
                                       METHODS, params, lam=lam, rho_h=rho))
    elif name == "groups":
        for cust, size, seed, run, rep in itertools.product(
                (1, 3, 5), (50, 100), seeds, (0, 1), range(reps)):
            tasks.append(make_task("groups", f"C{cust}", "MEDIUM", size, seed,
                                   run, rep, METHODS, params, customers=cust))
    elif name == "grid":
        # Twice as many epsilon points at half the budget per point: the same
        # total effort as the main campaign, spent on a finer grid.
        for size, seed, run in itertools.product(SIZES, seeds, (0, 1)):
            tasks.append(make_task("grid", "grid20", "MEDIUM", size, seed, run,
                                   0, ["FFD+SA-EDDTS", "ALNS"], params,
                                   grid_points=2 * GRID_POINTS,
                                   budget=BUDGET[size] / 2))
    elif name == "edge":
        # Follow-up screening for retained levels that sat at the edge of the
        # tested range with a non-negligible effect: one factor at a time,
        # around the levels retained by the L9 screening (from --params).
        configs = [
            ("ALNS", "ALNS_P5", {"ALNS_PENALTY_START": 5.0, "ALNS_PENALTY_END": 50.0}),
            ("ALNS", "ALNS_P10", {"ALNS_PENALTY_START": 10.0, "ALNS_PENALTY_END": 100.0}),
            ("ALNS", "ALNS_P20", {"ALNS_PENALTY_START": 20.0, "ALNS_PENALTY_END": 200.0}),
            ("NSGA-II", "NSGA_E2", {"NSGA2_ETA_C": 2, "NSGA2_CROSSOVER_PROB": 0.7}),
            ("NSGA-II", "NSGA_E5", {"NSGA2_ETA_C": 5, "NSGA2_CROSSOVER_PROB": 0.7}),
            ("NSGA-II", "NSGA_E10", {"NSGA2_ETA_C": 10, "NSGA2_CROSSOVER_PROB": 0.7}),
            ("NSGA-II", "NSGA_C05", {"NSGA2_ETA_C": 10, "NSGA2_CROSSOVER_PROB": 0.5}),
            ("NSGA-II", "NSGA_C06", {"NSGA2_ETA_C": 10, "NSGA2_CROSSOVER_PROB": 0.6}),
            # second round: eta_c = 2 was again the lowest level tested, and
            # p_c had only been varied at eta_c = 10
            ("NSGA-II", "NSGA_E1", {"NSGA2_ETA_C": 1, "NSGA2_CROSSOVER_PROB": 0.7}),
            ("NSGA-II", "NSGA_E05", {"NSGA2_ETA_C": 0.5, "NSGA2_CROSSOVER_PROB": 0.7}),
            ("NSGA-II", "NSGA_E2C05", {"NSGA2_ETA_C": 2, "NSGA2_CROSSOVER_PROB": 0.5}),
            ("NSGA-II", "NSGA_E2C09", {"NSGA2_ETA_C": 2, "NSGA2_CROSSOVER_PROB": 0.9}),
        ]
        for method, var, ov in configs:
            for preset, size, seed in TAGUCHI_INSTANCES:
                for run in (0, 1):
                    t = make_task("edge", var, preset, size, seed, run, 0,
                                  [method], params, overrides=ov)
                    t["edge"] = {"method": method, "config": var, "values": ov}
                    tasks.append(t)
    elif name == "ablation":
        variants = {
            "NO_SA": (["FFD+SA-EDDTS"], {"STAGE1_USE_SA": False}),
            "NO_COUPLING": (["FFD+SA-EDDTS"], {"EDDTS_PENALTY": 0.0}),
            "NO_RESTARTS": (["FFD+SA-EDDTS"], {"EDDTS_RESTARTS": False,
                                               "EDDTS_MAX_ITER_OVERRIDE": 10 ** 6}),
            "FIXED_NSGA2": (["FFD+SA-NSGA2"], {}),
        }
        abl_seeds = seeds if args.seeds else [11, 22, 33]
        for var, (meths, ov) in variants.items():
            for preset, size, seed in itertools.product(PRESETS, SIZES, abl_seeds):
                tasks.append(make_task("ablation", var, preset, size, seed, 0,
                                       0, meths, params, overrides=ov))
    else:
        raise SystemExit(f"unknown experiment {name}")
    return tasks


# ---------------------------------------------------------------------------
# Running one task
# ---------------------------------------------------------------------------

def load_core():
    sys.path.insert(0, str(HERE))
    with contextlib.redirect_stdout(io.StringIO()):
        import dyeing_core
    return dyeing_core


def stage1_counts(dc, jobs, capacity):
    groups = {}
    for j in jobs:
        groups.setdefault((j["color"], j["customer"]), []).append(j["size"])
    lb = sum(-(-sum(s) // capacity) for s in groups.values())
    return {"groups": len(groups), "lower_bound": int(lb)}


def fixed_batch_nsga2(dc, jobs, machines, setup_matrix, task):
    """Ablation: NSGA-II sequencing the Stage 1 batches of FFD+SA.

    Same batches as the proposed method, same total budget; only the second
    stage differs. The chromosome holds one random key per batch and is decoded
    exactly like the scheduling half of the benchmark NSGA-II.
    """
    import random
    t0 = time.perf_counter()
    initial = dc.simplified_ffd(jobs, machines, setup_matrix)
    batches, _ = dc.improved_simulated_annealing(initial, jobs, machines,
                                                 max_iterations=500)
    fixed = []
    for i, b in enumerate(batches, 1):
        fixed.append({"id": i, "family": b["family"], "customer": b["customer"],
                      "size": b["size"], "proc_time": b["proc_time"],
                      "due_date": b["due_date"],
                      "jobs": [j["id"] if isinstance(j, dict) else j for j in b["jobs"]]})

    class FixedNSGA2(dc.NSGA2_BatchScheduling):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.n_vars = len(fixed)
            self.mutation_prob = 1.0 / self.n_vars

        def decode(self, chrom):
            order = sorted(range(len(fixed)), key=lambda i: chrom[i])
            end = [0.0] * self.n_machines
            last = [None] * self.n_machines
            tard = setup_tot = 0.0
            for bi in order:
                b = fixed[bi]
                best_m, best_c = None, float("inf")
                for m in range(self.n_machines):
                    s = 0.0
                    if last[m] is not None and last[m] != b["family"]:
                        s = self.setup_times.get((last[m], b["family"]), 0)
                    c = end[m] + s + b["proc_time"]
                    if c < best_c:
                        best_c, best_m = c, m
                s = 0.0
                if last[best_m] is not None and last[best_m] != b["family"]:
                    s = self.setup_times.get((last[best_m], b["family"]), 0)
                end[best_m] += s + b["proc_time"]
                last[best_m] = b["family"]
                setup_tot += s
                for jid in b["jobs"]:
                    tard += max(0.0, end[best_m] - self.job_lookup[jid]["due_date"])
            return tard, setup_tot, None

    remaining = task["grid_points"] * task["budget"] - (time.perf_counter() - t0)
    g = dc.__dict__
    ga = FixedNSGA2(jobs, machines, setup_matrix, machines[0]["capacity"],
                    pop_size=g.get("NSGA2_POP_SIZE", 100), n_generations=10 ** 9,
                    crossover_prob=g.get("NSGA2_CROSSOVER_PROB", 0.9),
                    eta_c=g.get("NSGA2_ETA_C", 20), eta_m=g.get("NSGA2_ETA_M", 20),
                    time_limit=max(remaining, 1.0))
    results, _ = ga.solve()
    return [(float(r["fitness"][0]), float(r["fitness"][1])) for r in results]


def run_task(dc, task):
    import random
    import numpy as np
    for k, v in task["globals"].items():
        setattr(dc, k, v)
    dc.set_setup_parameters(task["lam"], task["rho_h"])
    gen = dc.RandomInstanceGenerator.from_preset(
        task["preset"], num_jobs=task["size"], num_machines=task["machines"],
        num_colors=task["colors"], num_customers=task["customers"],
        machine_capacity=task["capacity"], seed=task["gen_seed"])
    jobs, machines, meta, setup_matrix = gen.generate()
    dc.HEURISTIC_TIME_BUDGET = task["budget"]

    fronts, times, extra = {}, {}, {}
    extra["stage1"] = stage1_counts(dc, jobs, task["capacity"])
    for method in task["methods"]:
        if task["seed_mode"] == "run":
            s = stable_seed(task["gen_seed"], method, task["rep"])
            random.seed(s)
            np.random.seed(s)
            dc.RUN_SEED = task["rep"]
        dc.LAST_NSGA2_GENERATIONS = None
        dc.LAST_STAGE1_BATCHES = None
        t0 = time.perf_counter()
        if method == "FFD+SA-NSGA2":
            pts = fixed_batch_nsga2(dc, jobs, machines, setup_matrix, task)
        else:
            out = dc.generate_pareto_front_epsilon(
                jobs, machines, approach=method,
                num_points=task["grid_points"], setup_matrix=setup_matrix)
            front = out[0] if isinstance(out, tuple) else out
            pts = []
            for s in front or []:
                t = s.get("tardiness", s.get("Tardiness"))
                u = s.get("setup_time", s.get("Setup_Time"))
                if t is not None and u is not None:
                    pts.append((float(t), float(u)))
        times[method] = time.perf_counter() - t0
        fronts[method] = [list(p) for p in pts]
        if method == "NSGA-II":
            extra["nsga2_generations"] = dc.LAST_NSGA2_GENERATIONS
        if method == "FFD+SA-EDDTS" and dc.LAST_STAGE1_BATCHES is not None:
            extra["stage1"]["batches_used"] = len(dc.LAST_STAGE1_BATCHES)
    diag = getattr(gen, "horizon_diagnostics", None)
    return {"fronts": fronts, "times": times, "extra": extra,
            "diagnostics": diag}


# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------

def _windows_cores():
    """[(efficiency_class, [logical cpus])] per physical core, via the Win32 API."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    length = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
    buf = ctypes.create_string_buffer(length.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(length)):
        raise OSError(ctypes.get_last_error())
    raw, off, cores = buf.raw, 0, []
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    while off < length.value:
        rel = int.from_bytes(raw[off:off + 4], "little")
        size = int.from_bytes(raw[off + 4:off + 8], "little")
        if rel == 0:                                   # RelationProcessorCore
            eff = raw[off + 9]
            group_count = int.from_bytes(raw[off + 30:off + 32], "little")
            mask_off = off + 32
            cpus = []
            for gi in range(group_count):
                base = mask_off + gi * (ptr_size + 8)
                mask = int.from_bytes(raw[base:base + ptr_size], "little")
                group = int.from_bytes(raw[base + ptr_size:base + ptr_size + 2], "little")
                cpus += [group * 64 + b for b in range(ptr_size * 8) if mask >> b & 1]
            cores.append((eff, cpus))
        off += size
    return cores


def _linux_cores():
    root = Path("/sys/devices/system/cpu")
    seen, cores = set(), []
    for d in sorted(root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        sib = d / "topology" / "thread_siblings_list"
        if not sib.exists():
            continue
        txt = sib.read_text().strip()
        if txt in seen:
            continue
        seen.add(txt)
        cpus = []
        for part in txt.split(","):
            a, _, b = part.partition("-")
            cpus += list(range(int(a), int(b or a) + 1))
        cap = d / "cpu_capacity"
        eff = int(cap.read_text()) if cap.exists() else 0
        cores.append((eff, cpus))
    return cores


def detect_cores():
    try:
        cores = _windows_cores() if os.name == "nt" else _linux_cores()
    except Exception as exc:                                  # noqa: BLE001
        print(f"core detection failed ({exc}); pass --cores explicitly")
        return None
    return cores or None


def select_cores(spec):
    if spec in (None, "", "none"):
        return None
    if spec[0].isdigit():
        return [int(c) for c in spec.split(",")]
    cores = detect_cores()
    if not cores:
        raise SystemExit("could not detect cores; use e.g. --cores 0,2,4,6,8,10")
    classes = sorted({e for e, _ in cores})
    if spec == "p":
        chosen = [c for e, c in cores if e == classes[-1]]
    elif spec == "e":
        if len(classes) < 2:
            raise SystemExit("no efficiency cores detected")
        chosen = [c for e, c in cores if e == classes[0]]
    elif spec == "all":
        chosen = [c for _, c in cores]
    else:
        raise SystemExit(f"bad --cores value {spec}")
    return [c[0] for c in chosen]              # one logical CPU per physical core


def show_cores():
    cores = detect_cores()
    print(f"{platform.platform()}  |  {platform.processor()}")
    print(f"logical CPUs: {os.cpu_count()}")
    if not cores:
        return
    classes = sorted({e for e, _ in cores})
    for e, c in cores:
        kind = "P" if e == classes[-1] and len(classes) > 1 else ("E" if len(classes) > 1 else "-")
        print(f"  core class {e} ({kind})  logical CPUs {c}")
    for spec in ("p", "e"):
        try:
            print(f"--cores {spec}  ->  {select_cores(spec)}")
        except SystemExit as exc:
            print(f"--cores {spec}  ->  {exc}")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def worker(core, queue, out_dir, keep_logs):
    if core is not None:
        try:
            import psutil
            psutil.Process().cpu_affinity([core])
        except Exception as exc:                              # noqa: BLE001
            print(f"[worker {core}] could not pin: {exc}", flush=True)
    dc = load_core()
    host = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "core": core}
    while True:
        task = queue.get()
        if task is None:
            break
        path = out_dir / f"{task['key']}.json"
        if path.exists():
            continue
        log = io.StringIO()
        started = time.time()
        try:
            with contextlib.redirect_stdout(log):
                res = run_task(dc, task)
            rec = dict(task, **res, host=host, started=started,
                       finished=time.time())
            _write_json(path, rec)
        except Exception:                                     # noqa: BLE001
            (out_dir / f"{task['key']}.err").write_text(
                traceback.format_exc() + "\n\n" + log.getvalue()[-20000:],
                encoding="utf-8")
        if keep_logs:
            (out_dir / "logs").mkdir(exist_ok=True)
            (out_dir / "logs" / f"{task['key']}.log").write_text(
                log.getvalue(), encoding="utf-8")


def run_experiment(name, args):
    tasks = build_experiment(name, args)
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (out_dir / f"{t['key']}.json").exists()]
    cores = select_cores(args.cores) if args.cores else None
    if cores is None and args.cores not in (None, "none"):
        cores = None
    n_workers = args.workers or (len(cores) if cores else max(1, (os.cpu_count() or 2) // 2))
    if cores:
        try:
            import psutil  # noqa: F401
        except ImportError:
            raise SystemExit("psutil is needed for core pinning: python -m pip install psutil")
    if cores and n_workers > len(cores):
        raise SystemExit(f"{n_workers} workers but only {len(cores)} cores selected")
    total = sum(est_seconds(t) for t in todo)
    print(f"{name}: {len(tasks)} tasks, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to run")
    print(f"workers: {n_workers}   cores: {cores[:n_workers] if cores else 'not pinned'}")
    print(f"estimated time: {total / 3600:.1f} core-hours, about "
          f"{total / 3600 / n_workers:.1f} h wall-clock")
    if args.dry_run or not todo:
        return
    (out_dir / "_settings.json").write_text(json.dumps(
        {"experiment": name, "argv": sys.argv, "params": load_params(
            args.params, legacy=(name == "verify")), "cores": cores},
        indent=1), encoding="utf-8")

    todo.sort(key=est_seconds, reverse=True)     # longest first balances load
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for t in todo:
        queue.put(t)
    for _ in range(n_workers):
        queue.put(None)
    procs = []
    for i in range(n_workers):
        core = cores[i] if cores else None
        p = ctx.Process(target=worker, args=(core, queue, out_dir, args.logs))
        p.start()
        procs.append(p)

    t0, n0 = time.time(), len(tasks) - len(todo)
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(30)
            done = len(list(out_dir.glob("*.json"))) - 1 - n0
            errs = len(list(out_dir.glob("*.err")))
            el = time.time() - t0
            eta = el / done * (len(todo) - done) if done else float("nan")
            print(f"  {time.strftime('%H:%M')}  {done}/{len(todo)} done"
                  f"{f', {errs} errors' if errs else ''}, "
                  f"elapsed {el / 60:.0f} min, ~{eta / 60:.0f} min left", flush=True)
    except KeyboardInterrupt:
        print("interrupted; finished tasks are kept, rerun the same command to resume")
        for p in procs:
            p.terminate()
    # Unconsumed tasks left in the queue would otherwise keep the interpreter
    # waiting for its feeder thread at exit, and the prompt would not return.
    queue.cancel_join_thread()
    for p in procs:
        p.join(timeout=10)
    errs = list(out_dir.glob("*.err"))
    print(f"finished in {(time.time() - t0) / 60:.0f} min"
          f"{f'; {len(errs)} task(s) failed, see *.err' if errs else ''}")
    if name == "verify":
        verify_report(out_dir, args.archive)


# ---------------------------------------------------------------------------
# Verification against the archive
# ---------------------------------------------------------------------------

def verify_report(out_dir, archive):
    sys.path.insert(0, str(HERE))
    from analyse import hv_union
    arc = None
    if archive:
        p = Path(archive) / "data" / "raw_fronts.json"
        arc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if arc is None:
            print(f"archive not found at {p}")
    print("\nVERIFY (legacy settings, archived code paths)")
    for f in sorted(out_dir.glob("legacy__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        ikey = f"{r['preset']}_{r['size']}_{r['base_seed']}"
        print(f"\n{ikey} run {r['run']}   times: "
              + ", ".join(f"{m} {t:.0f}s" for m, t in r["times"].items()))
        if not arc or ikey not in arc:
            continue
        old = arc[ikey]["replications"][r["run"]]
        new = r["fronts"]
        lo, ln = old.get("LPT+EDD", [[None]])[0], new.get("LPT+EDD", [[None]])[0]
        same = lo[0] is not None and ln[0] is not None and all(
            abs(a - b) < 1e-6 for a, b in zip(lo, ln))
        print(f"  LPT+EDD archive {lo}  now {ln}  -> {'IDENTICAL' if same else 'DIFFERENT'}")
        # the archive stores the diagnostics of the second run of each key only
        if r["run"] == 1:
            od, nd = arc[ikey].get("diagnostics", {}), r.get("diagnostics") or {}
            for k in ("n_batches", "n_groups", "horizon_P"):
                if k in od:
                    print(f"  {k:10s} archive {od[k]}  now {nd.get(k)}")
        hv = hv_union({**{f'old {m}': v for m, v in old.items()},
                       **{f'new {m}': v for m, v in new.items()}})
        for m in METHODS:
            if f"old {m}" in hv:
                print(f"  HV (joint normalisation) {m:13s} archive "
                      f"{hv[f'old {m}']:.3f}  now {hv.get(f'new {m}', float('nan')):.3f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["cores", "smoke", "verify", "taguchi",
                                           "campaign", "sensitivity", "groups",
                                           "grid", "ablation", "edge", "nosa"])
    ap.add_argument("--params", help="JSON with tuned globals (from analyse.py taguchi)")
    ap.add_argument("--reps", type=int, default=1, help="replications per instance")
    ap.add_argument("--seeds", help="comma-separated base seeds (default 11,22,33,44,55)")
    ap.add_argument("--cores", default="p",
                    help="p (performance), e (efficiency), all, a list such as "
                         "0,2,4, or none (no pinning). Default p.")
    ap.add_argument("--workers", type=int, help="default: one per selected core")
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    ap.add_argument("--archive", default=str(HERE.parent / "v1.1.0"),
                    help="folder holding data/raw_fronts.json of the Zenodo archive")
    ap.add_argument("--instances", default="LOOSE_30_11_0,MEDIUM_50_22_1,TIGHT_80_33_0",
                    help="verify: PRESET_SIZE_SEED_RUN list")
    ap.add_argument("--logs", action="store_true", help="keep the console log of every task")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.experiment == "cores":
        show_cores()
        return
    if args.cores == "p" and not detect_cores():
        print("no core information; running unpinned")
        args.cores = None
    run_experiment(args.experiment, args)


if __name__ == "__main__":
    main()
