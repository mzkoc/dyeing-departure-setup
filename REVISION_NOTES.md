# Development notes of the revision code (packages v6 to v9b)

Kept as written during the revision, for the record. In these notes and in
code comments, "[7]" is the companion study (Koç, Sel and Kökçam,
*Mathematics*, 2026), its reference number in the submitted manuscript; it
is [8] in the revised one. The current documentation is `README.md`.

v1.2.0 (16 Nov 2026) adds one run made after v1.1.0 was archived: the
companion method at the budget of its own campaign at 30 and 60 jobs
(`run_own_budget.sh`, `own_budget.py`, `results_comp7own`), asked for in
the last internal review before resubmission. No other file changed,
except `README.md`, `CITATION.cff` and `LICENSE` (copyright holders).

---

# Revision experiments (JESTCH-D-26-02303, round 1) — package v8.1

These files extend the replication package of the paper. They change none of
the existing modules; copy them into the repository root, next to `core.py`,
and run them from there (WSL; the conda environment with Python 3.12 and
`requirements.txt`).

## What the experiments address

In the submitted study NSGA-II and ALNS worked on fixed FFD/BFD batches,
whereas CPM redistributed jobs between batches (L1) and assigned batches to
positions exactly (L4). The step **R** gives the benchmarks the same: on a
schedule's own timeline, each (family, customer) group redistributes its jobs
among its batches against their completion times and every family's batches
are assigned to its positions by the Hungarian method; Z2 cannot change.

| Method | File | What it is |
|---|---|---|
| CPM, CPM-blind, ALNS, NSGA-II | (existing) | as submitted |
| ALNS+R | `alns_r.py` | ALNS with R on every archive entrant |
| NSGA-II+R | `nsga2_r.py` | NSGA-II with R on an external archive and periodic batch refresh |
| CPM+R | `cpm_r.py` | ablation: R applied to CPM's front |
| CPM+ALNS+R[:mode][@share] | `hybrid.py` | the hybrid (below) |
| FFD+SA-EDDTS, FFD-EDDTS | `comp7.py` | the method of the companion study [7], its code unchanged (`comp7/`) |
| ALNS+R#k, NSGA-II+R#k | `params.py` | calibration rows (L9(3^4) plus row 0 = default) |

**Hybrid.** Within one clock: (1) the Corollary 3 schedule B, computed first
and without interruption, its time deducted from the clock, so every run
contains a Z2-optimal schedule by construction; (2) CPM on a share of the clock; (3)
CPM-A: every CPM point and B re-solved with batches redistributed against
their own schedule, while Z1 improves (the layer diagnosis showed that CPM's
batches, built for one timeline and reused for every support, are where it
loses tardiness); (4) ALNS+R on the rest. Designs: `union` (ALNS+R unseeded,
random stream seed + 500, fronts merged), `anchor` (only B enters ALNS+R's
archive), `seed` (all CPM-A points enter the archive). Time spent in (1)-(3)
is taken from the clock.

## Files

| File | Role |
|---|---|
| `ANALYSIS_PLAN.md` | instance sets, calibration rules, indicators, tests and decision rules, fixed before the runs |
| `regress.py` (+ `golden_v5.json`) | regression and guarantee checks; must print ALL PASS |
| `calibrate.py` | clock factor for this machine and number of workers |
| `fair_compare.py` | runs any set of methods on any replications, in parallel, resumable |
| `tune.py` | applies the calibration rule and writes `frozen.json` |
| `analyze_fair.py` | indicators, Friedman, Wilcoxon with Holm, paired effect sizes; several folders at once |
| `confirm.py` | claims C1-C7 and the secondary analyses of `ANALYSIS_PLAN.md`, rules applied as written |
| `jsonl_io.py` | safe reading of results files (cut-off lines, interrupted runs) |
| `run_comp7.sh` | experiment E1: the method of [7] on the validation instances |
| `run_v8.sh` | E1c ([7] with R), the ablations on the validation set, budget robustness |
| `regime.py` | results by regime e = \|F\| - m, and C4 by regime |
| `run_overnight.sh` | calibration -> freeze -> validation in one unattended run |
| `diagnose_layers.py` | the layer diagnosis (CPM's L2b-L4 on ALNS+R's supports) |

## Order of work

1. `python regress.py` — every line PASS.
2. Commit `ANALYSIS_PLAN.md` and the code before step 3 (dated evidence that
   the rules were fixed before the results were seen).
3. `bash run_overnight.sh` (defaults: 12 workers, factor 0.47; about 6-7 h):
   - calibration of ALNS+R and NSGA-II+R on replications 2-3, choice by the
     rule -> `frozen.json`;
   - calibration of the hybrid with the chosen ALNS+R parameters -> the
     hybrid's name added to `frozen.json`;
   - validation on replications 4-9 with the frozen configuration
     (`results_valid`);
   - verification scale, replications 0-3, for the CP-SAT gap
     (`results_cpsatgap`).
   An interrupted run resumes when the script is started again.
4. Analysis of the main table (the six main methods only) and the claims:

```
H=$(python -c "import json; print(json.load(open('frozen.json'))['hybrid'])")
python analyze_fair.py results_valid --pairs first --methods "$H,ALNS+R,NSGA-II+R,CPM,ALNS,NSGA-II" --tag main > analysis_valid.txt
python confirm.py results_valid --cpsatgap results_cpsatgap > confirm_valid.txt
```

4b. The method of [7] (once: `pip install pyomo==6.9.5`): `bash run_comp7.sh`
   (about 1 h) -> `analysis_comp7.txt`.

5. Seed variability on the development set, with the frozen configuration:

```
python fair_compare.py --scales verify small medium large xlarge --reps 0,1 --seeds 10 --workers 12 --factor 0.47 --params frozen.json --out results_dev --methods "$H,ALNS+R,NSGA-II+R,ALNS,NSGA-II,CPM,CPM+R,CPM-blind"
```

## Clock

Per instance: CPM anchor x factor x `--clock-mult`. Anchors are the mean CPU
times of CPM in the submitted main study (7, 136, 302, 281 s at 30, 60, 100,
150 jobs; inherited, hence 150 < 100); below 30 jobs 7 s x n/30, above 150
jobs 281 s x n/150. `--clock-mult` (default 1) is for budget-robustness runs.

## Records

`<out>/results.jsonl`, one line per run: instance and cell, method, seed,
clock, wall and CPU seconds, evaluations and evaluations/s, the parameters
used, the verified front as [Z1, Z2] pairs, the departures per family behind
every point (any departure-dependent water model can be evaluated
afterwards; their sigma-weighted sum is checked against Z2), and for the
hybrid the origin of every point and its parts (ALNS+R front, CPM-A front,
B).

## Fairness and hardware

Every method receives the same clock on every instance. Under WSL2, taskset
pins virtual CPUs, not physical performance or efficiency cores, so core
assignment is not controlled; tasks of one instance run concurrently on
different workers, which adds noise but no systematic bias. CPU seconds and
evaluations/s are recorded so that this can be checked.

## Changes in v7

No method, parameter, clock or instance changed; `alns_r.py`, `nsga2_r.py`,
`hybrid.py`, `params.py`, `cpm.py` and the other method files are those of
v6, so a v6 run can be resumed with v7.

- Interrupted runs (computer asleep or hibernating, battery or power cut): on start `fair_compare.py`
  sets aside a cut-off last line and any run whose wall time exceeds its
  clock by more than 25% plus 2 s, and runs them again; records are forced
  to disk as written. `python jsonl_io.py <folder>/results.jsonl` reports a
  file's state. `run_overnight.sh` calls each step twice so that runs set
  aside in the first pass are redone.
- `analyze_fair.py --pairs first`: H against each other listed method, Holm
  over exactly those comparisons (ANALYSIS_PLAN.md, section 4).
- `confirm.py`: C1-C7 and the secondary analyses.
- `comp7.py`, `comp7/`, `run_comp7.sh`: experiment E1.
- `regress.py`: checks for the method of [7] (skipped if Pyomo is missing)
  and for the handling of interrupted runs.

## Changes in v8

- `FFD+SA-EDDTS+R` (comp7.py): [7] on 85% of the clock, then R on its front,
  as CPM+R; [7]'s front before R is logged.
- Analyses snap Z2 to the 1/12 h grid before the dominance filter
  (`--z2-grid`, default 12; 0 for perturbed or pair-dependent setups); point
  origin as CPM part only, ALNS+R part only, or both.
- `analyze_fair.py --clock-mult` selects runs of one clock multiplier.
- `regime.py`: results by regime and C4 by regime; C3 reported descriptively
  (ANALYSIS_PLAN.md, amendment 2).
- `run_v8.sh`: tonight's runs (about 3.6 h).

## Changes in v8.1

- CPM+R and FFD+SA-EDDTS+R apply R with ALNS+R's frozen number of rounds
  (`r_rounds`, 5) through one shared function (`alns_r.r_on_front`), least
  tardiness first; whether R's budget ran out is logged (`r_info` / `comp7`).
  The development-set CPM+R runs of v6-v8 used 3 rounds.
- `fair_compare.py --redo M1,M2`: finished runs of these methods are set
  aside and run again.
- The hybrid logs the time of B, CPM, CPM-A and ALNS+R (`parts.seconds`).

## Changes in v9a

- `variants.py`: sensitivity variants (`perturb`, `sigx2`, `sigx4`, `skew`,
  `cluster`, `c5`, `q300`, `size`) and the scalability scales `x200` and
  `x300`; `instances.generate_instance` gained keyword options whose defaults
  give the main study's instances bit for bit (checked by hash in
  `regress.py`).
- `fair_compare.py --variant V --min-excess K`; records of `perturb` carry
  every front schedule evaluated on five perturbed instances (`perturbed`).
- `cpm_params.py`: CPM's hand-set parameters one at a time inside the hybrid
  (method names `<hybrid>~<knob>=<value>`).
- `check_comp7.py`: reproduction check of the adapter for [7] against its
  archive (`comp7/repro_reference.json`); `comp7.py` refactored so that the
  check runs the adapter's own code path (`comp7._run`), behaviour unchanged.
- `paper_tables.py budget`: supplementary Table S5.
- `run_v9a.sh`: tonight's runs, about 4.7 h on 12 workers.

## Changes in v9b

- `e2.py`, `e2_analyze.py`, `run_v9b.sh`: experiment E2 (about 2.2 h).
- `core.Instance.setup_matrix` (optional, E2 only) and `core.setup_time`
  reading it; `benchmarks.py` and `comp7.to_comp7` read setups through
  `core.setup_time`. Without a matrix every result is unchanged (golden
  checks).
- `hybrid.cpm_alns_r(search_inst=...)`: the ALNS+R part of the union design
  on another instance (H-mix).
