# Departure-dependent setups in parallel batch scheduling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23182944.svg)](https://doi.org/10.5281/zenodo.23182944)

Code, instance generator, analysis plan and raw results for

> M.Z. Koç, Ç. Sel. *Departure-dependent setups in parallel batch
> scheduling: a closed form, a certified setup bound and structure-derived
> components for water-efficient dyeing.* Engineering Science and Technology,
> an International Journal, revised manuscript JESTCH-D-26-02303, 2026.

| Release | Content | DOI |
|---|---|---|
| v1.1.0 | the revised manuscript: all experiments of the revision, with the code and results of v1.0.0 | [10.5281/zenodo.24230376](https://doi.org/10.5281/zenodo.24230376) |
| v1.0.0 | the submitted manuscript | [10.5281/zenodo.23182945](https://doi.org/10.5281/zenodo.23182945) |

The DOI in the badge, 10.5281/zenodo.23182944, stands for all versions and
resolves to the latest.

## The problem and the method in brief

Jobs are grouped into batches that share a colour family and a customer and
fit the vessel capacity; batches are processed on identical parallel
machines; a change of family on a machine costs a setup that depends only on
the family being left (assumption A3). The objectives are total tardiness
(Z1) and total setup time (Z2), the latter read as cleaning water. Under
block contiguity Z2 has a closed form in the family-to-machine assignment,
with a lower bound valid for every schedule and attained by a schedule B
built in one evaluation. The revised paper derives three components from
this structure (B; a redistribution-and-assignment step R that leaves Z2
unchanged; the ladder of attainable setup levels walked by CPM) and combines
them with an adaptive large neighbourhood search in a hybrid, H.

## Names in the result files and in the paper

| In the files | In the paper |
|---|---|
| `CPM+ALNS+R:union@0.1` | H, the hybrid (frozen configuration) |
| `CPM`, `CPM+R`, `CPM-blind` | CPM (the submitted method) and its two ablations |
| `ALNS+R`, `NSGA-II+R` | ALNS and NSGA-II with the step R, tuned (`frozen.json`) |
| `ALNS`, `NSGA-II` | the benchmarks of the submitted study, as submitted |
| `ALNS+R#k`, `NSGA-II+R#k` | calibration candidates (row 0 = untuned, rows 1-9 = L9 design) |
| `CPM+ALNS+R:<design>@<share>` | calibration candidates of the hybrid |
| `<H>~<parameter>=<value>` | CPM's hand-set parameters moved inside H (Table S4) |
| `FFD+SA-EDDTS`, `FFD-EDDTS` | SA-EDDTS and FFD+EDDTS, the companion study's method with and without annealing |
| `FFD+SA-EDDTS+R` | SA-EDDTS+R |
| `H-max`, `H-mean`, `H-mix`, `[7]+R` (E2) | H-max, H-mean, H-mix and SA-EDDTS+R of Section 7.4 |

In code comments, `ANALYSIS_PLAN.md` and `REVISION_NOTES.md`, "[7]" is the
companion study (Koç, Sel and Kökçam, *Mathematics*, 2026), its reference
number in the submitted manuscript; it is [8] in the revised one. Its code is
kept unchanged in `comp7/` (version 1.3.0 of its archive,
doi:10.5281/zenodo.23904163); `comp7/SOURCE.md` gives the file hashes and
how the copy used for the runs relates to that version.

## Files

| File | Role |
|---|---|
| `core.py` | data model and the single evaluation routine used by every method; Z2 closed form and bound; optional pair-dependent setup matrix (E2) |
| `instances.py` | instance generator (630 instances, deterministic seeds); keyword options for the sensitivity variants |
| `variants.py` | sensitivity variants and the 200- and 300-job scales |
| `layer1.py`, `layer234.py`, `cpm.py` | CPM: batching and redistribution (L1), support, batch counts, block order and Hungarian assignment (L2-L4), ladder sweep |
| `alns_r.py`, `nsga2_r.py`, `cpm_r.py` | ALNS+R, NSGA-II+R, CPM+R |
| `hybrid.py` | the hybrid: B, CPM, CPM-A, ALNS+R in one clock; designs `union`, `anchor`, `seed` |
| `benchmarks.py` | NSGA-II, ALNS and the LPT+EDD dispatching baseline |
| `comp7.py`, `comp7/` | adapter for the companion study's method and its unchanged code |
| `check_comp7.py` | reproduction check of that adapter against the companion archive |
| `e2.py` | experiment E2 (setups that depend on both families) |
| `params.py`, `cpm_params.py` | calibration designs; CPM's parameter levels |
| `fair_compare.py` | runs any set of methods on any replications under the equal clock, in parallel, resumable |
| `calibrate.py` | clock factor for a machine and a number of workers |
| `tune.py`, `tune_share.py` | the calibration rule; writes `frozen.json` |
| `analyze_fair.py`, `confirm.py`, `regime.py`, `e2_analyze.py` | indicators, tests and the decision rules of the analysis plan |
| `paper_tables.py`, `paper_figures.py` | the tables and Figure 3 of the revised paper |
| `diagnose_layers.py` | the layer diagnosis (Table 10) |
| `jsonl_io.py` | safe reading of result files (cut-off lines, interrupted runs) |
| `regress.py`, `golden_v5.json` | regression and guarantee checks; must print `ALL PASS` |
| `verify_p1.py` | brute-force check of Proposition 1 and Corollary 3 |
| `cpsat.py`, `verify_run.py`, `run_cpsat.py` | CP-SAT reference on the verification scale |
| `run_overnight.sh`, `run_comp7.sh`, `run_v8.sh`, `run_v9a.sh`, `run_v9b.sh` | the revision runs, in the order they were made |
| `ANALYSIS_PLAN.md` | instance sets, clock, calibration rule, indicators, tests and decision rules, fixed before the validation run, with five dated amendments |
| `frozen.json` | the calibrated configuration |
| `REVISION_NOTES.md` | development notes of the revision packages |
| `runner.py`, `walltime.py`, `analysis.py`, `make_tables.py`, `figures.py`, `pilot.py`, `analyze_pilot.py` | the submitted study (v1.0.0) and the first equal-clock pilot |

`core.py`, `instances.py` and `benchmarks.py` gained options in the revision
that leave every result of v1.0.0 unchanged (checked by `regress.py`); the
other files of v1.0.0 are unchanged.

## Result folders

Each folder holds `results.jsonl`, one record per run. Files ending in
`.suspended` hold runs that were interrupted (computer asleep, power cut;
wall time above 1.25 times the clock plus 2 s) and were run again
(`ANALYSIS_PLAN.md`, amendment 1); `.bad` files hold cut-off lines; a `.redo`
file holds runs replaced by a rerun after a correction (`REVISION_NOTES.md`,
v8.1). Only `results.jsonl` enters the analyses.

| Folder | Runs | Replications | Records | Paper |
|---|---|---|---|---|
| `results_calib` | ALNS+R#0-9, NSGA-II+R#0-9, five hybrid designs | 2-3 | 3150 | Section 6.4; Tables S2-S3 |
| `results_valid` | H, CPM, ALNS+R, NSGA-II+R, ALNS, NSGA-II | 4-9 | 2268 | Sections 7.1-7.2, 7.8; Tables 7, 11; Figure 3 |
| `results_cpsatgap` | H, ALNS+R, CPM at the verification scale | 0-3 | 324 | Section 7.5 |
| `results_valid_extra` | SA-EDDTS+R, CPM+R, CPM-blind | 4-9 | 1134 | Sections 7.1, 7.3; Table 8 |
| `results_comp7` | SA-EDDTS and FFD+EDDTS | 4-9 | 756 | Section 7.3; Tables 8, S9 |
| `results_comp7x2` | SA-EDDTS at twice the clock, 100 and 150 jobs | 4-9 | 108 | Section 7.3 |
| `results_dev` | eight methods, ten seeds | 0-1 | 6678 | Section 7.7 ("Seeds") |
| `results_budget` | H, ALNS+R, NSGA-II+R, CPM at 0.25, 0.5 and 2 times the clock | 4-6 | 2268 | Section 7.7; Table S5 |
| `results_sens` | sensitivity variants and perturbed data, cells with e > 0 at 60 and 150 jobs | 4-9 | 1152 | Section 7.7; Table S6 |
| `results_scale` | 200 and 300 jobs | 4-9 | 144 | Section 7.7; Table S7 |
| `results_cpmparams` | CPM's parameters inside H, 60 and 100 jobs | 2-3 | 540 | Table S4 |
| `results_comp7repro` | the companion method by its own runner and by the adapter | companion tasks | 12 | Section 6.2; Table S10 |
| `results_e2` | E2, 60 and 150 jobs, lambda 1 and 2 | 4-9 | 972 | Section 7.4; Tables 9, S8 |
| `diag` | layer diagnosis: CPM's layers on ALNS+R's supports, 60 and 100 jobs | 0-1 | 36 | Section 7.6; Table 10 |
| `results_fair` | development: the first equal-clock comparisons (earlier code) | 0-1 | 648 | Section 6.1 |
| `results_tune` | development: the first designs and shares of the hybrid, among them the seeded design (earlier code) | 2-3 | 576 | Section 6.1 |
| `results/` | the submitted study (v1.0.0) | 0-9 | | submitted manuscript |

`results_fair` and `results_tune` hold development runs, made with earlier
versions of the code while the revision was designed; no table of the
revised paper is computed from them. Like `diag`, they use only
replications 0-3: no run of the revised methods touched the validation
replications 4-9 before the frozen configuration was run on them.

A record holds the instance and its cell, the method, seed, clock, wall and
CPU seconds, evaluations and evaluations per second, the parameters used, the
front as [Z1, Z2] pairs, every one re-evaluated through `core.py`, the
departures per family behind every point (their sigma-weighted sum is checked
against Z2), and, for the hybrid, the origin of every point and its parts
(B, the CPM-A front, the ALNS+R front) with their times.

## Setup

Python 3.12 (the runs used a conda environment under WSL2, Ubuntu, on an
Intel Core i9-13900H). The versions used are pinned in `requirements.txt`;
Pyomo is needed only because the companion code imports it at load time.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python regress.py                 # every line PASS, last line ALL PASS
python verify_p1.py               # Proposition 1 and Corollary 3 by brute force
```

## Rebuilding the tables and figures of the revised paper

Run in the repository root; each command prints the body of the table.

| Table or figure | Command |
|---|---|
| Table 7 | `python paper_tables.py main results_valid` |
| Table 8 | `python paper_tables.py comp7 results_valid results_comp7 results_valid_extra` |
| Table 9 | `python paper_tables.py e2 results_e2` |
| Table 10 | `python diagnose_layers.py --summary diag` |
| Table 11 | `python paper_tables.py water results_valid` |
| Figures 1, 2 | `python figures.py --out figures` |
| Figure 3 | `python paper_figures.py budget results_valid --out fig3_budget.pdf` |
| Tables S2, S3 | `python paper_tables.py calib results_calib` |
| Table S4 | `python paper_tables.py cpm results_cpmparams` |
| Table S5 | `python paper_tables.py budget results_valid results_budget` |
| Table S6 | `python paper_tables.py sens results_sens` |
| Section S6, perturbed data | `python paper_tables.py perturb results_sens` |
| Table S7 | `python paper_tables.py scale results_scale` |
| Table S8 | `python paper_tables.py e2regime results_e2` |
| Table S9 | `python paper_tables.py eddts results_valid results_comp7` |
| Table S10 | `python paper_tables.py repro results_comp7repro` |

The numbers quoted in the text come from the same scripts and from:

```bash
H=$(python -c "import json; print(json.load(open('frozen.json'))['hybrid'])")
python confirm.py results_valid --cpsatgap results_cpsatgap      # decisions of the analysis plan
python analyze_fair.py results_valid --pairs first --methods "$H,ALNS+R,NSGA-II+R,CPM,ALNS,NSGA-II" --tag main
python regime.py results_valid results_comp7 --methods "$H,FFD+SA-EDDTS,ALNS+R,CPM" --ref "$H"
python e2_analyze.py results_e2 --by-regime
```

## Rerunning the experiments

Every method receives the same wall clock on every instance: the mean CPU
time CPM needed per instance in the submitted study at that scale (7, 136,
302 and 281 s at 30, 60, 100 and 150 jobs; 7 s x n/30 below 30 jobs and
281 s x n/150 above 150), times a machine factor measured by `calibrate.py`
(0.47 with 12 workers on the machine above). Runs under a wall clock do not
reproduce exactly. All scripts are resumable; they call each step twice so
that interrupted runs are redone.

```bash
python calibrate.py --scale medium --workers 12      # prints the factor F
W=12 F=0.47 bash run_overnight.sh     # calibration -> frozen.json -> validation -> CP-SAT gap (6-7 h)
W=12 F=0.47 bash run_comp7.sh         # the companion method (about 1 h)
W=12 F=0.47 bash run_v8.sh            # SA-EDDTS+R, ablations, budget robustness (about 3.6 h)
W=12 F=0.47 bash run_v9a.sh           # reproduction check, sensitivity, scalability, CPM parameters (about 4.7 h)
W=12 F=0.47 bash run_v9b.sh           # E2 (about 2.2 h)
```

The ten-seed development run and the companion method at twice the clock
(`ANALYSIS_PLAN.md`, sections 6 and 8):

```bash
python fair_compare.py --scales verify small medium large xlarge --reps 0,1 --seeds 10 --workers 12 --factor 0.47 \
    --params frozen.json --out results_dev --methods "$H,ALNS+R,NSGA-II+R,ALNS,NSGA-II,CPM,CPM+R,CPM-blind"
python fair_compare.py --scales large xlarge --reps 4,5,6,7,8,9 --workers 12 --factor 0.47 --clock-mult 2 \
    --out results_comp7x2 --methods FFD+SA-EDDTS
```

## The submitted study (v1.0.0)

`results/` holds the raw outputs of the submitted manuscript, unchanged.
Its tables are rebuilt with

```bash
python analysis.py --results results/part_a results/part_b results/part_c results/part_d \
                   --cpsat results/cpsat_verification.csv \
                   --walltime results/walltime/walltime.csv --out results/tables
python make_tables.py --tables results/tables --out results/tables/latex
```

and its runs repeated with `runner.py` (evaluation budgets, so they reproduce
exactly), `run_cpsat.py` and `walltime.py`, as described in the README of
v1.0.0.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The companion code in `comp7/` carries its own licence
(`comp7/LICENSE`).
