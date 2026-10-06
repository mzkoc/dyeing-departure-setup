# Departure-dependent setups in parallel batch scheduling

Code, instance generator and raw results for

> M.Z. Koç, Ç. Sel, M. Dolmacı. *Exploiting departure-dependent setup
> structure in parallel batch scheduling: a bi-objective matheuristic for
> water-efficient dyeing operations.* Manuscript submitted for publication,
> 2026.

Every table and figure of the results section can be rebuilt from the files
in `results/` in a few minutes; the experiments themselves can be rerun from
scratch with the commands below.

## The problem in one paragraph

Jobs are grouped into batches that share a colour family and a customer and
fit the vessel capacity; batches are processed on identical parallel
machines; a change of family on a machine costs a setup that depends only on
the family being left. The two objectives are total tardiness (Z1) and total
setup time (Z2), the latter converted to cleaning water. Under block
contiguity Z2 has a closed form in the family-to-machine assignment
(Proposition 1), with an attained lower bound (Corollary 3), and the proposed
matheuristic (CPM) builds the front by walking the resulting ladder of setup
levels.

## Contents

| File | Role |
|---|---|
| `core.py` | Data model and the single evaluation routine used by every method; Z2 closed form and bound |
| `instances.py` | Instance generator (630 instances, deterministic seeds) |
| `layer1.py` | L1: batching and due-date-aware redistribution |
| `layer234.py` | L2a support, L2b batch counts, L3 block order, L4 Hungarian assignment |
| `cpm.py` | The proposed matheuristic: ladder sweep and outer feedback loop |
| `benchmarks.py` | NSGA-II, ALNS and the LPT+EDD dispatching baseline |
| `bounds.py` | Counting lower bound on total tardiness |
| `cpsat.py`, `verify_run.py`, `run_cpsat.py` | CP-SAT reference on the verification scale |
| `verify_p1.py` | Brute-force check of Proposition 1 and Corollary 3 |
| `runner.py` | Main experiment (checkpointed, resumable) |
| `walltime.py` | Equal wall-clock check |
| `analysis.py` | Indicators, statistical tests and summary tables from the raw outputs |
| `make_tables.py` | LaTeX bodies of Tables 6–10 from the output of `analysis.py` |
| `figures.py` | Figures 1–4 |
| `results/` | Raw outputs of the runs reported in the paper |

## Setup

Python 3.12. The versions used for the reported runs are pinned in
`requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Rebuilding the tables and figures from the stored results

```bash
python analysis.py --results results/part_a results/part_b results/part_c results/part_d \
                   --cpsat results/cpsat_verification.csv \
                   --walltime results/walltime/walltime.csv --out results/tables
python make_tables.py --tables results/tables --out results/tables/latex
python figures.py --out figures
```

| Paper | Produced by | File |
|---|---|---|
| Table 6 | `analysis.py`, `make_tables.py` | `results/tables/table6_main.csv`, `latex/table6_main.tex` |
| Table 7 | `analysis.py`, `make_tables.py` | `friedman_igd_plus.csv`, `nemenyi_igd_plus.csv`, `latex/table7_friedman.tex` |
| Table 8 | `analysis.py`, `make_tables.py` | `table8_coverage.csv`, `latex/table8_coverage.tex` |
| Table 9 | `analysis.py`, `make_tables.py` | `table9_ablation.csv`, `latex/table9_ablation.tex` |
| Table 10 | `analysis.py`, `make_tables.py` | `walltime_summary.csv`, `walltime_friedman.csv`, `latex/table10_walltime.tex` |
| Section 7.4 (regimes) | `analysis.py` | `by_regime.csv`, `report.txt` |
| Section 7.7 (CP-SAT) | `analysis.py` | `report.txt` |
| Figures 1–4 | `figures.py` | `figures/fig1_gantt.pdf`, `fig2_p1.pdf`, `fig34_scaling.pdf`, `fig5_evidence.pdf` |

`results/tables/report.txt` collects every number quoted in the text of
Section 7 that is not in a table.

## Rerunning the experiments

**1. Proposition 1 (seconds).**

```bash
python verify_p1.py
```

**2. Main study.** The scales can run in parallel, each into its own
directory. Runs are checkpointed; repeating a command resumes it.

```bash
python runner.py --scales verify small --out results/part_a
python runner.py --scales medium       --out results/part_b
python runner.py --scales large        --out results/part_c
python runner.py --scales xlarge       --out results/part_d
```

Effort is a count of evaluations (10 000 per method and instance), so the
fronts reproduce exactly on any machine; CPU times depend on the machine and
on concurrent load. On an Intel Core i9-13900H with the four scales running
in parallel, the four parts took about 1, 6, 15 and 16 hours.

**3. CP-SAT reference (verification scale, replications 0–4).**

```bash
python run_cpsat.py
```

**4. Equal wall-clock check.** The clock per scale is the mean CPU time of
the proposed method in the main study.

```bash
python walltime.py --main-results results/part_a/results.csv results/part_b/results.csv \
                   results/part_c/results.csv results/part_d/results.csv --out results/walltime
```

Runs under a wall clock do not reproduce exactly; that is inherent to the
question they answer.

## Output format

`results.csv` has one row per instance and method: the extreme points of the
front (`z1_min`, `z2_min`), front size, hypervolume, IGD, the Z2 bound and
whether it was attained, cleaning water, CPU time, evaluations used and
feasibility. `fronts.jsonl` holds every returned front as `[Z1, Z2]` pairs.
Every schedule behind every point was re-evaluated through `core.py` before
it was written. `cpsat_verification.csv` and `walltime/walltime.csv` hold
the two checks.

## Citation

See `CITATION.cff`; the archived release will carry a Zenodo DOI.

## Licence

MIT, see `LICENSE`.
