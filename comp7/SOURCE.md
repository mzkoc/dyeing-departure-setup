# Code of the companion study, unchanged

Files copied without modification from the replication archive of the
companion study (Koç, Sel and Kökçam, Mathematics), version 1.3.0
(doi:10.5281/zenodo.23904163), folder `code/`:

| File | SHA-256 |
|---|---|
| `dyeing_core.py` | `3ddce7ad2a0d71090365370cf3d9d268deb851caba04bb554f28d95217ed7aa6` |
| `rerun.py` | `13f16a599d5f1079335cac8b95ad81fc38dd5eccd9943c9e78c1f3c2dc5b3608` |
| `tuned_params.json` | `d90efbcb8f5d474caad35e465266f0758de019d94ef5754d01a6d8fc3d9bd301` |

The experiments of this repository were run with the authors' copy of that
archive before its publication (labelled 1.2.1 in the analysis plan; never
published). In it `dyeing_core.py` and `tuned_params.json` are identical to
version 1.3.0, and `rerun.py` differs only by the experiment `nosa` that
version 1.3.0 adds (the companion campaign without the annealing step of
Stage 1); `load_params`, `make_task`, `run_task` and `stable_seed`, the
functions used here, are identical. The file above is that of version 1.3.0.

`LICENSE` is the archive's licence (MIT for code). `comp7.py` in the
repository root uses `rerun.load_params("tuned_params.json")`, i.e. the
parameters and correction switches of the revised companion study, and calls
`dyeing_core.generate_pareto_front_epsilon(..., approach="FFD+SA-EDDTS")`
with the instance translated as described in `comp7.py`. `dyeing_core.py`
imports Pyomo at load time (it is not used by FFD+SA-EDDTS): `pip install
pyomo==6.9.5`.

`repro_reference.json` is not a file of the archive: it holds the least
tardiness, least setup and front size of FFD+SA-EDDTS in the five stored
runs of three 30-job tasks of the companion campaign (`results/campaign/`, the
SHA-256 of each source file recorded with it) and the parameters recorded by
that campaign, for `check_comp7.py`.
