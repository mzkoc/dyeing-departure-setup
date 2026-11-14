# Analysis plan — JESTCH-D-26-02303, revision round 1

Written before the calibration (`results_calib`) and validation
(`results_valid`) runs of package v6, and committed before they start. No
method, parameter, clock or instance set is changed after the validation run
starts; if a defect is found afterwards, the fix and the rerun are reported as
such.

## 1. Instance sets

Generator `instances.py`, `build_matrix(include_verify=True)`: verification
scale (12, 16, 20 jobs; m in {1,2,3}, {2,3,4}, {2,3,4}), and 30, 60, 100, 150
jobs (three machine counts each); tightness 0.3, 0.5, 0.7.

| Set | Replications | Use |
|---|---|---|
| Development | 0, 1 | pilot studies and the v5 comparisons; seed-variability study (section 6); no confirmatory claim |
| Calibration | 2, 3 | choice of parameters and of the hybrid (section 3) |
| Validation | 4-9 | confirmatory comparison (sections 4-5); 54 instances per main scale, 162 at the verification scale; run once |
| CP-SAT gap | verification scale, 0-4 | the 135 instances of the submitted CP-SAT study; descriptive only (includes calibration replications 2-3, stated as such) |

## 2. Clock and hardware

- Clock per instance = CPM anchor x 0.47. Anchors: the mean CPU time of CPM
  in the submitted main study, 7, 136, 302, 281 s at 30, 60, 100, 150 jobs
  (inherited, hence 150 < 100). Below 30 jobs: 7 s x n/30. Above 150 jobs
  (scalability runs): 281 s x n/150.
- Factor 0.47 measured by `calibrate.py` with 12 concurrent workers (WSL2,
  Intel i9-13900H); CPM reproduced the stored Z1 on 12/12 instances.
- 12 concurrent workers; one task = one (instance, method, seed) run; tasks
  longest first. Wall seconds, CPU seconds and evaluations/s are logged per
  run.

## 3. Calibration rule (calibration set, all five scales)

- ALNS+R: L9(3^4) over dmax {0.25, 0.35, 0.50}, segment {10, 25, 50}, decay
  {0.6, 0.8, 0.9}, rounds of R {1, 3, 5}; plus row 0 = the settings used
  before calibration. NSGA-II+R: population {30, 60, 120}, mutation {0.1, 0.2,
  0.4}, batch refresh {5, 10, 20}, rounds of R {1, 3, 5}; plus row 0. The
  array is fractional; the choice is among the ten tested rows only.
- Rule: lowest mean IGD+ rank over the calibration instances (ranks per
  instance among the candidates; normalisation and IGD+ reference from the
  candidates' runs); ties (mean ranks within 0.05) broken by the higher mean
  normalised HV.
- Hybrid, with the chosen ALNS+R parameters: candidates union@0.02,
  union@0.05, union@0.1, union@0.2, anchor@0.05. Same rule, after keeping the
  candidates that attain the Z2 bound on every calibration instance (all do by
  construction; checked).
- Unchanged: ALNS and NSGA-II (the submitted benchmarks), CPM, CPM-blind,
  CPM+R. Seeds: replication (+1000 i for further seeds); the union design's
  ALNS+R stream uses seed + 500.

## 4. Validation: methods and indicators

- Main set: the frozen hybrid H, ALNS+R, NSGA-II+R, CPM, ALNS, NSGA-II.
  Per instance, objectives are normalised by the ideal and nadir of the union
  of these six fronts; IGD+ uses the non-dominated set of that union; HV the
  point (1.1, 1.1).
- Indicators: IGD+, HV, least tardiness Z1min, attainment of the Z2 bound.
- Tests, per scale: paired Wilcoxon signed-rank on per-instance values; Holm
  adjustment within each scale and indicator over the five comparisons
  H vs ALNS+R, NSGA-II+R, CPM, ALNS, NSGA-II.
- Effect sizes: paired probability of superiority PS = (wins + ties/2)/n;
  median paired percentage difference for Z1min.
- Ablations are computed separately and do not enter the main normalisation:
  H against H_A+B (its own ALNS+R part plus the Corollary 3 schedule B, both
  logged) and against ALNS+R+B (the ALNS+R benchmark run plus B), normalised
  by the union of the fronts compared.

## 5. Claims and decision rules (alpha = 0.05 after Holm)

- C1, front quality: "H gives a better front than X at scale s" if IGD+ is
  significantly better and PS >= 0.60. Claimed per scale; "from k jobs
  upward" only if it holds at every larger scale.
- C2, least tardiness against ALNS+R: "H matches ALNS+R" at scale s if H is
  not significantly worse on Z1min and the median paired difference is at
  most +1.0%. Otherwise the loss is reported.
- C3, the ladder beyond one schedule ("level traversal"): kept if H is better
  than H_A+B on HV (Holm over the five scales, PS >= 0.60) at two or more of
  the 60, 100 and 150-job scales. Otherwise the claim is removed and the CPM
  part is described as supplying the certified setup end only.
- C4, against the practical rival ALNS+R+B: reported and tested; "H is at
  least as good" if it is not significantly worse on HV or IGD+.
- C5, setup end: H attains the Z2 bound on every instance (by construction,
  verified per run); the attainment rates of the other methods are reported.
- C6, verification scale against CP-SAT (CP-SAT gap set): median and maximum
  gap of H's Z1min to the proven optimum where CP-SAT proved it, otherwise to
  CP-SAT's best value; compared descriptively with CPM's gap in the submitted
  paper (median 11.5%). The CP-SAT model limits each group to its FFD batch
  count.
- C7, single-machine instances: front size and Z1min gap of H against CPM
  (which returns one point on every such instance).

## 6. Secondary analyses (descriptive)

- Setup-budget curves: for L_k = Z2* + k x 0.25 h, k = 0..8, each method's
  least Z1 with Z2 <= L_k, relative to the best method at that budget, and the
  share of instances with no point within L_k.
- Origin of H's front points (CPM part or ALNS+R part), by scale.
- Speed and quality: within each method, rank correlation between
  evaluations/s and IGD+ across seeds of the same instance (development set,
  10 seeds).
- Water: from the logged departures, W = sum_f w_f d_f with
  w_f = w0 + kappa sigma_f; liquor ratios 1:6, 1:8, 1:10, 1:15 and
  w0 in {0, V/2, V}.
- Seed variability (development set, 10 seeds, frozen configuration):
  per-instance standard deviation and coefficient of variation of Z1min, HV
  and IGD+.

Later experiments (the method of [7], misspecification of A3, budget
robustness at 0.25/0.5/2 x clock, parameter and instance sensitivity, 200-300
jobs, CP-SAT fronts) are exploratory; their designs are appended here before
they are run.

## 7. Amendments

### Amendment 1 (6 November 2026; package v7; written before any validation result is analysed)

During the calibration run the laptop's battery ran out and the run
resumed afterwards; the progress counter shows about two hours more elapsed
than the runs account for. A run in progress during such a pause keeps its
wall clock running while it does not compute. Nothing below depends on any
result.

- Interrupted runs. A run whose wall time exceeds its clock by more than 25%
  plus 2 s is treated as interrupted: every method stops on the wall clock
  (measured maximum 0.5% over it), so such a run computed for less than its
  clock. It is set aside (`<out>/results.jsonl.suspended`) and run again.
  CPU time is not used for this, because with more workers than free cores
  it falls below wall time for every run alike.
- Cut-off lines. A line left incomplete by a power cut is set aside
  (`<out>/results.jsonl.bad`); the run is done again. Records are now forced
  to disk as they are written.
- Holm family. Section 4 adjusts over the five comparisons of H with the
  other main methods. `analyze_fair.py` adjusted over every pair present and
  did not test H against ALNS and NSGA-II; `--pairs first` now does exactly
  what section 4 says, and `confirm.py` applies the rules of section 5 as
  written.
- C6 uses replications 0-3 from `results_cpsatgap` and replication 4 from
  `results_valid` (the verification-scale instances of the validation set),
  together the 135 instances of section 1. The 11.5% of the submitted paper
  is CPM's median distance over the 24 instances CP-SAT proved optimal (6.8%
  over all 135); both are reported.

## 8. Exploratory experiments (designs fixed before they are run)

### E1. The method of the companion study [7] (reviewer 2, comment 2)

- Method: FFD+SA-EDDTS of [7], code of its archive version 1.2.1 used
  unchanged (`comp7/`), with the correction switches and tuned parameters of
  the revised [7]; ten subproblems (two payoff runs, eight grid points) as in
  [7]'s campaign. Input translated: setup matrix sigma(f, g) = sigma_f for
  f != g (A3). Every returned schedule is re-evaluated by
  `core.evaluate_schedule`; a disagreement is counted and the point dropped.
  [7]'s parameters are not retuned here: its screening found the method
  insensitive to its three parameters (level means within 0.02 normalised
  HV).
- Variant FFD-EDDTS (no annealing step): [7]'s own ablation found the
  annealing step does not help, so both are run and the better is reported
  as [7] alongside the other.
- Effort: the clock of section 2; [7]'s rule of equal budgets per
  subproblem, the budget set once Stage 1 has finished to (clock - time
  used) / 10.
- Instances: validation set (replications 4-9, all five scales), one seed
  (the replication number), 12 workers, factor 0.47; `run_comp7.sh`.
- Analysis: H, CPM, ALNS+R and the two variants, normalised by the union of
  these five; Wilcoxon with Holm over the eight pairs `analyze_fair.py`
  tests for these methods (H against FFD+SA-EDDTS, FFD-EDDTS, CPM and
  ALNS+R; CPM against ALNS+R and FFD+SA-EDDTS; ALNS+R against FFD+SA-EDDTS;
  FFD-EDDTS against FFD+SA-EDDTS); attainment of the Z2 bound; setup-budget
  curves.
  Descriptive: the question is how much exploiting A3 gains over a method
  that does not use it, not a confirmatory claim.

### E1b. The method of [7] with twice the clock (fixed 6 November 2026, after E1)

E1 left [7]'s method far from the others at 100 and 150 jobs, where its
per-subproblem budget is 12-13 s and its own campaign gave 200 s per front at
100 jobs. To see whether the gap is a budget effect: FFD+SA-EDDTS with
`--clock-mult 2` on the validation instances at 100 and 150 jobs, compared
with H and CPM at the normal clock (normalisation by the union of the three).
Descriptive.

### Calibration rerun (6 November 2026)

The 12 calibration runs interrupted by the battery pause were run again under
amendment 1 after the validation run; `tune.py` chose ALNS+R#8 and
NSGA-II+R#8 again, the configuration used in the validation run.

### Amendment 2 (7 November 2026; after the validation results, raised by an external reader)

- C3 cannot fail. H's front is the non-dominated union of its ALNS+R part
  and its CPM part, and the CPM part contains B, so H_A+B is weakly dominated
  by H and HV(H) >= HV(H_A+B) on every instance (minimum difference over the
  378 instances: 0). C3 therefore measures whether the ladder adds points,
  not whether it pays for its share of the clock. It is reported
  descriptively (share of instances where it adds HV, mean gain). The
  inferential test of the ladder is C4 (H against ALNS+R+B, which charges the
  ladder its time); the claim about level traversal rests on C4 alone. The
  decision rule of C3 in section 5 is withdrawn; this is stated in the paper.
- Z2 grid. Setup times are multiples of 1/12 h stored to four decimals, so
  the same setup level summed in a different order can differ in the fourth
  decimal and count as two non-dominated points (under 1% of points in every
  method). Analyses put Z2 on the 1/12 h grid before the dominance filter
  (`--z2-grid 12`, default; `0` for perturbed or pair-dependent setups). On
  the validation results no decision changes (largest change in any p_H:
  0.043 -> 0.045). Point origin is reported as CPM part only, ALNS+R part
  only, or both.
- Results by regime. Results involving the method of [7] and C4 are also
  reported by e = |F| - m (< 0, = 0, > 0), descriptively (`regime.py`); [7]
  holds |F| > m, so e > 0 is the regime its method was designed for.
- Wording. The paper does not call the analysis "pre-registered"; it states
  that the decision rules were fixed in ANALYSIS_PLAN.md before the
  validation run and that this file is archived with the code, and it lists
  amendments 1 and 2.

## 9. Further exploratory experiments (designs fixed 7 November 2026, before they are run)

### E1c. The method of [7] with the step R

FFD+SA-EDDTS+R: [7] on 85% of the clock, then R on every point of its front
within the rest (as CPM+R); the front before R is logged. Validation set,
one seed. Analysis with H, CPM, ALNS+R, ALNS and FFD+SA-EDDTS (from E1),
overall and by regime. Purpose: separate what R explains (tardiness) from
what the structure explains (setup end).

### Ablations on the validation set

CPM+R and CPM-blind (deterministic) on replications 4-9, so that every
number in the paper comes from held-out instances; the development-set
results are kept for the supplementary material.

### Budget robustness

H, ALNS+R, NSGA-II+R and CPM with the frozen configuration at 0.25, 0.5 and
2 times the clock on replications 4-6 of the validation set (189 instances);
the x1 runs are those of the validation run. Per multiplier: analyze_fair
--pairs first (H against the other three). Across multipliers: each method's
mean normalised HV and IGD+, normalised per instance by the union of all
methods and multipliers.

### E2. Misspecification of A3 (reviewer 1 comment 1, reviewer 2 comment 6)

- True setup matrix, Eq. (21) of [7] on this paper's families:
  sigma(f, g) = gamma_f sigma_f (1 + lambda max(0, d_f - d_g) / 7) for
  f != g, 0 for f = g; sigma_f from instances.FAMILY_POOL; d_f the position
  in FAMILY_POOL (light to dark); gamma_f = 1.8 for the chromatic families
  Pastel and Brown (as pink and brown in [7]), 1 otherwise; lambda in
  {1.0, 2.0} ([7]'s baseline and its strongest level; lambda = 0 is A3).
- Planning value of a family's setup: sigma_max(f) = max_{g != f}
  sigma(f, g) (a wash standard sized for the worst successor) - primary;
  sigma_mean(f) - variant (H only, lambda = 1). Fixed before the run.
- Certified bracket of the true optimum: LB = Corollary 3 with
  sigma_min(f) = min_{g != f} sigma(f, g) (every departure from f costs at
  least that, so LB <= Z2*); UB = true setup of the Corollary 3 schedule
  planned with sigma_max. For e <= 0 both are 0 and B remains optimal.
- Methods: H-max (the frozen hybrid planning with sigma_max; its final
  schedules evaluated under the true matrix; also logged within the same
  run: H-max+D, each schedule's family blocks on each machine reordered light
  to dark, both versions entering the front); H-mean; ALNS+R native (the true
  matrix in the evaluator); FFD+SA-EDDTS+R native (with [7]'s front before R
  logged).
- Instances: validation set, 60 and 150 jobs (replications 4-9, 108
  instances, all regimes) at both lambda levels.
- Analysis: all fronts under the true matrix, --z2-grid 0, normalisation by
  the union of the methods; pairwise IGD+ and HV (H-max against each other),
  least setup relative to LB and UB, by regime. Either outcome is reported:
  if H-max is competitive, planning with A3 is a usable approximation; if
  not, the result marks the boundary between this paper and [7].

### Sensitivity (compact, one factor at a time)

- Base: the e > 0 cells at 60 jobs (m = 3) and 150 jobs (m = 4), tau = 0.5,
  replications 4-9 (12 instances); base results from the validation run.
- Factors: sigma x2 and x4 (setup to processing ratio; reviewer 2 comment 7
  on weak conflict); p and sigma with +/-20% error (each family's p and
  sigma multiplied by U[0.8, 1.2], five draws per instance; methods plan on
  the nominal instance, fronts evaluated on each draw; reviewer 1 comment 2);
  skewed customer shares (shares proportional to 1/k) and clustered due
  dates (two clusters) (reviewer 1 comment 5); Q = 300 kg; job sizes
  U[100, 350] kg.
- Methods: H, ALNS+R, NSGA-II+R, CPM, frozen configuration, normal clock.
- CPM's hand-set parameters: one at a time on the calibration replications
  (2, 3) at 60 and 100 jobs, inside the hybrid; one table in the
  supplementary material.

### Scalability

200 and 300 jobs, |F| = 8, |C| = 3, m in {4, 8, 12}, tau = 0.5,
replications 4-9; H, ALNS+R, NSGA-II+R, CPM; clock 281 s x n/150 x 0.47.

### Reproduction check of the adapter for [7]

Three tasks of [7]'s own campaign (30 jobs): [7]'s runner (rerun.py) and
this repository's wrapper on [7]'s native instance and matrix, with [7]'s
budget; least tardiness, least setup and front size against the values
stored in [7]'s archive.

### Dropped

CP-SAT fronts (C6 answers the exactness question at 12-20 jobs) and the
adaptation of [33] (replaced by a table of why each method of Table 1 does
not transfer).

### Amendment 3 (7 November 2026, before E1c, the validation ablations, budget robustness and section 9 runs)

- R in CPM+R and in FFD+SA-EDDTS+R uses ALNS+R's frozen number of rounds
  (5), least-tardiness point first, with exhaustion of R's budget logged;
  runs of these two methods made with 3 rounds are set aside and redone.
  The development-set CPM+R runs (3 rounds) are reported as such.
- Wording: B is computed first and without interruption; its time is
  deducted from the clock (it is not "outside the clock").
- E2 additions, fixed before E2 is run:
  (a) Guarantee of sigma_max planning. With a fixed sequence, true setups
      no longer than planned give completion times no later than planned,
      so for every schedule planned with sigma_max: true Z1 <= planned Z1
      and true Z2 <= planned Z2. Checked on every E2 schedule (a violation
      is a bug); no such guarantee holds for sigma_mean.
  (b) H-mix: B and CPM-A planned with sigma_max, the ALNS+R part searching
      with the true matrix, everything evaluated with the true matrix (union
      design, same clock and parameters as H).
  (c) +D defined: on each machine the batches are stably sorted by depth,
      light to dark (merging the pieces of a family), then R (redistribution
      and exact assignment under the true matrix) is applied; original and
      +D schedules both enter the front; the number of machines whose last
      family changes is logged.
  (d) Exact setup optimum without splitting and under contiguity: with
      every family on one machine in one block, the minimum true setup
      depends only on the families' assignment and order; Held-Karp over
      subsets plus a partition of the families into at most m groups gives
      it exactly (Z2*_contig). Reported between LB (valid for every
      schedule, including split ones) and UB, labelled "exact without
      splitting, under contiguity".
- Sensitivity: all three tau levels of the base cells (18 instances per
  scale), so that a Holm-adjusted Wilcoxon test can reach 0.05 (with 6
  instances the smallest two-sided p is 0.031). Criterion for the +/-20%
  error: for each draw, each method's front is re-evaluated on the perturbed
  instance and filtered for dominance; indicators per draw with the union of
  the methods' perturbed fronts; per instance the mean over the five draws;
  H against the other three as in section 4; plus the relative change of
  Z1min from the nominal plan to the perturbed evaluation.
- Scalability: the time of B, CPM, CPM-A and ALNS+R inside H is logged and
  reported as shares of the clock.

### Amendment 4 (7 November 2026, evening; after the v8.1 runs, before the v9a runs)

The section 9 runs are split. v9a (tonight): the reproduction check of the
adapter for [7], sensitivity, scalability and CPM's parameters, all with
the frozen methods. v9b (8 November): E2, whose design is unchanged.
Definitions fixed before v9a is run:

- Sensitivity base: the variant `perturb` plans on the nominal instance
  (identical to the validation instance) and its nominal fronts are the base
  of the one-factor comparisons; the validation-run fronts of the same
  instances are kept as a check. Factor `c5` added (five customers instead
  of three; reviewer 2, comment 7, names the number of customers). Clustered
  due dates: each job at 0.25 or 0.75 of the Fisher window (probability 1/2
  each), jittered by U(-0.05, 0.05) of the window. Skewed customers: shares
  1 : 1/2 : 1/3. Q = 300 kg and job sizes U[100, 350] kg change the Fisher
  makespan proxy and hence the due dates, so tightness stays defined
  relative to the load. Every variant keeps the seed of its base instance.
  Analysis per variant and scale: H against ALNS+R, NSGA-II+R and CPM as in
  section 4 (IGD+ PS with Holm over the three, Z1min median difference,
  attainment of the setup optimum), and the change of each method's
  indicators relative to the base. Perturbation: as in amendment 3, with
  --z2-grid 0.
- CPM's parameters, one at a time inside the hybrid (`cpm_params.py`),
  calibration replications 2-3 at 60 and 100 jobs (36 instances), default
  run in the same batch: outer passes 2/4/6, first-sweep share 0.5/0.7/0.9,
  ladder levels per sweep 20/40/80, stall levels 3/5/10, exhaustive block
  orders up to 4/6/8, exact joint block orders below 16/64/256, CPM-A rounds
  3/6/12 (defaults in the middle). Reported: mean HV and IGD+ (union of the
  15 configurations), Z1min median difference against the default, and
  Wilcoxon against the default without adjustment, as a screening table in
  the supplement; no parameter is changed for the paper.
- Scalability: H, ALNS+R, NSGA-II+R, CPM at 200 and 300 jobs (|F| = 8,
  |C| = 3, m = 4/8/12, tau = 0.5, replications 4-9, 18 instances each),
  clocks 176.1 and 264.1 s; same indicators as the main comparison, by
  regime, and the time shares of B, CPM, CPM-A and ALNS+R inside H.
- Reproduction check of the adapter for [7] (`check_comp7.py`): three tasks
  of [7]'s campaign (30 jobs, LOOSE/MEDIUM/TIGHT), two runs each by [7]'s
  own runner and by the adapter, against the five stored runs per task. The
  adapter passes if its least tardiness and least setup lie as close to the
  stored range as the own runner's values do (plus 1%), and if its
  parameters equal those recorded by [7]'s campaign.
- Budget robustness is reported as supplementary Table S5.

### Amendment 5 (8 November 2026, before the E2 run)

Implementation of E2, fixed before it is run (e2.py, e2_analyze.py):

- Bracket checked before the run, as it needs no search: on all 108 E2
  instances and both lambda levels LB(sigma-) = UB, so the Corollary 3
  schedule planned with sigma+ is setup-optimal under the true matrix and
  Z2* is known exactly (the families left before a final block are the
  lightest, and a change to a darker family carries no penalty in Eq. (21)).
  Z2*_contig equals it. The analysis therefore reports attainment of Z2*.
- H-max and H-mean run the frozen hybrid on 95% of the clock; +D (light to
  dark, then R under the true matrix) uses the remaining 5%, so that H-max+D
  has the same clock as the other methods; H-max is the front of the same
  run before +D (5% less time, which by the budget results costs it at most
  a few tenths of a per cent). The H-max front is the planned front
  (non-dominated under the planning values) evaluated under the true
  matrix; H-mean also gets +D, logged as H-mean+D.
- H-mix: B, CPM and CPM-A on the sigma+ instance, ALNS+R on the true
  instance (hybrid.cpm_alns_r, argument search_inst), union under the true
  matrix, full clock.
- ALNS+R and the constructive heuristics of benchmarks.py read the setup
  through core.setup_time, which uses the matrix when an instance has one
  (identical results under A3; golden checks unchanged).
- [7]+R native: [7] receives this paper's family names with the true matrix
  in its own format; its search reads setups only through that matrix.
- Main set for normalisation: H-max, H-max+D, H-mix, ALNS+R, [7]+R. H-max
  against each (IGD+ PS, Holm over the four; Z1min median difference),
  attainment of Z2*, by regime; [7] before and after R; H-mean against
  H-max at lambda = 1; guarantee violations (expected none) and the optimism
  of sigma_mean planning; +D statistics.

### Note (9 November 2026): version of the companion code

Version 1.2.1 of the companion archive, named in section 8, was the authors'
copy before publication and was never published; the archive was published
as version 1.3.0 under the same DOI. Checked by SHA-256 and diff:
`dyeing_core.py` and `tuned_params.json` are identical, and `rerun.py` differs
only by the added experiment `nosa`; the functions used here (`load_params`,
`make_task`, `run_task`, `stable_seed`) are identical. No run is affected.
