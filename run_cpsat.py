"""CP-SAT reference on the verification scale (Section 7.7 of the paper).

Replications 0-4 of the verification cells, 135 instances. The warm start
uses the proposed method with a budget of 3 000 evaluations; on this scale
the method stops well short of it (about 200 evaluations on average and
never more than 1 200), so the budget does not bind and the warm start
matches the main study.
"""

from verify_run import run_verification, summarise

csv2 = run_verification(
    out_dir     = "results",
    taus        = (0.3, 0.5, 0.7),
    reps        = 5,
    det_time    = 30.0,
    eval_budget = 3000,
    verbose     = True,
)
summarise(csv2)
