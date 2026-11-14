"""Parameters of the benchmarks and their calibration designs.

ALNS+R and NSGA-II+R are calibrated with the same protocol as the hybrid
(tune.py, calibration replications 2 and 3). Each has four three-level
factors, crossed in the standard L9(3^4) orthogonal array. The array is a
fractional design: it is used here only to choose among the nine tested
combinations plus the default (row 0), not to predict untested ones.

Method names:   ALNS+R#0 .. ALNS+R#9,  NSGA-II+R#0 .. NSGA-II+R#9
Row 0 is the default, which reproduces the runs made before calibration.

A frozen choice is a JSON file (written by tune.py) such as
    {"ALNS+R": {"dmax": 0.5, ...}, "NSGA-II+R": {...},
     "hybrid": "CPM+ALNS+R:union@0.05"}
and is passed to fair_compare.py with --params; plain "ALNS+R" and
"NSGA-II+R", and the ALNS+R part of the hybrid, then use it.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

# L9(3^4): rows of levels (0, 1, 2) for four factors
L9 = [(0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2),
      (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1),
      (2, 0, 2, 1), (2, 1, 0, 2), (2, 2, 1, 0)]

FACTORS = {
    "ALNS+R": [("dmax", (0.25, 0.35, 0.50)),      # largest destroy fraction
               ("segment", (10, 25, 50)),          # weight-update segment
               ("decay", (0.6, 0.8, 0.9)),         # weight reaction
               ("r_rounds", (1, 3, 5))],           # rounds of R
    "NSGA-II+R": [("pop_size", (30, 60, 120)),
                  ("p_mut", (0.1, 0.2, 0.4)),
                  ("refresh", (5, 10, 20)),        # batch refresh period
                  ("r_rounds", (1, 3, 5))],
}
# the defaults are the middle levels: the settings used before calibration
DEFAULTS = {base: {name: levels[1] for name, levels in fs}
            for base, fs in FACTORS.items()}


def row_params(base: str, row: int) -> Dict[str, float]:
    if row == 0:
        return dict(DEFAULTS[base])
    lv = L9[row - 1]
    return {name: levels[lv[i]] for i, (name, levels)
            in enumerate(FACTORS[base])}


def parse_variant(method: str) -> Optional[Tuple[str, int]]:
    """("ALNS+R", 3) for "ALNS+R#3"; None if the name has no #row."""
    if "#" not in method:
        return None
    base, _, row = method.partition("#")
    if base not in FACTORS:
        raise ValueError(f"no calibration design for {base}")
    r = int(row)
    if not 0 <= r <= 9:
        raise ValueError(f"row must be 0..9: {method}")
    return base, r


def load_frozen(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return json.load(fh)


def params_for(method: str, frozen: dict) -> Dict[str, float]:
    """Parameters a benchmark method runs with."""
    v = parse_variant(method)
    if v is not None:
        return row_params(*v)
    if method in FACTORS:
        return dict(frozen.get(method, DEFAULTS[method]))
    return {}
