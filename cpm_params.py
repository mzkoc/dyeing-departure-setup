"""CPM's hand-set parameters, one at a time inside the hybrid (reviewer 2,
comment 7: passes, budget split, stopping rule, enumeration limits).

A method name "<hybrid>~<knob>=<value>" runs the hybrid with one parameter
changed, e.g. "CPM+ALNS+R:union@0.1~outer_passes=2". `override(name)` is a
context manager that sets the parameter for the duration of the run and
restores the default afterwards (pool workers are reused between runs).

  knob            where                                  default  levels
  outer_passes    L1 feedback passes (cpm_report)        4        2, 6
  first_share     share of the first sweep (cpm_report)  0.7      0.5, 0.9
  pareto_passes   ladder levels per sweep (solve_once)   40       20, 80
  stall_levels    levels without Z1 gain before stop     5        3, 10
  block_perm      exhaustive block orders up to          6        4, 8
  joint_orders    exact joint block orders below         64       16, 256
  adapt_rounds    CPM-A rounds (hybrid.adapt)            6        3, 12
"""

from __future__ import annotations

import contextlib

LEVELS = {"outer_passes": (4, (2, 6)), "first_share": (0.7, (0.5, 0.9)),
          "pareto_passes": (40, (20, 80)), "stall_levels": (5, (3, 10)),
          "block_perm": (6, (4, 8)), "joint_orders": (64, (16, 256)),
          "adapt_rounds": (6, (3, 12))}


def split(method: str):
    """('CPM+ALNS+R:union@0.1', 'outer_passes', 2) or (method, None, None)."""
    if "~" not in method:
        return method, None, None
    base, kv = method.split("~", 1)
    k, v = kv.split("=", 1)
    if k not in LEVELS:
        raise ValueError(f"unknown CPM parameter {k}")
    d = LEVELS[k][0]
    return base, k, (float(v) if isinstance(d, float) else int(v))


def variant_names(hybrid: str):
    """The default and every non-default level, one at a time."""
    out = [hybrid]
    for k, (_, alts) in LEVELS.items():
        out += [f"{hybrid}~{k}={v}" for v in alts]
    return out


def _slots():
    import cpm
    import hybrid
    import layer234
    kw = cpm.cpm_report.__kwdefaults__
    return {
        "outer_passes": (lambda: kw["outer_passes"],
                         lambda v: kw.__setitem__("outer_passes", v)),
        "first_share": (lambda: kw["first_share"],
                        lambda v: kw.__setitem__("first_share", v)),
        "pareto_passes": (
            lambda: cpm.solve_once.__kwdefaults__["max_passes"],
            lambda v: cpm.solve_once.__kwdefaults__.__setitem__(
                "max_passes", v)),
        "stall_levels": (lambda: cpm.STALL_LEVELS,
                         lambda v: setattr(cpm, "STALL_LEVELS", v)),
        "block_perm": (lambda: layer234.MAX_BLOCK_PERM,
                       lambda v: setattr(layer234, "MAX_BLOCK_PERM", v)),
        "joint_orders": (lambda: layer234.MAX_JOINT_ORDERS,
                         lambda v: setattr(layer234, "MAX_JOINT_ORDERS", v)),
        "adapt_rounds": (lambda: hybrid.adapt.__defaults__[0],
                         lambda v: setattr(hybrid.adapt, "__defaults__",
                                           (v,))),
    }


def current():
    return {k: get() for k, (get, _) in _slots().items()}


@contextlib.contextmanager
def override(method: str):
    _, k, v = split(method)
    if k is None:
        yield
        return
    get, put = _slots()[k]
    old = get()
    put(v)
    try:
        yield
    finally:
        put(old)
