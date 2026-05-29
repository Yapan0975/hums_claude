"""EvidLife-Map package root.

This is the importable entry point for the paper's three modules:

    * :mod:`evidlife_map.m1_evidential` — Dirichlet evidential per-voxel fusion
      (paper §III.B, Eq 1–6).
    * :mod:`evidlife_map.m2_loop_closure` — vacuity-conditioned loop closure
      (paper §III.C, Eq 7–9).
    * :mod:`evidlife_map.m3_decay` — vacuity-driven voxel decay (paper §III.D,
      Eq 10–12).

The :mod:`core` package holds the shared abstractions; :mod:`eval` holds the
metrics that §IV will consume; :mod:`baselines` holds the wrappers for the
six baseline systems listed in paper_tables.md Table III.
"""

from __future__ import annotations

__version__ = "0.1.0a1"
__all__ = ["__version__"]
