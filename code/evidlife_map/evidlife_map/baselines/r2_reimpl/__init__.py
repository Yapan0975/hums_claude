"""R2 [jiao2024r2] re-implementation (paper_tables.md Table III row 1).

R2 publishes no code, so this directory is the **canonical** R2-reimpl that
the §IV RQ1 main table compares against. See ``r2_pipeline.py`` for the
six-stage pipeline and ``bayes_filter.py`` for the per-voxel argmax-Bayes
update we recover from the R2 cleaned-text reference.
"""

from __future__ import annotations

from evidlife_map.baselines.r2_reimpl.bayes_filter import (
    ArgmaxBayesAccumulator,
    argmax_bayes_update,
)
from evidlife_map.baselines.r2_reimpl.r2_pipeline import R2Pipeline, R2PipelineConfig

__all__ = [
    "ArgmaxBayesAccumulator",
    "R2Pipeline",
    "R2PipelineConfig",
    "argmax_bayes_update",
]
