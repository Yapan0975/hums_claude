"""M2 — Vacuity-conditioned loop closure (paper §III.C, Eq 7–9).

    Eq 7 — submap descriptor d(S) = (h_class, h_vac)        :mod:`descriptor`
    Eq 8 — similarity score (cosine on class, CE on vac)    :mod:`matcher`
    Eq 9 — geometric registration consistency               :mod:`consistency`

The descriptor's ``h_vac`` channel is the load-bearing distinguishing feature
relative to Hydra / Kimera-Multi loop-closure descriptors.
"""

from __future__ import annotations

from evidlife_map.m2_loop_closure.consistency import disagreement_score
from evidlife_map.m2_loop_closure.descriptor import SubmapDescriptor, build_submap_descriptor
from evidlife_map.m2_loop_closure.matcher import descriptor_similarity, top_k_matches

__all__ = [
    "SubmapDescriptor",
    "build_submap_descriptor",
    "descriptor_similarity",
    "disagreement_score",
    "top_k_matches",
]
