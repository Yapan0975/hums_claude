"""Unit tests for M2 submap descriptor + matcher (paper §III.C Eq 7-9)."""

from __future__ import annotations

import torch

from evidlife_map.m2_loop_closure.consistency import disagreement_score, fuse_overlap
from evidlife_map.m2_loop_closure.descriptor import build_submap_descriptor
from evidlife_map.m2_loop_closure.matcher import descriptor_similarity, top_k_matches


def test_descriptor_dims(num_classes_plus_one: int) -> None:
    alpha = 1.0 + torch.rand(64, num_classes_plus_one) * 10.0
    desc = build_submap_descriptor(alpha, submap_id=7, n_vac_bins=10)
    assert desc.h_class.shape == (num_classes_plus_one,)
    assert desc.h_vac.shape == (10,)
    assert desc.submap_id == 7


def test_descriptor_l1_normalised(num_classes_plus_one: int) -> None:
    alpha = 1.0 + torch.rand(64, num_classes_plus_one) * 10.0
    desc = build_submap_descriptor(alpha, submap_id=0)
    assert torch.allclose(desc.h_class.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(desc.h_vac.sum(), torch.tensor(1.0), atol=1e-5)


def test_similarity_is_symmetric_when_beta_split(num_classes_plus_one: int) -> None:
    alpha_a = 1.0 + torch.rand(64, num_classes_plus_one) * 10.0
    alpha_b = 1.0 + torch.rand(64, num_classes_plus_one) * 10.0
    d_a = build_submap_descriptor(alpha_a, submap_id=0)
    d_b = build_submap_descriptor(alpha_b, submap_id=1)
    s_ab = descriptor_similarity(d_a, d_b, beta=0.0)
    s_ba = descriptor_similarity(d_b, d_a, beta=0.0)
    # Cosine on class is symmetric.
    assert abs(s_ab - s_ba) < 1e-5


def test_top_k_excludes_self(num_classes_plus_one: int) -> None:
    descs = [
        build_submap_descriptor(
            1.0 + torch.rand(32, num_classes_plus_one) * 5.0, submap_id=i,
        )
        for i in range(5)
    ]
    query = descs[2]
    matches = top_k_matches(query, descs, k=3, beta=0.5, exclude_self=True)
    assert all(d.submap_id != query.submap_id for d, _ in matches)
    assert len(matches) == 3


def test_disagreement_small_for_near_one_hot_overlap(num_classes_plus_one: int) -> None:
    """Near-one-hot identical posteriors give a near-zero disagreement (Eq 9)."""
    n = 10
    alpha = torch.ones(n, num_classes_plus_one)
    alpha[:, 3] = 1.0e6   # essentially one-hot at class 3
    score = disagreement_score(alpha, alpha)
    assert score < 1e-4

    # Disagreement against an orthogonal one-hot should be high.
    alpha_other = torch.ones(n, num_classes_plus_one)
    alpha_other[:, 7] = 1.0e6
    score_other = disagreement_score(alpha, alpha_other)
    assert score_other > 0.9


def test_disagreement_is_symmetric_and_non_negative(num_classes_plus_one: int) -> None:
    a = 1.0 + torch.rand(10, num_classes_plus_one) * 5.0
    b = 1.0 + torch.rand(10, num_classes_plus_one) * 5.0
    s_ab = disagreement_score(a, b)
    s_ba = disagreement_score(b, a)
    assert s_ab >= 0.0
    assert abs(s_ab - s_ba) < 1e-5


def test_fuse_overlap_preserves_alpha_ge_one(num_classes_plus_one: int) -> None:
    a = 1.0 + torch.rand(8, num_classes_plus_one) * 4.0
    b = 1.0 + torch.rand(8, num_classes_plus_one) * 4.0
    fused = fuse_overlap(a, b)
    assert torch.all(fused >= 1.0 - 1e-6)
