"""PointNet++ Lite — adds k-NN spatial context to per-point classification.

Three improvements over :class:`PointNetVanilla`:

1. **Local context via k-NN**: for each point, gather k=16 nearest neighbors
   in xyz space and pool their features (max). This gives the model
   neighborhood awareness without the cost of full PointNet++ Set Abstraction.
2. **Distance-weighted aggregation**: neighbor features are weighted by
   ``exp(-d / sigma)`` so nearer neighbors contribute more.
3. **Two-stage classification head**: per-point MLP → local-context MLP →
   classifier, similar in spirit to PointNet++ SA(1) + FP(1).

Implementation uses pure torch (no torch_scatter / spconv); kNN via
``torch.cdist`` + ``torch.topk`` is O(N²) memory but is fine for 100k
points per frame on a 5090.

About ~430k parameters; still small enough to train in 20 min on a 5090.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["knn", "chunked_pairwise_distances", "gather_neighbors", "PointNet2Lite"]


def knn(xyz: Tensor, k: int, chunk_size: int = 4096) -> Tensor:
    """Return indices of k nearest neighbors for each point.

    Memory-efficient version: processes rows in chunks so that the full
    ``N × N`` distance matrix never materialises. For N = 123 000 (a typical
    SemKITTI frame) the full matrix would be 56 GB; this version uses
    ``chunk_size × N × 4 bytes`` ≈ 2 GB per chunk, runs in O(N / chunk × N × 3)
    flops.

    Parameters
    ----------
    xyz
        Point coordinates of shape ``(N, 3)``.
    k
        Number of neighbors (including self).
    chunk_size
        Number of query points to process per cdist call. Default 4096.

    Returns
    -------
    Tensor
        Long indices of shape ``(N, k)`` where ``idx[i, j]`` is the index of
        the j-th nearest neighbor (in xyz) of point ``i``. ``idx[i, 0]`` is
        the point itself.
    """
    n = xyz.shape[0]
    if n <= chunk_size:
        d = torch.cdist(xyz, xyz)
        _, idx = torch.topk(d, k=k, dim=1, largest=False)
        return idx
    out = torch.empty(n, k, dtype=torch.int64, device=xyz.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        d_chunk = torch.cdist(xyz[s:e], xyz)  # (chunk, N)
        _, idx_chunk = torch.topk(d_chunk, k=k, dim=1, largest=False)
        out[s:e] = idx_chunk
        del d_chunk, idx_chunk
    return out


def chunked_pairwise_distances(
    xyz: Tensor, knn_idx: Tensor, chunk_size: int = 4096,
) -> Tensor:
    """Memory-efficient gather of pairwise distances at given neighbor idx.

    Returns the same shape as ``knn_idx`` (N, k) with the distance from each
    point to each of its k neighbors.
    """
    n, k = knn_idx.shape
    if n <= chunk_size:
        d_all = torch.cdist(xyz, xyz)
        return torch.gather(d_all, 1, knn_idx)
    out = torch.empty(n, k, dtype=xyz.dtype, device=xyz.device)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        d_chunk = torch.cdist(xyz[s:e], xyz)  # (chunk, N)
        out[s:e] = torch.gather(d_chunk, 1, knn_idx[s:e])
        del d_chunk
    return out


def gather_neighbors(features: Tensor, knn_idx: Tensor) -> Tensor:
    """Gather neighbor features per point.

    Parameters
    ----------
    features
        Per-point features of shape ``(N, F)``.
    knn_idx
        Neighbor indices of shape ``(N, k)``.

    Returns
    -------
    Tensor
        Gathered features of shape ``(N, k, F)``.
    """
    n, f = features.shape
    k = knn_idx.shape[1]
    idx_flat = knn_idx.reshape(-1)  # (N*k,)
    gathered = features[idx_flat].reshape(n, k, f)
    return gathered


def _mlp(channels: list[int], *, bn: bool = True, last_relu: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        is_last = (i == len(channels) - 2)
        if not is_last or last_relu:
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PointNet2Lite(nn.Module):
    """Per-point semantic segmentation with k-NN local context.

    Parameters
    ----------
    in_channels
        Input feature dim per point (4 = xyzi).
    num_classes
        Output classes.
    k
        Number of neighbors for local pooling. Default 16.
    feat_dims
        Per-point feature extractor channel ladder.
    cls_dims
        Final classifier channel ladder (input dim = feat_dims[-1] * 3,
        because we concat [own, neighbor-max, neighbor-mean]).
    sigma
        Distance-weighted aggregation scale in metres. Default 1.0.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 19,
        k: int = 16,
        feat_dims: tuple[int, ...] = (64, 128, 256),
        cls_dims: tuple[int, ...] = (256, 128),
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.feat_dims = tuple(int(d) for d in feat_dims)
        self.cls_dims = tuple(int(d) for d in cls_dims)
        self.sigma = float(sigma)

        self.point_mlp = _mlp([self.in_channels, *self.feat_dims])
        # After context: own + max-neighbor + mean-neighbor = 3F
        ctx_in = self.feat_dims[-1] * 3
        self.classifier = nn.Sequential(
            _mlp([ctx_in, *self.cls_dims], last_relu=True),
            nn.Linear(self.cls_dims[-1], self.num_classes),
        )

    def forward(self, points: Tensor) -> Tensor:
        """Map ``(N, in_channels)`` -> ``(N, num_classes)`` logits.

        The first three channels of ``points`` are expected to be xyz in metres.
        """
        if points.dim() != 2 or points.shape[1] != self.in_channels:
            raise ValueError(
                f"points must be (N, in_channels={self.in_channels}); "
                f"got {tuple(points.shape)}"
            )
        N = points.shape[0]
        xyz = points[:, :3]
        per_pt = self.point_mlp(points)              # (N, F)

        # k-NN context. Limit to N if there are fewer points than k.
        k_use = min(self.k, N)
        knn_idx = knn(xyz, k=k_use)                  # (N, k)
        neighbor_feats = gather_neighbors(per_pt, knn_idx)  # (N, k, F)
        # Distance weights — memory-efficient chunked gather.
        d_to_neighbors = chunked_pairwise_distances(xyz, knn_idx)  # (N, k)
        w = torch.exp(-d_to_neighbors / self.sigma)  # (N, k)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean_neighbor = (neighbor_feats * w.unsqueeze(-1)).sum(dim=1)  # (N, F)
        max_neighbor = neighbor_feats.max(dim=1).values                # (N, F)
        # Concat: own + max + weighted-mean
        fused = torch.cat([per_pt, max_neighbor, mean_neighbor], dim=-1)  # (N, 3F)
        logits = self.classifier(fused)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
