"""OpenVox [deng2025openvox] published-numbers adapter.

Until the OpenVox code is released, we only carry their reported numbers for
the capability-axis comparison in §V. This file holds the small dataclass
that the §IV table builder consumes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenVoxPublishedNumbers:
    """Reported numbers from OpenVox arXiv 2502.16528.

    The numbers below are deliberately ``None`` until the W6 paper-trawl
    sub-agent fills them in from the paper. **Do not invent values.**
    """

    replica_zero_shot_mIoU: float | None = None
    replica_zero_shot_mAcc: float | None = None
    scannet_zero_shot_mIoU: float | None = None
    scannet_zero_shot_mAcc: float | None = None
    source_arxiv_id: str = "2502.16528"
