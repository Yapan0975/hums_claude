/* nvblox_evidential — per-voxel record (paper §III.A, §III.F).
 *
 * EvidentialVoxel stores the Dirichlet concentration α_v ∈ R^{C+1}, the last
 * evidence-update timestamp (consumed by the M3 decay kernel), and the pose
 * index of the last contributing observation (so we can attribute removed
 * voxels back to the LVIO stream for debugging).
 *
 * Layout — at C+1 = 20 (closed-set RQ1/RQ4/RQ5):
 *
 *     float    alpha[20]              =  80 B
 *     uint32_t last_update_ts         =   4 B
 *     uint16_t pose_idx               =   2 B
 *     uint16_t _padding               =   2 B
 *     ------------------------------- = 88 B
 *
 * 512 voxels per 8×8×8 block ⇒ 44 032 B ≈ 44 kB per fully-allocated block,
 * matching the paper §III.B implementation paragraph and the T1.5 nvblox
 * deep-read §9.
 *
 * The template parameter ``Cp1`` is the *compile-time* value of C+1; the
 * downstream layer cake instantiates EvidentialLayer<EvidentialVoxel<20>> for
 * closed-set runs and EvidentialLayer<EvidentialVoxel<15>> for open-set
 * (14-known + 1 unknown).
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace nvblox_evidential {

template <std::size_t Cp1>
struct EvidentialVoxel {
    static_assert(Cp1 >= 2,
                  "EvidentialVoxel requires C+1 >= 2 (matches paper §III.A)");

    // Dirichlet concentration α_v ∈ R^{C+1}. Invariant: α_k >= 1.0f
    // (paper Eq 2 requires this for vacuity to be well-defined).
    float alpha[Cp1];

    // Seconds since the layer's epoch t0; ``0`` means "never updated".
    // 32 bits give ~136 years of headroom; plenty for any single mission.
    std::uint32_t last_update_ts;

    // Index into the LVIO pose ring buffer at the time of last update.
    // Used by the debugger to attribute decayed voxels back to a frame.
    std::uint16_t pose_idx;

    // Explicit padding so sizeof(EvidentialVoxel) is a multiple of 4 B.
    std::uint16_t _padding;
};

// Compile-time check on the layout — keeps the published "88 B / voxel at
// C+1 = 20" claim accurate.
static_assert(sizeof(EvidentialVoxel<20>) == (20 * sizeof(float) + 8),
              "EvidentialVoxel<20> must be 88 B; if this fails, update "
              "README.md and paper §III.F memory accounting accordingly.");

// Vacuity helper (paper §III.B Eq 2):  u_v = (C+1) / S_v
template <std::size_t Cp1>
__host__ __device__ inline float vacuity_from(const EvidentialVoxel<Cp1>& v) {
    float s = 0.0f;
    for (std::size_t k = 0; k < Cp1; ++k) {
        s += v.alpha[k];
    }
    return static_cast<float>(Cp1) / s;
}

}  // namespace nvblox_evidential
