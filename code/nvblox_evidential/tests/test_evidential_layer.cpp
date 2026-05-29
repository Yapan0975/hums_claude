/* GTest unit tests for nvblox_evidential::EvidentialLayer.
 *
 * These tests deliberately exercise only the host-side bookkeeping (the
 * P1 skeleton routes through cudaMalloc, so they require a CUDA device).
 * Mark the gtest binary with the [needs_gpu] label so CI can skip it
 * when running on a CPU-only runner.
 */

#include "nvblox_evidential/evidential_layer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using nvblox_evidential::EvidentialLayer;

namespace {

constexpr std::size_t kCp1 = 20;

TEST(EvidentialLayer, EmptyHasZeroVoxels) {
    EvidentialLayer<kCp1> layer(0.25f, 0u);
    EXPECT_EQ(layer.numVoxels(), 0u);
}

TEST(EvidentialLayer, AddEvidenceCreatesVoxel) {
    EvidentialLayer<kCp1> layer(0.25f, 0u);
    std::vector<float> evidence(kCp1, 0.5f);
    layer.addEvidence(/*voxel_key=*/42ULL,
                      evidence.data(),
                      /*timestamp_s=*/1u,
                      /*pose_idx=*/0u);
    EXPECT_EQ(layer.numVoxels(), 1u);
}

TEST(EvidentialLayer, VacuityOfFreshVoxelIsBelowOne) {
    EvidentialLayer<kCp1> layer(0.25f, 0u);
    std::vector<float> evidence(kCp1, 1.0f);
    layer.addEvidence(7ULL, evidence.data(), 1u, 0u);
    const float v = layer.getVacuity(7ULL);
    // Uniform prior alpha=1; adding evidence 1.0 each gives alpha=2 each =>
    // S = 2 * Cp1; vacuity = Cp1 / S = 0.5. Allow rounding.
    EXPECT_NEAR(v, 0.5f, 1e-4f);
}

TEST(EvidentialLayer, GetAlphaOfMissingReturnsUniformPrior) {
    EvidentialLayer<kCp1> layer(0.25f, 0u);
    std::vector<float> alpha(kCp1, -1.0f);
    const bool found = layer.getAlpha(99ULL, alpha.data());
    EXPECT_FALSE(found);
    for (std::size_t k = 0; k < kCp1; ++k) {
        EXPECT_EQ(alpha[k], 1.0f);
    }
}

TEST(EvidentialLayer, DecayRespectsHysteresis) {
    EvidentialLayer<kCp1> layer(0.25f, 0u);
    std::vector<float> evidence(kCp1, 2.0f);
    layer.addEvidence(5ULL, evidence.data(), /*timestamp_s=*/10u, 0u);

    // Sweep at t = 11 s with a hysteresis of 5 s => voxel is too fresh.
    const std::size_t touched_short = layer.decay(/*timestamp_s=*/11u,
                                                  /*tau_min_s=*/60.0f,
                                                  /*tau_max_s=*/3600.0f,
                                                  /*age_hysteresis_s=*/5.0f);
    EXPECT_EQ(touched_short, 0u);
}

}  // namespace
