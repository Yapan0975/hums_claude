/* nvblox_evidential — EvidentialIntegrator (paper §III.F).
 *
 * P1 skeleton: per-scan integration loop is wired up using a brute-force
 * floor-divide voxelisation. Production swaps in the upstream nvblox
 * projective LiDAR integrator so the GPU block allocator and the
 * ray-cast logic are reused unchanged.
 */

#include "nvblox_evidential/evidential_integrator.h"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nvblox_evidential {

namespace {

inline std::uint64_t packVoxelKey(std::int32_t x,
                                  std::int32_t y,
                                  std::int32_t z) {
    // Same 21-bit packing as evidlife_map/core/voxel_map.py.
    constexpr std::uint64_t kMask = (1ULL << 21) - 1ULL;
    const std::uint64_t ux = static_cast<std::uint64_t>(x) & kMask;
    const std::uint64_t uy = static_cast<std::uint64_t>(y) & kMask;
    const std::uint64_t uz = static_cast<std::uint64_t>(z) & kMask;
    return (ux << 42) | (uy << 21) | uz;
}

inline void transformPoint(const float* pose,
                           float x, float y, float z,
                           float& xo, float& yo, float& zo) {
    // pose is 4x4 row-major.
    xo = pose[0] * x + pose[1] * y + pose[2] * z + pose[3];
    yo = pose[4] * x + pose[5] * y + pose[6] * z + pose[7];
    zo = pose[8] * x + pose[9] * y + pose[10] * z + pose[11];
}

}  // namespace

template <std::size_t Cp1>
void EvidentialIntegrator<Cp1>::integrateScan(const float* points_sensor,
                                              const float* probs_per_point,
                                              std::size_t num_points,
                                              const float* pose_world_T_sensor,
                                              std::uint32_t timestamp_s,
                                              std::uint16_t pose_idx) {
    const float voxel_size = layer_->voxelSize();

    // Aggregate per-voxel evidence on the host first; one batched upload.
    std::unordered_map<std::uint64_t, std::vector<float>> aggregated;
    aggregated.reserve(num_points);

    for (std::size_t i = 0; i < num_points; ++i) {
        float xw, yw, zw;
        transformPoint(pose_world_T_sensor,
                       points_sensor[3 * i + 0],
                       points_sensor[3 * i + 1],
                       points_sensor[3 * i + 2],
                       xw, yw, zw);
        const std::int32_t ix = static_cast<std::int32_t>(std::floor(xw / voxel_size));
        const std::int32_t iy = static_cast<std::int32_t>(std::floor(yw / voxel_size));
        const std::int32_t iz = static_cast<std::int32_t>(std::floor(zw / voxel_size));
        const std::uint64_t key = packVoxelKey(ix, iy, iz);

        auto& acc = aggregated[key];
        if (acc.empty()) acc.assign(Cp1, 0.0f);
        for (std::size_t k = 0; k < Cp1; ++k) {
            acc[k] += probs_per_point[i * Cp1 + k];
        }
    }

    // Single batched call into the layer.
    std::vector<std::uint64_t> keys;
    std::vector<float> evidence_flat;
    keys.reserve(aggregated.size());
    evidence_flat.reserve(aggregated.size() * Cp1);
    for (auto& kv : aggregated) {
        keys.push_back(kv.first);
        evidence_flat.insert(evidence_flat.end(), kv.second.begin(), kv.second.end());
    }
    if (!keys.empty()) {
        layer_->addEvidenceBatch(keys.data(),
                                 evidence_flat.data(),
                                 keys.size(),
                                 timestamp_s,
                                 pose_idx);
    }
}

template class EvidentialIntegrator<20>;
template class EvidentialIntegrator<15>;

}  // namespace nvblox_evidential
