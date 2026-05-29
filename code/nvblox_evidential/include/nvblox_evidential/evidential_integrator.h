/* nvblox_evidential — EvidentialIntegrator (paper §III.B / §III.F).
 *
 * Sits at the same nvblox level as LidarIntegrator / TsdfIntegrator. Consumes:
 *
 *     - one LiDAR scan (points in sensor frame)
 *     - the sensor pose in world frame from the LVIO stack
 *     - per-point class probabilities from the EDL head
 *
 * Performs ray-cast splatting into the EvidentialLayer using nvblox's standard
 * projective integrator, then forwards the per-voxel aggregated evidence to
 * `EvidentialLayer::addEvidenceBatch` (paper Eq 4).
 */
#pragma once

#include <cstdint>
#include <cstddef>

#include "nvblox_evidential/evidential_layer.h"

namespace nvblox_evidential {

template <std::size_t Cp1>
class EvidentialIntegrator {
   public:
    explicit EvidentialIntegrator(EvidentialLayer<Cp1>* layer)
        : layer_(layer) {}

    // points_sensor:   N x 3   (float, sensor frame, metres)
    // probs_per_point: N x Cp1 (float, sum-to-one per row, EDL head output)
    // pose_world_T_sensor: 4x4 row-major (float)
    void integrateScan(const float* points_sensor,
                       const float* probs_per_point,
                       std::size_t num_points,
                       const float* pose_world_T_sensor,
                       std::uint32_t timestamp_s,
                       std::uint16_t pose_idx);

   private:
    EvidentialLayer<Cp1>* layer_;
};

}  // namespace nvblox_evidential
