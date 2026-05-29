/* nvblox_evidential — serialiser for inter-session resume (paper §III.C M2).
 *
 * Two formats:
 *
 *   * binary  — packed `EvidentialVoxel<Cp1>` records keyed by voxel key.
 *               Used by the inter-session fusion path so we can drop
 *               session A's α_v into session B's layer without recomputation.
 *
 *   * .pcd    — text PCL format with per-point colour encoding vacuity. Used
 *               by the demo viewer.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#include "nvblox_evidential/evidential_layer.h"

namespace nvblox_evidential {

template <std::size_t Cp1>
bool save_layer_binary(const EvidentialLayer<Cp1>& layer,
                       const std::string& path);

template <std::size_t Cp1>
bool load_layer_binary(EvidentialLayer<Cp1>& layer,
                       const std::string& path);

template <std::size_t Cp1>
bool save_layer_pcd(const EvidentialLayer<Cp1>& layer,
                    const std::string& path,
                    bool colour_by_vacuity = true);

}  // namespace nvblox_evidential
