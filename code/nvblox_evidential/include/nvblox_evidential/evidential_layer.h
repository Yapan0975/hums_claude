/* nvblox_evidential — EvidentialLayer (paper §III.F).
 *
 * Subclasses the upstream nvblox BlockLayer for an 8×8×8 EvidentialVoxel
 * block. Exposes the API the Python bridge consumes:
 *
 *     addEvidence(voxel_key, evidence_vector)   — paper Eq 4
 *     getAlpha(voxel_key)                       — read α_v
 *     getVacuity(voxel_key)                     — paper Eq 2
 *     decay(timestamp)                          — paper Eq 12 sweep
 *
 * The block size matches the canonical nvblox value (8) so that the GPU
 * streaming-multiprocessor tiling, mesh extractor, and BlockLayer iteration
 * order are all reused unchanged (paper §III.F: "the only structural change
 * relative to a baseline nvblox semantic layer is the wider voxel record
 * and the additional per-voxel timestamp").
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "nvblox_evidential/evidential_voxel.h"

namespace nvblox_evidential {

template <std::size_t Cp1>
class EvidentialLayer {
   public:
    using VoxelType = EvidentialVoxel<Cp1>;
    static constexpr std::size_t kBlockEdge = 8;
    static constexpr std::size_t kVoxelsPerBlock = kBlockEdge * kBlockEdge * kBlockEdge;

    EvidentialLayer(float voxel_size_m, std::uint32_t epoch_seconds);
    ~EvidentialLayer();

    EvidentialLayer(const EvidentialLayer&) = delete;
    EvidentialLayer& operator=(const EvidentialLayer&) = delete;

    // -----------------------------------------------------------------------
    // M1 — paper Eq 4 conjugate evidence accumulation.
    // -----------------------------------------------------------------------
    void addEvidence(std::uint64_t voxel_key,
                     const float* evidence_vector,
                     std::uint32_t timestamp_s,
                     std::uint16_t pose_idx);

    // Batched variant — the host hands in N evidence vectors and N keys, the
    // GPU kernel does the per-block aggregation.
    void addEvidenceBatch(const std::uint64_t* voxel_keys,
                          const float* evidence_matrix,   // N x (C+1) row-major
                          std::size_t batch_size,
                          std::uint32_t timestamp_s,
                          std::uint16_t pose_idx);

    // -----------------------------------------------------------------------
    // Accessors — paper §III.E consumers.
    // -----------------------------------------------------------------------
    bool getAlpha(std::uint64_t voxel_key, float* alpha_out) const;
    float getVacuity(std::uint64_t voxel_key) const;

    // -----------------------------------------------------------------------
    // M3 — paper Eq 12 decay kernel.
    //
    // Returns the number of voxels actually touched (those with age >
    // ``age_hysteresis_s`` per the implementation note in paper §III.F).
    // -----------------------------------------------------------------------
    std::size_t decay(std::uint32_t timestamp_s,
                      float tau_min_s,
                      float tau_max_s,
                      float age_hysteresis_s);

    // -----------------------------------------------------------------------
    // Bookkeeping.
    // -----------------------------------------------------------------------
    std::size_t numVoxels() const noexcept;
    std::size_t numBlocks() const noexcept;
    float voxelSize() const noexcept { return voxel_size_m_; }

   private:
    float voxel_size_m_;
    std::uint32_t epoch_seconds_;
    // Opaque pointer to the upstream nvblox BlockLayer instance. Defined out
    // of line in the .cu translation unit so that this header stays free of
    // CUDA includes.
    struct Impl;
    Impl* impl_;
};

}  // namespace nvblox_evidential
