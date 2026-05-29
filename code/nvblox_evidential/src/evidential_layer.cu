/* nvblox_evidential — EvidentialLayer CUDA kernels (paper §III.B Eq 4, Eq 12).
 *
 * P1 skeleton: parameter types, kernel launches, and host-side bookkeeping
 * are wired up; the upstream-nvblox-side block allocator + GPU block-map
 * lookup are stubbed with std::unordered_map for the W1 bring-up. The
 * production replacement is the upstream BlockLayer<EvidentialVoxel>; see
 * the corresponding nvblox docs.
 */

#include "nvblox_evidential/evidential_layer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace nvblox_evidential {

// ===========================================================================
// CUDA kernels
// ===========================================================================
template <std::size_t Cp1>
__global__ void accumulateEvidenceKernel(EvidentialVoxel<Cp1>* voxel,
                                         const float* evidence) {
    const std::size_t k = threadIdx.x;
    if (k >= Cp1) {
        return;
    }
    voxel->alpha[k] += evidence[k];
}

template <std::size_t Cp1>
__global__ void decayKernel(EvidentialVoxel<Cp1>* voxel,
                            float delta_t_s,
                            float tau_min_s,
                            float tau_max_s) {
    const std::size_t k = threadIdx.x;
    if (k >= Cp1) {
        return;
    }
    // Eq 2 — vacuity (compute once per voxel, on thread 0 only).
    __shared__ float multiplier;
    if (k == 0) {
        float s = 0.0f;
        for (std::size_t i = 0; i < Cp1; ++i) {
            s += voxel->alpha[i];
        }
        const float vacuity = static_cast<float>(Cp1) / s;
        const float tau = tau_min_s + (tau_max_s - tau_min_s) * (1.0f - vacuity);
        multiplier = expf(-delta_t_s / tau);
    }
    __syncthreads();

    const float decayed = (voxel->alpha[k] - 1.0f) * multiplier + 1.0f;
    voxel->alpha[k] = decayed < 1.0f ? 1.0f : decayed;
}

// ===========================================================================
// Implementation struct — opaque to header consumers.
// ===========================================================================
template <std::size_t Cp1>
struct EvidentialLayer<Cp1>::Impl {
    // Voxel-key → on-device pointer. The skeleton allocates one device record
    // per voxel; production swaps this for the upstream BlockLayer.
    std::unordered_map<std::uint64_t, EvidentialVoxel<Cp1>*> voxel_map;
    std::unordered_map<std::uint64_t, std::uint32_t> last_update;
};

// ===========================================================================
// Layer methods.
// ===========================================================================
template <std::size_t Cp1>
EvidentialLayer<Cp1>::EvidentialLayer(float voxel_size_m,
                                      std::uint32_t epoch_seconds)
    : voxel_size_m_(voxel_size_m),
      epoch_seconds_(epoch_seconds),
      impl_(new Impl{}) {}

template <std::size_t Cp1>
EvidentialLayer<Cp1>::~EvidentialLayer() {
    for (auto& kv : impl_->voxel_map) {
        cudaFree(kv.second);
    }
    delete impl_;
}

template <std::size_t Cp1>
void EvidentialLayer<Cp1>::addEvidence(std::uint64_t voxel_key,
                                       const float* evidence_vector,
                                       std::uint32_t timestamp_s,
                                       std::uint16_t pose_idx) {
    auto it = impl_->voxel_map.find(voxel_key);
    if (it == impl_->voxel_map.end()) {
        EvidentialVoxel<Cp1>* dev = nullptr;
        cudaMalloc(&dev, sizeof(EvidentialVoxel<Cp1>));
        EvidentialVoxel<Cp1> init{};
        for (std::size_t k = 0; k < Cp1; ++k) {
            init.alpha[k] = 1.0f;
        }
        init.last_update_ts = timestamp_s;
        init.pose_idx = pose_idx;
        cudaMemcpy(dev, &init, sizeof(EvidentialVoxel<Cp1>),
                   cudaMemcpyHostToDevice);
        impl_->voxel_map[voxel_key] = dev;
        it = impl_->voxel_map.find(voxel_key);
    }
    float* dev_evidence = nullptr;
    cudaMalloc(&dev_evidence, sizeof(float) * Cp1);
    cudaMemcpy(dev_evidence, evidence_vector, sizeof(float) * Cp1,
               cudaMemcpyHostToDevice);

    accumulateEvidenceKernel<Cp1><<<1, Cp1>>>(it->second, dev_evidence);
    cudaDeviceSynchronize();
    cudaFree(dev_evidence);

    impl_->last_update[voxel_key] = timestamp_s;
}

template <std::size_t Cp1>
void EvidentialLayer<Cp1>::addEvidenceBatch(const std::uint64_t* voxel_keys,
                                            const float* evidence_matrix,
                                            std::size_t batch_size,
                                            std::uint32_t timestamp_s,
                                            std::uint16_t pose_idx) {
    // P1: trivial fallback loop. Production: aggregate per-block then launch
    // a single grid-stride kernel.
    for (std::size_t i = 0; i < batch_size; ++i) {
        addEvidence(voxel_keys[i],
                    evidence_matrix + i * Cp1,
                    timestamp_s,
                    pose_idx);
    }
}

template <std::size_t Cp1>
bool EvidentialLayer<Cp1>::getAlpha(std::uint64_t voxel_key,
                                    float* alpha_out) const {
    auto it = impl_->voxel_map.find(voxel_key);
    if (it == impl_->voxel_map.end()) {
        for (std::size_t k = 0; k < Cp1; ++k) alpha_out[k] = 1.0f;
        return false;
    }
    EvidentialVoxel<Cp1> host{};
    cudaMemcpy(&host, it->second, sizeof(EvidentialVoxel<Cp1>),
               cudaMemcpyDeviceToHost);
    std::memcpy(alpha_out, host.alpha, sizeof(float) * Cp1);
    return true;
}

template <std::size_t Cp1>
float EvidentialLayer<Cp1>::getVacuity(std::uint64_t voxel_key) const {
    float buf[Cp1];
    (void)getAlpha(voxel_key, buf);
    float s = 0.0f;
    for (std::size_t k = 0; k < Cp1; ++k) s += buf[k];
    return static_cast<float>(Cp1) / s;
}

template <std::size_t Cp1>
std::size_t EvidentialLayer<Cp1>::decay(std::uint32_t timestamp_s,
                                        float tau_min_s,
                                        float tau_max_s,
                                        float age_hysteresis_s) {
    std::size_t touched = 0;
    for (auto& kv : impl_->voxel_map) {
        const std::uint64_t key = kv.first;
        EvidentialVoxel<Cp1>* dev = kv.second;
        auto it = impl_->last_update.find(key);
        const std::uint32_t last = (it != impl_->last_update.end())
                                       ? it->second
                                       : epoch_seconds_;
        const float age = static_cast<float>(timestamp_s) -
                          static_cast<float>(last);
        if (age <= age_hysteresis_s) continue;

        decayKernel<Cp1><<<1, Cp1>>>(dev, age, tau_min_s, tau_max_s);
        ++touched;
    }
    cudaDeviceSynchronize();
    return touched;
}

template <std::size_t Cp1>
std::size_t EvidentialLayer<Cp1>::numVoxels() const noexcept {
    return impl_->voxel_map.size();
}

template <std::size_t Cp1>
std::size_t EvidentialLayer<Cp1>::numBlocks() const noexcept {
    // Skeleton: report distinct top-level keys / voxels-per-block.
    return (impl_->voxel_map.size() + kVoxelsPerBlock - 1) / kVoxelsPerBlock;
}

// ===========================================================================
// Explicit template instantiation (closed-set 20-D + open-set 15-D).
// ===========================================================================
template class EvidentialLayer<20>;
template class EvidentialLayer<15>;

}  // namespace nvblox_evidential
