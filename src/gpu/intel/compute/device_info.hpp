/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_COMPUTE_DEVICE_INFO_HPP
#define GPU_INTEL_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"
#include "gpu/intel/compute/device_types.hpp"

#include "ngen_core.hpp"
#include "xpu/utils.hpp"

#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// Needed workaround for future HW extensions
uint64_t get_future_extensions(
        compute::gpu_arch_t gpu_arch, bool mayiuse_systolic);

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    status_t init(impl::engine_t *engine,
            const std::vector<uint8_t> &cache_blob = {}) {
        if (!cache_blob.empty()) {
            CHECK(init_from_cache_blob(cache_blob));
            return init_serialized_device_info(cache_blob);
        }

        CHECK(init_device_name(engine));
        CHECK(init_arch(engine));
        CHECK(init_runtime_version(engine));
        CHECK(init_extensions(engine));
        CHECK(init_attributes(engine));
        fixup_l3_cache_size();

        CHECK(init_attributes_common(engine));

        if (dnnl_version()->gpu_runtime == DNNL_RUNTIME_OCL) {
            CHECK(init_serialized_device_info());
        }

        return status::success;
    }

    std::string get_cl_ext_options() const;

    bool has(device_ext_t ext) const { return extensions_ & (uint64_t)ext; }
    bool has_native(native_ext_t ext) const {
        return native_extensions_ & (uint64_t)ext;
    }
    gpu_arch_t gpu_arch() const { return gpu_arch_; }
    const ngen::Product &gpu_product() const { return gpu_product_; }
    const ngen::ProductFamily &gpu_product_family() const {
        return gpu_product_.family;
    }
    int stepping_id() const { return gpu_product_.stepping; }
    uint64_t native_extensions() const { return native_extensions_; }
    bool is_integrated() const;
    uint32_t ip_version() const { return ip_version_; }
    int max_eus_per_wg() const { return max_eus_per_wg_; }
    static int max_eus_per_wg(gpu_arch_t gpu_arch);

    static int max_exec_size(gpu_arch_t gpu_arch);

    int max_exec_size() const { return max_exec_size(gpu_arch()); }
    int max_subgroup_size(data_type_t type = data_type::undef) const;
    static int max_subgroup_size(gpu_arch_t gpu_arch);
    static int grf_size(gpu_arch_t gpu_arch);
    int grf_size() const { return grf_size(gpu_arch_); };
    int min_subgroup_size() const;
    size_t max_wg_size(bool large_grf_mode, size_t subgroup_size = 0) const;
    int eu_count() const { return eu_count_; }
    int hw_threads() const { return hw_threads_[0]; }
    int hw_threads(bool large_grf_mode) const {
        return hw_threads_[large_grf_mode ? 1 : 0];
    }
    static int threads_per_eu(gpu_arch_t gpu_arch, bool large_grf_mode = false);
    static int max_slm_size(gpu_arch_t gpu_arch);
    static int max_slm_size_per_tg(gpu_arch_t gpu_arch);
    static int max_slm_size_per_tg(
            gpu_arch_t gpu_arch, int tg_size, bool large_grf_mode = false);
    size_t l3_cache_size() const { return l3_cache_size_; }
    size_t icache_size() const;
    size_t max_kernel_param_size() const { return max_kernel_param_size_; }
    uint32_t device_address_bits() const { return device_address_bits_; }

    const xpu::runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }

    bool mayiuse_ngen_kernels() const { return mayiuse_ngen_kernels_; }

    bool mayiuse_systolic() const { return mayiuse_systolic_; }
    bool mayiuse_large_grf_mode() const { return mayiuse_systolic_; }

    bool mayiuse_non_uniform_work_groups() const {
        return mayiuse_non_uniform_work_groups_;
    }

    /// Returns true if the engine can directly access pointers from system allocators
    bool mayiuse_system_memory_allocators() const {
        return mayiuse_system_memory_allocators_;
    }

    bool mayiuse_sub_group(int size) const;

    bool mayiuse_float_atomic_add(data_type_t type) const;

    bool has_native(data_type_t type) const;

    const std::vector<uint8_t> &get_cache_blob() const {
        return serialized_device_info_.get_data();
    }

    status_t get_cache_blob_size(size_t *size) const {
        (*size) = serialized_device_info_.get_data().size();
        return status::success;
    }

    status_t get_cache_blob(size_t size, uint8_t *cache_blob) const {
        const auto &cb = serialized_device_info_.get_data();
        if (size != cb.size()) return status::invalid_arguments;
        std::memcpy(cache_blob, cb.data(), size);
        return status::success;
    }

protected:
    virtual status_t init_device_name(impl::engine_t *engine) = 0;
    virtual status_t init_arch(impl::engine_t *engine) = 0;
    virtual status_t init_runtime_version(impl::engine_t *engine) = 0;
    virtual status_t init_extensions(impl::engine_t *engine) = 0;
    virtual status_t init_attributes(impl::engine_t *engine) = 0;

    compute::gpu_arch_t gpu_arch_ = compute::gpu_arch_t::unknown;
    ngen::Product gpu_product_ = {};
    uint32_t ip_version_ = 0;
    bool mayiuse_systolic_ = false;
    bool mayiuse_ngen_kernels_ = false;
    bool mayiuse_system_memory_allocators_ = false;

    std::string name_;
    xpu::runtime_version_t runtime_version_;

    // total number of hardware threads:
    // [0] - default mode
    // [1] - large GRF mode
    int32_t hw_threads_[2] = {0, 0};
    int32_t eu_count_ = 0;
    int32_t max_eus_per_wg_ = 0;
    int32_t max_subgroup_size_ = 16;
    int max_exec_size_ = 0;
    size_t max_wg_size_ = 0;
    size_t l3_cache_size_ = 0;
    size_t max_kernel_param_size_ = 1024;
    uint32_t device_address_bits_ = 64;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecture.
    uint64_t extensions_ = 0;
    // native extensions, may differ from support reported by runtime.
    uint64_t native_extensions_ = 0;

private:
    status_t init_attributes_common(impl::engine_t *engine);
    status_t init_serialized_device_info(
            const std::vector<uint8_t> &cache_blob = {});
    status_t init_from_cache_blob(const std::vector<uint8_t> &cache_blob);
    void fixup_l3_cache_size();

    bool mayiuse_non_uniform_work_groups_ = false;

    serialization_stream_t serialized_device_info_;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
