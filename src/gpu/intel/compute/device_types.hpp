/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_DEVICE_TYPES_HPP
#define GPU_INTEL_COMPUTE_DEVICE_TYPES_HPP
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

enum class gpu_arch_t { unknown, xe_lp, xe_hp, xe_hpg, xe_hpc, xe2, xe3 };

static inline std::string to_string(gpu_arch_t arch) {
#define CASE(_case) \
    if (arch == gpu_arch_t::_case) return STRINGIFY(_case)
    CASE(xe_lp);
    CASE(xe_hp);
    CASE(xe_hpg);
    CASE(xe_hpc);
    CASE(xe2);
    CASE(xe3);
    return "unknown";
#undef CASE
}

static inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(xe_lp);
    CASE(xe_hp);
    CASE(xe_hpg);
    CASE(xe_hpc);
    CASE(xe2);
    CASE(xe3);
    return gpu_arch_t::unknown;
#undef CASE
}

enum class device_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    khr_fp16 = 1ull << 0,
    khr_fp64 = 1ull << 1,
    // OpenCL atomics
    khr_global_int32_base_atomics     = 1ull << 2,
    khr_global_int32_extended_atomics = 1ull << 3,
    khr_int64_base_atomics            = 1ull << 4,
    khr_int64_extended_atomics        = 1ull << 5,
    khr_local_int32_base_atomics      = 1ull << 6,
    khr_local_int32_extended_atomics  = 1ull << 7,
    ext_float_atomics                 = 1ull << 8,
    // Intel specific XeLP+
    intel_subgroups              = 1ull << 16,
    intel_required_subgroup_size = 1ull << 17,
    intel_subgroups_char         = 1ull << 18,
    intel_subgroups_short        = 1ull << 19,
    intel_subgroups_long         = 1ull << 20,
    intel_subgroup_local_block_io = 1ull << 21,
    intel_dot_accumulate          = 1ull << 22,
    // Intel specific XeHP+
    intel_global_float_atomics                      = 1ull << 23,
    intel_subgroup_matrix_multiply_accumulate       = 1ull << 24,
    intel_subgroup_split_matrix_multiply_accumulate = 1ull << 25,
    intel_variable_eu_thread_count                  = 1ull << 26,
    intel_unified_shared_memory                     = 1ull << 27,
    // Future extensions
    future_bf16_cvt                                 = 1ull << 31,
    last
    // clang-format on
};

static inline const char *ext2cl_str(device_ext_t ext) {
#define CASE(x) \
    case device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(khr_fp16)
        CASE(khr_fp64)

        CASE(khr_global_int32_base_atomics)
        CASE(khr_global_int32_extended_atomics)
        CASE(khr_int64_base_atomics)
        CASE(khr_int64_extended_atomics)
        CASE(khr_local_int32_base_atomics)
        CASE(khr_local_int32_extended_atomics)
        CASE(ext_float_atomics)

        CASE(intel_subgroups)
        CASE(intel_required_subgroup_size)
        CASE(intel_subgroups_char)
        CASE(intel_subgroups_short)
        CASE(intel_subgroups_long)

        CASE(intel_subgroup_local_block_io)
        CASE(intel_dot_accumulate)

        CASE(intel_global_float_atomics)
        CASE(intel_subgroup_matrix_multiply_accumulate)
        CASE(intel_subgroup_split_matrix_multiply_accumulate)
        CASE(intel_variable_eu_thread_count)
        CASE(intel_unified_shared_memory)
        CASE(future_bf16_cvt)
        default: return nullptr;
    }
#undef CASE
}

enum class native_ext_t : uint64_t {
    // clang-format off
    // OpenCL data types
    fp32_atomic_add = 1ull << 0,
    fp32_atomic_min_max = 1ull << 1,
    fp32_atomic_load_store = 1ull << 2,
    fp16_atomic_add = 1ull << 3,
    fp16_atomic_min_max = 1ull << 4,
    fp16_atomic_load_store = 1ull << 5,
    fp64_atomic_add = 1ull << 6,
    fp64_atomic_min_max = 1ull << 7,
    fp64_atomic_load_store = 1ull << 8,
    last
};


} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
