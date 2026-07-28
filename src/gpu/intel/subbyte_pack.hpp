/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GPU_INTEL_SUBBYTE_PACK_HPP
#define GPU_INTEL_SUBBYTE_PACK_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/serialization.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/types_interop.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct primitive_t;
class engine_t;

// The subbyte_pack kernel takes a temporary buffer holding one sub-byte element
// per byte and packs into the destination.
struct subbyte_pack_desc_t {
    status_t init(const memory_desc_t &dst_md);
    explicit operator bool() const { return bool(conf_); }
    size_t span() const { return span_; }

    struct conf_t : public trivially_serializable_t<conf_t> {
        status_t create_generator(
                const engine_t &engine, compute::kernel_bundle_t &bundle) const;
        const std::vector<const char *> &get_kernel_names() const;
        explicit operator bool() const { return bits != 0; }

        int bits = 0;
        int ndims = 0;
        bool use_int32_offset = false;
        bool require_stateless_addressing = false;
        uint8_t pad[2] = {};
    };
    const conf_t &conf() const { return conf_; }

    compute::int64x6_t dims() const { return dims_; }
    compute::int64x6_t strides() const { return strides_; }
    compute::nd_range_t nd_range() const {
        return compute::nd_range_t(compute::range_t(into<size_t>(gws_[0]),
                into<size_t>(gws_[1]), into<size_t>(gws_[2])));
    }

private:
    static constexpr int max_ndims = 6;

    conf_t conf_;

    // Destination layout, ordered from inner-most to outer-most axis.
    compute::int64x6_t dims_ = {};
    compute::int64x6_t strides_ = {};
    size_t span_ = 0;
    dim_t gws_[3] = {1, 1, 1};
};

struct subbyte_pack_t {
    status_t create(const subbyte_pack_desc_t &desc, primitive_t &primitive,
            impl::engine_t *engine);
    status_t operator()(const exec_ctx_t &ctx, const memory_storage_t &src,
            const memory_storage_t &dst) const;
    explicit operator bool() const { return bool(kernel_); }

private:
    compute::kernel_t kernel_;
    compute::nd_range_t nd_range_;
    compute::int64x6_t dims_ = {};
    compute::int64x6_t strides_ = {};
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
