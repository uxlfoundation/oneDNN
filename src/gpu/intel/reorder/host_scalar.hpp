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

#ifndef GPU_INTEL_REORDER_HOST_SCALAR_HPP
#define GPU_INTEL_REORDER_HOST_SCALAR_HPP

#include <cstring>

#include "common/host_scalar_memory_storage.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/reorder/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

// Reorder implementation for copying a host-side scalar to Intel GPU memory.
// Supports only exact copy (same data type, no scales/zero-points).
struct host_scalar_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public reorder::pd_t {
        using reorder::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:host_scalar", host_scalar_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            memory_desc_wrapper src_mdw(src_md());
            memory_desc_wrapper dst_mdw(dst_md());

            // Source must be a host scalar.
            VDISPATCH_REORDER(
                    src_mdw.is_host_scalar_desc(), VERBOSE_BAD_ENGINE_KIND);

            // Only default attributes (no scales, zero-points, or post-ops).
            VDISPATCH_REORDER(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            // Same data type required.
            VDISPATCH_REORDER(src_mdw.data_type() == dst_mdw.data_type(),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Destination must be a single element.
            VDISPATCH_REORDER(dst_mdw.nelems() == 1, VERBOSE_INCONSISTENT_MDS,
                    "src", "dst");

            return status::success;
        }

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(impl::engine_t *engine) override { return status::success; }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto *mem_src = ctx.input(DNNL_ARG_FROM);
        if (!mem_src) return status::invalid_arguments;

        auto *src_storage
                = utils::downcast<const host_scalar_memory_storage_t *>(
                        mem_src->memory_storage());

        memory_desc_wrapper dst_mdw(pd()->dst_md());
        const size_t size = dst_mdw.data_type_size();

        // Read scalar value from host storage.
        alignas(host_scalar_memory_storage_t::max_scalar_align) char
                scalar_buf[host_scalar_memory_storage_t::max_scalar_size];
        CHECK(src_storage->get_scalar_value(scalar_buf, size));

        // Map destination GPU memory, copy the scalar, and unmap.
        auto &dst_storage = CTX_OUT_STORAGE(DNNL_ARG_TO);
        void *dst_mapped = nullptr;
        CHECK(dst_storage.map_data(&dst_mapped, ctx.stream(), size));
        if (!dst_mapped) return status::runtime_error;

        std::memcpy(dst_mapped, scalar_buf, size);

        CHECK(dst_storage.unmap_data(dst_mapped, ctx.stream()));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
