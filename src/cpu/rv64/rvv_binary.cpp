/******************************************************************************
* Copyright 2025
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
******************************************************************************/

#include <assert.h>
#include <memory>
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/rv64/rvv_binary_kernels.hpp"

#include "cpu/rv64/rvv_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Data type dispatch for RVV binary
static inline void compute_binary_rvv(const alg_kind_t alg, const void *x,
        const void *y, void *dst, const int8_t *c, const dim_t len,
        const data_type_t dt) {
    switch (dt) {
        case data_type::f32:
            rvv_binary_apply_f32(alg, x, y, dst, c, len, dt);
            break;
        case data_type::s32:
            rvv_binary_apply_s32(alg, x, y, dst, c, len, dt);
            break;
        case data_type::s8:
            rvv_binary_apply_s8(alg, x, y, dst, c, len, dt);
            break;
        case data_type::u8:
            rvv_binary_apply_u8(alg, x, y, dst, c, len, dt);
            break;
        default: assert(!"Unsupported data type for RVV binary");
    }
}

status_t rvv_binary_t::execute(const exec_ctx_t &ctx) const {
    const data_type_t dt = pd()->dst_md()->data_type;
    const void *src0 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_0);
    const void *src1 = CTX_IN_MEM(const void *, DNNL_ARG_SRC_1);
    // src2 is optional (only for ternary ops like select); treat as s8 mask
    const int8_t *src2_s8 = nullptr;
    if (pd()->is_ternary_op()) {
        src2_s8 = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC_2);
    }
    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));
    const memory_desc_wrapper src2_d(pd()->src_md(2));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto nelems = dst_d.nelems(true);

    const auto alg_kind = pd()->desc()->alg_kind;

    const auto off0 = src0_d.offset0();
    const auto off1 = src1_d.offset0();
    const auto offd = dst_d.offset0();
    const auto off2 = pd()->is_ternary_op() ? src2_d.offset0() : 0;

    const auto *x_base = static_cast<const char *>(src0)
            + off0 * types::data_type_size(dt);
    const auto *y_base = static_cast<const char *>(src1)
            + off1 * types::data_type_size(dt);
    auto *d_base = static_cast<char *>(dst) + offd * types::data_type_size(dt);
    const int8_t *c_base = pd()->is_ternary_op() ? (src2_s8 + off2) : nullptr;

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start = 0, end = 0;
        balance211(nelems, nthr, ithr, start, end);
        if (start == end) return;

        const void *thr_x = static_cast<const void *>(
                x_base + start * types::data_type_size(dt));
        const void *thr_y = static_cast<const void *>(
                y_base + start * types::data_type_size(dt));
        void *thr_d = static_cast<void *>(
                d_base + start * types::data_type_size(dt));
        const int8_t *thr_c = c_base ? (c_base + start) : nullptr;
        const dim_t len = end - start;

        compute_binary_rvv(alg_kind, thr_x, thr_y, thr_d, thr_c, len, dt);
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
