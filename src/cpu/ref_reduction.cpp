/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <math.h>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/ref_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <typename T>
void init_acc(T &acc, alg_kind_t alg, data_type_t src_dt) {
    using namespace alg_kind;
    using namespace nstl;

    switch (alg) {
        case reduction_max: acc = types::lowest_value<T>(src_dt); break;
        case reduction_min: acc = types::max_value<T>(src_dt); break;
        case reduction_mean:
        case reduction_sum: acc = T(0); break;
        case reduction_mul: acc = T(1); break;
        case reduction_norm_lp_max:
        case reduction_norm_lp_sum:
        case reduction_norm_lp_power_p_max:
        case reduction_norm_lp_power_p_sum: acc = T(0); break;
        default: assert(!"unknown alg");
    }
}

template <typename T>
void accumulate(T &acc, const T &src, alg_kind_t alg, float p) {
    using namespace alg_kind;

    switch (alg) {
        case reduction_max: acc = nstl::max(acc, src); break;
        case reduction_min: acc = nstl::min(acc, src); break;
        case reduction_mean:
        case reduction_sum: acc += src; break;
        case reduction_mul: acc *= src; break;
        case reduction_norm_lp_max:
        case reduction_norm_lp_sum:
        case reduction_norm_lp_power_p_max:
        case reduction_norm_lp_power_p_sum:
            acc += powf(nstl::abs(src), p);
            break;
        default: assert(!"unknown alg");
    }
}

void finalize(float &acc_f32, alg_kind_t alg, float p, float eps, dim_t n) {
    using namespace alg_kind;

    switch (alg) {
        case reduction_mean: acc_f32 /= n; break;
        case reduction_norm_lp_max:
            acc_f32 = nstl::max(acc_f32, eps);
            acc_f32 = powf(acc_f32, 1.0f / p);
            break;
        case reduction_norm_lp_sum:
            acc_f32 += eps;
            acc_f32 = powf(acc_f32, 1.0f / p);
            break;
        case reduction_norm_lp_power_p_max:
            acc_f32 = nstl::max(acc_f32, eps);
            break;
        case reduction_norm_lp_power_p_sum: acc_f32 += eps; break;
        default: break;
    }
}

template <typename T>
T ker(dim_t l_offset, alg_kind_t alg, float p, dim_t reduce_size,
        const dims_t idle_pos, const dims_t reduce_dims,
        const memory_desc_t *src_md, const void *src);

template <>
int ker(dim_t l_offset, alg_kind_t alg, float p, dim_t reduce_size,
        const dims_t idle_pos, const dims_t reduce_dims,
        const memory_desc_t *src_md, const void *src) {
    const memory_desc_wrapper src_mdw(src_md);
    const int ndims = src_mdw.ndims();
    const dim_t src_idle_off = src_mdw.off_v(idle_pos);
    dims_t reduce_pos;

    int acc {0};
    init_acc(acc, alg, src_mdw.data_type());
    for (dim_t r = 0; r < reduce_size; ++r) {
        utils::l_dims_by_l_offset(reduce_pos, r, reduce_dims, ndims);
        const dim_t src_reduce_off = src_mdw.off_v(reduce_pos);
        const dim_t src_off = src_idle_off + src_reduce_off;
        const int s = io::load_int_value(src_mdw.data_type(), src, src_off);
        accumulate(acc, s, alg, p);
    }

    return acc;
}

template <>
float ker(dim_t l_offset, alg_kind_t alg, float p, dim_t reduce_size,
        const dims_t idle_pos, const dims_t reduce_dims,
        const memory_desc_t *src_md, const void *src) {
    const memory_desc_wrapper src_mdw(src_md);
    const int ndims = src_mdw.ndims();
    const dim_t src_idle_off = src_mdw.off_v(idle_pos);
    dims_t reduce_pos;

    float acc {0};
    init_acc(acc, alg, src_mdw.data_type());
    for (dim_t r = 0; r < reduce_size; ++r) {
        utils::l_dims_by_l_offset(reduce_pos, r, reduce_dims, ndims);
        const dim_t src_reduce_off = src_mdw.off_v(reduce_pos);
        const dim_t src_off = src_idle_off + src_reduce_off;
        const float s = io::load_float_value(src_mdw.data_type(), src, src_off);
        accumulate(acc, s, alg, p);
    }

    return acc;
}

status_t ref_reduction_t::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_mdw(pd()->src_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());

    const int ndims = src_mdw.ndims();
    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();
    const auto acc_type = types::default_accum_data_type(
            src_mdw.data_type(), dst_mdw.data_type());

    const auto alg = pd()->desc()->alg_kind;
    const auto p = pd()->desc()->p;
    const auto eps = pd()->desc()->eps;

    dims_t reduce_dims;
    dim_t reduce_size {1}, idle_size = dst_mdw.nelems();

    for (int d = 0; d < ndims; ++d) {
        reduce_dims[d] = dim_t {1};
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        if (is_reduction_dim) {
            reduce_dims[d] = src_dims[d];
            reduce_size *= reduce_dims[d];
        }
    }

    parallel_nd(idle_size, [= COMPAT_THIS_CAPTURE](dim_t l_offset) {
        dims_t idle_pos;
        utils::l_dims_by_l_offset(idle_pos, l_offset, dst_mdw.dims(), ndims);

        float acc_f32 = 0.f;
        if (types::is_integral_dt(acc_type)) {
            int acc = ker<int>(l_offset, alg, p, reduce_size, idle_pos,
                    reduce_dims, pd()->src_md(), src);
            acc_f32 = static_cast<float>(acc);
        } else {
            acc_f32 = ker<float>(l_offset, alg, p, reduce_size, idle_pos,
                    reduce_dims, pd()->src_md(), src);
        }

        finalize(acc_f32, alg, p, eps, reduce_size);

        const dim_t dst_off = dst_mdw.off_v(idle_pos);
        ref_post_ops_t::args_t args;
        args.dst_val = io::load_float_value(dst_mdw.data_type(), dst, dst_off);
        args.ctx = &ctx;
        args.l_offset = l_offset;
        args.dst_md = pd()->dst_md();
        ref_post_ops->execute(acc_f32, args);

        io::store_float_value(dst_mdw.data_type(), acc_f32, dst, dst_off);
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
