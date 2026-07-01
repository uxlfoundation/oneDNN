/*******************************************************************************
* Copyright 2026 openKylin community
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

#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/resampling_utils.hpp"
#include "cpu/rv64/jit_uni_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace resampling_utils;

template <cpu_isa_t isa>
jit_uni_resampling_fwd_t<isa>::jit_uni_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_resampling_fwd_t<isa>::~jit_uni_resampling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_resampling_kernel_t<isa, d_type>(pd()->conf_)));
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const data_t *src0 = src + src_d.off_l(0);
    data_t *dst0 = dst + dst_d.off_l(0);

    const auto &conf = pd()->conf_;
    const dim_t MB = conf.mb, C = conf.c;
    const dim_t ID = conf.id, IH = conf.ih, IW = conf.iw;
    const dim_t OD = conf.od, OH = conf.oh, OW = conf.ow;
    const dim_t ndims = conf.ndims;
    const alg_kind_t alg = conf.alg;
    const bool is_ncsp = conf.tag_kind == jit_resampling_tag_kind_t::ncsp;

    const dim_t in_sp = ID * IH * IW;
    const dim_t out_sp = OD * OH * OW;

    // Channel grouping. ncsp: one strided group over all C. nspc/blocked:
    // channel-contiguous groups of B (= C for nspc, = the inner block for
    // blocked), iterated over Cb outer blocks. sp_scale is the element step
    // between adjacent spatial points (x64's inner_stride).
    const dim_t B = is_ncsp ? C : (conf.block ? conf.block : C);
    const dim_t Cb = is_ncsp ? 1 : utils::div_up(C, B);
    const dim_t src_sp_scale = is_ncsp ? 1 : B;
    const dim_t dst_sp_scale = is_ncsp ? 1 : B;
    const dim_t src_vec_byte_stride
            = (is_ncsp ? in_sp : (dim_t)1) * conf.dt_size;
    const dim_t dst_vec_byte_stride
            = (is_ncsp ? out_sp : (dim_t)1) * conf.dt_size;
    const dim_t src_mb_stride = is_ncsp ? C * in_sp : Cb * in_sp * B;
    const dim_t dst_mb_stride = is_ncsp ? C * out_sp : Cb * out_sp * B;
    const dim_t src_cb_stride = is_ncsp ? 0 : in_sp * B;
    const dim_t dst_cb_stride = is_ncsp ? 0 : out_sp * B;

    // Fused binary (f32 only). The injector runs in indirect mode: it reads the
    // rhs base from a per-binary ORIGIN pointer array and adds the shared byte
    // offset the driver hands it per output point (see jit_resampling_args_t).
    enum { BC_NONE, BC_SCALAR, BC_PER_OC, BC_FULL } bcast = BC_NONE;
    std::vector<const void *> po_rhs;
    if (conf.fuse_binary) {
        int bin_idx = 0;
        for (int i = 0; i < conf.post_ops.len(); i++) {
            if (!conf.post_ops.entry_[i].is_binary()) continue;
            bin_idx = i;
            const memory_desc_wrapper s1_d(
                    conf.post_ops.entry_[i].binary.src1_desc);
            const auto *base = static_cast<const char *>(ctx.host_ptr(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
            po_rhs.push_back(base + s1_d.off_l(0) * sizeof(float));
        }
        const memory_desc_wrapper s1(
                conf.post_ops.entry_[bin_idx].binary.src1_desc);
        if (s1.nelems(true) == 1)
            bcast = BC_SCALAR;
        else if (s1.nelems() == C)
            bcast = BC_PER_OC;
        else
            bcast = BC_FULL;
    }
    const void *const po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    auto sp_index = [=](dim_t id, dim_t ih, dim_t iw) {
        return (id * IH + ih) * IW + iw;
    };

    parallel_nd(MB, Cb, OD, OH, OW,
            [&](dim_t mb, dim_t cb, dim_t od, dim_t oh, dim_t ow) {
        jit_resampling_args_t args = {};
        // Valid channels in this group (last blocked group may be partial).
        args.channels = is_ncsp ? C : ((C - cb * B) < B ? (C - cb * B) : B);
        args.src_vec_byte_stride = src_vec_byte_stride;
        args.dst_vec_byte_stride = dst_vec_byte_stride;

        const dim_t out_sp_off = (od * OH + oh) * OW + ow;
        data_t *p_dst = dst0 + mb * dst_mb_stride + cb * dst_cb_stride
                + out_sp_off * dst_sp_scale;
        args.dst = p_dst;

        const data_t *src_base = src0 + mb * src_mb_stride + cb * src_cb_stride;

        if (alg == alg_kind::resampling_nearest) {
            const dim_t id = ndims >= 5 ? nearest_idx(od, OD, ID) : 0;
            const dim_t ih = ndims >= 4 ? nearest_idx(oh, OH, IH) : 0;
            const dim_t iw = nearest_idx(ow, OW, IW);
            args.src[0] = src_base + sp_index(id, ih, iw) * src_sp_scale;
        } else {
            dim_t d_idx[2] = {0, 0}, h_idx[2] = {0, 0}, w_idx[2] = {0, 0};
            float d_w[2] = {1.f, 0.f}, h_w[2] = {1.f, 0.f}, w_w[2] = {1.f, 0.f};
            int dn = 1, hn = 1, wn = 2;

            const linear_coeffs_t wc(ow, OW, IW);
            w_idx[0] = wc.idx[0];
            w_idx[1] = wc.idx[1];
            w_w[0] = wc.wei[0];
            w_w[1] = wc.wei[1];
            if (ndims >= 4) {
                const linear_coeffs_t hc(oh, OH, IH);
                h_idx[0] = hc.idx[0];
                h_idx[1] = hc.idx[1];
                h_w[0] = hc.wei[0];
                h_w[1] = hc.wei[1];
                hn = 2;
            }
            if (ndims >= 5) {
                const linear_coeffs_t dc(od, OD, ID);
                d_idx[0] = dc.idx[0];
                d_idx[1] = dc.idx[1];
                d_w[0] = dc.wei[0];
                d_w[1] = dc.wei[1];
                dn = 2;
            }

            int c = 0;
            for (int i = 0; i < dn; i++)
                for (int j = 0; j < hn; j++)
                    for (int k = 0; k < wn; k++) {
                        args.src[c] = src_base
                                + sp_index(d_idx[i], h_idx[j], w_idx[k])
                                        * src_sp_scale;
                        args.weights[c] = d_w[i] * h_w[j] * w_w[k];
                        c++;
                    }
        }

        if (conf.fuse_binary) {
            args.post_op_rhs = po_rhs_arr;
            switch (bcast) {
                case BC_SCALAR: args.post_op_off0 = 0; break;
                case BC_PER_OC:
                    args.post_op_off0 = cb * B * (dim_t)sizeof(float);
                    break;
                case BC_FULL:
                    args.post_op_off0 = (p_dst - dst0) * (dim_t)sizeof(float);
                    break;
                default: break;
            }
        }

        (*kernel_)(&args);

        // Blocked layout zero-pads the channel dimension: when C is not a
        // multiple of the block, the last block's padded channels
        // [valid, block) must be zero in the output. The kernel wrote only the
        // valid channels (contiguous from p_dst), so zero the padding tail here.
        // nspc/ncsp have no channel padding (block == 0). Padding must stay zero
        // regardless of post-ops (they apply to logical elements only).
        if (conf.block && cb == Cb - 1 && args.channels < B) {
            data_t *pd = static_cast<data_t *>(args.dst);
            for (dim_t c = args.channels; c < B; c++)
                pd[c] = data_t(0.f);
        }
    });

    return status::success;
}

template struct jit_uni_resampling_fwd_t<v>;
template struct jit_uni_resampling_fwd_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
