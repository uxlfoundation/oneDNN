/*******************************************************************************
* Copyright 2016-2025 Intel Corporation
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

#include <atomic>
#include <riscv_vector.h>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/rv64/rvv_gemm_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

namespace {
struct im_pos_t {
    im_pos_t() : n {0}, g {0}, od {0}, sp {0}, ic {0}, oc {0} {}
    dim_t n, g, od, sp, ic, oc;
    bool do_im2col(const im_pos_t &prev) const {
        return true
                && (n != prev.n || g != prev.g || od != prev.od || sp != prev.sp
                        || ic != prev.ic);
    }
};
} // namespace

status_t riscv_gemm_convolution_fwd_t::execute_forward_nspc(
        const exec_ctx_t &ctx) const {
    auto src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst_base = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const conv_gemm_conf_t &jcp = pd()->jcp_;
    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_forward_thr_nspc(ctx, ithr, nthr, src_base,
                wei_base, bia_base, dst_base, scratchpad);
        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

status_t riscv_gemm_convolution_fwd_t::execute_forward_thr_nspc(
        const exec_ctx_t &ctx, const int ithr, const int nthr,
        const data_t *src_base, const data_t *wei_base, const data_t *bia_base,
        data_t *dst_base, const memory_tracking::grantor_t &scratchpad) const {
    const conv_gemm_conf_t &jcp = pd()->jcp_;

    // Src Format: mb-spatial-groups-input_channels
    const dim_t src_mb_stride = jcp.id * jcp.ih * jcp.iw * jcp.ngroups * jcp.ic;
    const dim_t src_g_stride = jcp.ic;
    // Wei Format: spatial-input_channels-groups-output_channels
    const dim_t wei_g_stride = pd()->with_groups() ? jcp.oc : 0;

    // Dst Format: mb-spatial-groups-output_channels
    const dim_t dst_mb_stride = jcp.od * jcp.oh * jcp.ow * jcp.ngroups * jcp.oc;
    const dim_t dst_g_stride = jcp.oc;
    const dim_t dst_os_stride = jcp.ngroups * jcp.oc;

    data_t *__restrict col = scratchpad.get<data_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    data_t *__restrict imtr = scratchpad.get<data_t>(key_conv_gemm_imtr)
            + (ptrdiff_t)ithr * jcp.is * jcp.ic;

    dim_t g {0}, n {0}, ohb {0}, owb {0};
    dim_t start = 0, end = 0;
    const bool is_problem_3d = pd()->ndims() == 5;

    assert(IMPLICATION(is_problem_3d,
            jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow
                    && jcp.ic_block == jcp.ic));
    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    const dim_t nb_oh = div_up(jcp.oh, jcp.oh_block);
    const dim_t nb_ow = div_up(jcp.ow, jcp.ow_block);
    // threads share work across mini-batch, groups, and blocked width/height
    const dim_t work_amount = jcp.mb * jcp.ngroups * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);

    if (jcp.im2col_sz && is_problem_3d) {
        // jit_gemm_convolution_utils::im2col_dt_3d() requires external
        // data initialization by zeroes

        ptrdiff_t i = 0;
        while (i < jcp.im2col_sz) {
            size_t vl = __riscv_vsetvl_e32m1(jcp.im2col_sz - i);
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            __riscv_vse32_v_f32m1(col + i, v_zero, vl);
            i += vl;
        }
    }

    for (dim_t iwork = start; iwork < end; ++iwork) {
        dim_t oh = ohb * jcp.oh_block;
        dim_t ow = owb * jcp.ow_block;
        const data_t *__restrict src
                = src_base + n * src_mb_stride + g * src_g_stride;
        const data_t *__restrict wei = wei_base + g * wei_g_stride;

        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);
        if (jcp.im2col_sz && is_problem_3d) {
            jit_gemm_convolution_utils::transpose_dt(jcp, src, imtr);
        }

        for (int od = 0; od < jcp.od; od++) {
            data_t *__restrict dst = dst_base + n * dst_mb_stride
                    + g * dst_g_stride
                    + ((od * jcp.oh + oh) * jcp.ow + ow) * dst_os_stride;
            if (jcp.im2col_sz) {
                if (is_problem_3d)
                    jit_gemm_convolution_utils::im2col_dt_3d<data_t, data_t>(
                            jcp, imtr, col, od);
                else
                    jit_gemm_convolution_utils::im2col_dt<data_t, data_t>(
                            jcp, src, imtr, col, oh, h_step, ow, w_step);
            }

            const dim_t M = jcp.oc;
            const dim_t K = jcp.ks * jcp.ic;
            const dim_t N = h_step * w_step;
            const dim_t LDA = M * jcp.ngroups;
            const dim_t LDB = jcp.im2col_sz ? N : K * jcp.ngroups;
            const dim_t LDC = M * jcp.ngroups;
            const char *BT = jcp.im2col_sz ? "T" : "N";
            const data_t onef = 1.f;
            const float beta = this->beta_;
            const data_t *__restrict src_od
                    = src + od * jcp.oh * jcp.ow * jcp.ngroups * jcp.ic;
            status_t st = extended_sgemm("N", BT, &M, &N, &K, &onef, wei, &LDA,
                    jcp.im2col_sz ? col : (data_t *)src_od, &LDB, &beta, dst,
                    &LDC);
            if (st != status::success) return st;

            if (jcp.with_bias || jcp.with_eltwise || jcp.with_binary) {
                parallel(0, [&](int ithr, int nthr) {
                    dim_t start, end;
                    balance211(N * jcp.oc, nthr, ithr, start, end);

                    const size_t first_oc = start % jcp.oc;
                    const size_t last_oc = (end - 1) % jcp.oc;
                    const size_t first_os = start / jcp.oc;
                    const size_t last_os = (end - 1) / jcp.oc;

                    for (size_t os = first_os; os <= last_os; ++os) {
                        const size_t start_oc = (os == first_os) ? first_oc : 0;
                        const size_t end_oc
                                = (os == last_os) ? last_oc : jcp.oc - 1;

                        const data_t *__restrict bia_arr
                                = bia_base ? bia_base + g * jcp.oc : nullptr;
                        data_t *__restrict dst_arr = dst + os * dst_os_stride;

                        if (jcp.with_bias) {
                            size_t n_elems = end_oc - start_oc + 1;
                            if (n_elems > 0) {
                                size_t oc = 0;
                                const data_t *b_ptr = bia_arr + start_oc;
                                data_t *d_ptr = dst_arr + start_oc;

                                while (oc < n_elems) {
                                    size_t vl = __riscv_vsetvl_e32m1(
                                            n_elems - oc);
                                    vfloat32m1_t v_dst = __riscv_vle32_v_f32m1(
                                            d_ptr + oc, vl);
                                    vfloat32m1_t v_bias = __riscv_vle32_v_f32m1(
                                            b_ptr + oc, vl);
                                    v_dst = __riscv_vfadd_vv_f32m1(
                                            v_dst, v_bias, vl);
                                    __riscv_vse32_v_f32m1(
                                            d_ptr + oc, v_dst, vl);
                                    oc += vl;
                                }
                            }
                        }

                        if (jcp.with_eltwise || jcp.with_binary) {
                            bool fast_relu_done = false;
                            if (jcp.with_eltwise && jcp.post_ops.len() == 1) {
                                // fast branch for ReLU case
                                const auto &eltwise
                                        = jcp.post_ops.entry_.back().eltwise;

                                if (eltwise.alg == alg_kind::eltwise_relu) {
                                    const auto alpha = eltwise.alpha;
                                    const auto scale = eltwise.scale;

                                    size_t n_elems = end_oc - start_oc + 1;
                                    if (n_elems > 0) {
                                        size_t oc = 0;
                                        data_t *d_ptr = dst_arr + start_oc;

                                        while (oc < n_elems) {
                                            size_t vl = __riscv_vsetvl_e32m1(
                                                    n_elems - oc);
                                            vfloat32m1_t v_dst
                                                    = __riscv_vle32_v_f32m1(
                                                            d_ptr + oc, vl);

                                            vbool32_t mask
                                                    = __riscv_vmflt_vf_f32m1_b32(
                                                            v_dst, 0.0f, vl);
                                            v_dst = __riscv_vfmul_vf_f32m1_m(
                                                    mask, v_dst, alpha, vl);

                                            v_dst = __riscv_vfmul_vf_f32m1(
                                                    v_dst, scale, vl);

                                            __riscv_vse32_v_f32m1(
                                                    d_ptr + oc, v_dst, vl);
                                            oc += vl;
                                        }
                                    }

                                    fast_relu_done = true;
                                }
                            }
                            if (!fast_relu_done) {
                                ref_post_ops_t::args_t args;
                                args.ctx = &ctx;
                                args.dst_md = pd()->dst_md();

                                for (size_t oc = start_oc; oc <= end_oc; oc++) {
                                    // jcp.od is not part of jcp.os, so multiply
                                    // jcp.od to get spatial offset.
                                    args.l_offset = (g * jcp.oc + oc)
                                            * (jcp.os * jcp.od);
                                    post_ops_->execute(dst_arr[oc], args);
                                }
                            }
                        }
                    }
                });
            }
        }
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    }
    return status::success;
}

status_t riscv_gemm_convolution_fwd_t::execute_forward_ncsp(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto col = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_col);

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    // The second arg in template means sub_offset0 = true
    // See `blk_off` method definition.
    const size_t src_mb_stride = src_d.blk_off<false, true>(1);
    const size_t src_g_stride = src_d.blk_off<false, true>(0, 1) * jcp.ic;

    const size_t dst_mb_stride = dst_d.blk_off<false, true>(1);
    const size_t dst_g_stride = dst_d.blk_off<false, true>(0, 1) * jcp.oc;

    const size_t weights_oc_size = jcp.ic * jcp.ks;
    const size_t weights_g_size = weights_oc_size * jcp.oc;
    const bool is_problem_3d = pd()->ndims() == 5;

    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    assert(IMPLICATION(is_problem_3d,
            jcp.os_block == jcp.os && jcp.ic_block == jcp.ic
                    && jcp.os_nb_block == 1));

    status_t st = status::success;
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        // non-blocked jit_gemm_convolution_utils::im2col_3d() requires
        // external data initialization by zeroes
        const bool outer_padding = jcp.os_nb_block == 1;
        if (outer_padding && is_problem_3d) {
            for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                _col[i] = (data_t)0;
        }
        auto inner_ker = [&](int spatial, const im_pos_t &curr, im_pos_t &prev,
                                 im_pos_t &step, const im_pos_t &end) {
            const data_t *_src
                    = src + curr.n * src_mb_stride + curr.g * src_g_stride;
            step.oc = nstl::min(
                    jcp.oc_block, nstl::min(jcp.oc, end.oc) - curr.oc);
            step.sp = nstl::min(jcp.os_block,
                    nstl::min(jcp.os - curr.sp, end.sp - spatial));
            step.ic = nstl::min(
                    jcp.ic_block, nstl::min(jcp.ic, end.ic) - curr.ic);
            bool do_im2col = curr.do_im2col(prev);
            prev = curr;

            if (jcp.im2col_sz && do_im2col) {
                if (!is_problem_3d)
                    jit_gemm_convolution_utils::im2col<float>(jcp, _src, _col,
                            curr.sp, step.sp, curr.ic, step.ic);
                else
                    jit_gemm_convolution_utils::im2col_3d<float>(
                            jcp, _src, _col, curr.od, 0, jcp.os);
            }
            const data_t one = 1.0;

            const dim_t M = jcp.os * jcp.od;
            const dim_t m = step.sp;
            const dim_t LDA = jcp.im2col_sz ? m : M;
            data_t *_dst = dst + curr.n * dst_mb_stride + curr.g * dst_g_stride
                    + curr.oc * M + curr.od * jcp.os + curr.sp;
            const dim_t K = step.ic * jcp.ks;
            const dim_t LDB = jcp.ic * jcp.ks;
            const dim_t N = step.oc;

            // TODO: what if this->beta_ != 0 && != 1 ?
            const float beta = (curr.ic == 0) ? this->beta_ : one;
            const float *_source = jcp.im2col_sz
                    ? _col
                    : _src + curr.ic * M + curr.od * jcp.os + curr.sp;
            const data_t *_weights = weights + curr.g * weights_g_size
                    + curr.oc * weights_oc_size + curr.ic * jcp.ks;

            status_t st = extended_sgemm("N", "N", &m, &N, &K, &one, _source,
                    &LDA, _weights, &LDB, &beta, _dst, &M);
            if (st != status::success) return st;

            if (curr.ic == jcp.ic - step.ic) {
                // TODO: for "outer threading" we have parallel section within
                // outermost "parallel". It is not good. Consider to use
                // "parallel" here with number of threads passed as parameter
                const int oc_start = curr.g * jcp.oc + curr.oc;
                if (jcp.with_eltwise || jcp.with_binary) {
                    bool fast_relu_done = false;
                    if (jcp.with_eltwise && jcp.post_ops.len() == 1) {
                        // fast branch for ReLU case
                        const auto &eltwise
                                = jcp.post_ops.entry_.back().eltwise;
                        if (eltwise.alg == alg_kind::eltwise_relu) {
                            parallel_nd(step.oc, [&](dim_t oc) {
                                data_t b = jcp.with_bias ? bias[oc_start + oc]
                                                         : 0;
                                data_t *d_ = _dst + oc * M;

                                int oS = 0;
                                while (oS < m) {
                                    size_t vl = __riscv_vsetvl_e32m1(m - oS);
                                    vfloat32m1_t v_d = __riscv_vle32_v_f32m1(
                                            d_ + oS, vl);
                                    v_d = __riscv_vfadd_vf_f32m1(v_d, b, vl);
                                    vbool32_t mask = __riscv_vmflt_vf_f32m1_b32(
                                            v_d, 0.0f, vl);
                                    v_d = __riscv_vfmul_vf_f32m1_m(
                                            mask, v_d, eltwise.alpha, vl);
                                    v_d = __riscv_vfmul_vf_f32m1(
                                            v_d, eltwise.scale, vl);
                                    __riscv_vse32_v_f32m1(d_ + oS, v_d, vl);
                                    oS += vl;
                                }
                            });
                            fast_relu_done = true;
                        }
                    }
                    if (!fast_relu_done) {
                        parallel_nd(step.oc, [&](dim_t oc) {
                            data_t b = jcp.with_bias ? bias[oc_start + oc] : 0;
                            data_t *d_ = _dst + oc * M;

                            ref_post_ops_t::args_t args;
                            args.ctx = &ctx;
                            args.dst_md = pd()->dst_md();
                            args.l_offset = d_ - dst;

                            for (int oS = 0; oS < m; ++oS) {
                                d_[oS] += b;
                                post_ops_->execute(d_[oS], args);
                                args.l_offset++;
                            }
                        });
                    }

                } else if (jcp.with_bias) {
                    parallel_nd(step.oc, [&](dim_t oc) {
                        data_t b = bias[oc_start + oc];
                        data_t *d_ = _dst + oc * M;

                        int oS = 0;
                        while (oS < m) {
                            size_t vl = __riscv_vsetvl_e32m1(m - oS);
                            vfloat32m1_t v_d
                                    = __riscv_vle32_v_f32m1(d_ + oS, vl);
                            v_d = __riscv_vfadd_vf_f32m1(v_d, b, vl);
                            __riscv_vse32_v_f32m1(d_ + oS, v_d, vl);
                            oS += vl;
                        }
                    });
                }
            }

            return status::success;
        };
        im_pos_t start, end;
        end.ic = jcp.ic;

        if (!is_problem_3d) {
            dim_t sp_work = jcp.mb * jcp.ngroups * jcp.od * jcp.os;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, dim_t(jcp.nthr_oc));
        } else {
            dim_t sp_work = jcp.mb * jcp.ngroups * jcp.od;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, dim_t(jcp.nthr_oc));
            start.sp *= jcp.os;
            end.sp *= jcp.os;
        }

        im_pos_t curr, prev, step;
        prev.n = prev.g = prev.od = prev.sp = prev.ic = -1;
        step.oc = jcp.oc_block;
        step.sp = jcp.os_block;
        step.ic = jcp.ic_block;

        if (jcp.loop_order == gemm_loop_rlb)
            for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                for (int spatial = start.sp; spatial < end.sp;
                        spatial += step.sp) {
                    nd_iterator_init(spatial, curr.n, jcp.mb, curr.g,
                            jcp.ngroups, curr.od, jcp.od, curr.sp, jcp.os);
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc) {
                        status_t st_thr
                                = inner_ker(spatial, curr, prev, step, end);
                        if (st_thr != status::success) {
                            st = st_thr;
                            return;
                        }
                    }
                }
        else if (jcp.loop_order == gemm_loop_lrb)
            for (int spatial = start.sp; spatial < end.sp; spatial += step.sp) {
                nd_iterator_init(spatial, curr.n, jcp.mb, curr.g, jcp.ngroups,
                        curr.od, jcp.od, curr.sp, jcp.os);
                for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc) {
                        status_t st_thr
                                = inner_ker(spatial, curr, prev, step, end);
                        if (st_thr != status::success) {
                            st = st_thr;
                            return;
                        }
                    }
            }
        else
            st = status::unimplemented;
    });

    return st;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl