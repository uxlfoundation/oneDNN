/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_im2row_convolution.hpp"

#include <algorithm>
#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

size_t im2row_size_bytes(
        const kai_im2row_convolution_fwd_t::pd_t &pd, size_t src_dt_size) {
    return static_cast<size_t>(pd.gemm_args()._Msize) * pd.gemm_args()._Ksize
            * src_dt_size;
}

void copy_strided_width(char *&dst, const char *&src, dim_t width_run,
        size_t row_bytes, size_t src_step) {
    for (dim_t i = 0; i < width_run; ++i) {
        std::memcpy(dst, src, row_bytes);
        src += src_step;
        dst += row_bytes;
    }
}

} // namespace

status_t kai_im2row_convolution_fwd_t::pd_t::init(const engine_t *engine) {
    name_ = "im2row:kai";
    CHECK(init_common(engine));
    if (is_1x1(*this)) name_ = "im2row_1x1:kai";
    init_gemm_args(MB() * OH() * OW(), IC() * KH() * KW(), 1, 1, 1, false);
    VDISPATCH_CONV(is_1x1(*this), "only supports 1x1 kernels");
    VDISPATCH_CONV(has_zero_top_left_padding(*this),
            "only supports zero top and left padding");
    VDISPATCH_CONV(source_pixels_in_bounds(*this),
            "requires every 1x1 patch to lie within the source image");
    CHECK(init_gemm(engine));
    auto scratchpad = scratchpad_registry().registrar();
    book_datapath_scratchpad(scratchpad);
    return status::success;
}

void kai_im2row_convolution_fwd_t::pd_t::book_datapath_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    const size_t bytes
            = im2row_size_bytes(*this, types::data_type_size(gemm_src_dt_));
    scratchpad.book<char>(memory_tracking::names::key_conv_gemm_col, bytes);
}

status_t kai_im2row_convolution_fwd_t::init(engine_t *engine) {
    return init_common(engine);
}

void kai_im2row_convolution_fwd_t::copy_nhwc_im2row(
        const kernel_call_args_t &args, char *im2row) const {
    const auto &pd
            = static_cast<const kai_im2row_convolution_fwd_t::pd_t &>(args.pd);
    const dim_t MB = pd.MB();
    const dim_t IC = pd.IC();
    const dim_t OH = pd.OH();
    const dim_t OW = pd.OW();
    const dim_t KSH = pd.KSH();
    const dim_t KSW = pd.KSW();
    const size_t src_dt_size = types::data_type_size(pd.src_md()->data_type);
    const size_t row_bytes = static_cast<size_t>(IC) * src_dt_size;
    const dim_t rows = MB * OH * OW;
    const dim_t batch_rows = OH * OW;
    const size_t src_pixel_stride_bytes = static_cast<size_t>(IC) * src_dt_size;
    const size_t src_scanline_stride_bytes = pd.IW() * src_pixel_stride_bytes;
    const size_t src_image_stride_bytes = pd.IH() * src_scanline_stride_bytes;
    const size_t src_h_step_bytes
            = static_cast<size_t>(KSH) * src_scanline_stride_bytes;
    const size_t src_ow_step_bytes
            = static_cast<size_t>(KSW) * src_pixel_stride_bytes;
    const char *src_base = args.src_base;
    // Limit copying workers by scanline count, including the batch dimension.
    const int active_nthr = static_cast<int>(
            std::min<dim_t>(dnnl_get_current_num_threads(), MB * OH));

    // Keep the OpenMP team size stable across im2row and GEMM while limiting
    // the number of threads that do copy work.
    const int team_nthr = active_nthr > 1 ? dnnl_get_current_num_threads() : 1;

    parallel(team_nthr, [=](int ithr, int) {
        if (ithr >= active_nthr) return;

        dim_t start = 0;
        dim_t end = 0;
        balance211(rows, active_nthr, ithr, start, end);

        dim_t mb = start / batch_rows;
        dim_t rem = start - mb * batch_rows;
        dim_t oh = rem / OW;
        dim_t ow = rem - oh * OW;
        dim_t row = start;
        char *dst = im2row + static_cast<size_t>(row) * row_bytes;
        const char *src = src_base
                + static_cast<size_t>(mb) * src_image_stride_bytes
                + static_cast<size_t>(oh) * src_h_step_bytes
                + static_cast<size_t>(ow) * src_ow_step_bytes;
        // For 1x1 with unit horizontal stride, consecutive patches in an
        // output row occupy contiguous NHWC storage. Copy each run into the
        // im2row buffer.
        const bool contiguous_width = KSW == 1;

        while (row < end) {
            const dim_t width_run = std::min(end - row, OW - ow);
            if (contiguous_width) {
                const size_t bytes = static_cast<size_t>(width_run) * row_bytes;
                std::memcpy(dst, src, bytes);
                src += static_cast<size_t>(width_run) * src_pixel_stride_bytes;
                dst += bytes;
            } else {
                copy_strided_width(
                        dst, src, width_run, row_bytes, src_ow_step_bytes);
            }
            row += width_run;
            ow += width_run;
            if (ow == OW && row < end) {
                ow = 0;
                if (++oh == OH) {
                    oh = 0;
                    ++mb;
                }
                src = src_base
                        + static_cast<size_t>(mb) * src_image_stride_bytes
                        + static_cast<size_t>(oh) * src_h_step_bytes;
            }
        }
    });
}

status_t kai_im2row_convolution_fwd_t::prepare_src(
        const kernel_call_args_t &args, char *im2row) const {
    copy_nhwc_im2row(args, im2row);
    return status::success;
}

status_t kai_im2row_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;
    kernel_call_args_t args {ctx, *pd(), *kernel};
    CHECK(prepare(args));
    CHECK(setup_kernel_arrays(args));
    return execute_kernel(args);
}

status_t kai_im2row_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd = static_cast<const pd_t &>(args.pd);
    auto *im2row = args.ctx.get_scratchpad_grantor().get<char>(
            memory_tracking::names::key_conv_gemm_col);
    CHECK(prepare_src(args, im2row));
    const void *A = im2row;
    const int lda = static_cast<int>(pd.gemm_args()._Ksize);
    const int A_batch_stride = 0;
    const int A_multi_stride = 0;
    const int C_batch_stride = 0;
    const int C_multi_stride = 0;

    args.kernel.set_arrays_generic(A, lda, A_batch_stride, A_multi_stride,
            args.B(), args.ldb(), args.B_multi_stride(), args.C(), args.ldc(),
            C_batch_stride, C_multi_stride, args.bias(),
            args.bias_multi_stride());
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
