/*******************************************************************************
* Copyright 2020-2026 Arm Ltd. and affiliates
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

#include <cstdint>
#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

template <typename T>
void linearize_volume_nchw(const char *src_base, char *dst_base,
        dim_t top_left_x, dim_t top_left_y, dim_t kernel_width,
        dim_t kernel_height, dim_t kernel_depth, dim_t input_w, dim_t input_h,
        size_t input_stride_x_bytes, size_t input_stride_y_bytes,
        size_t input_stride_z_bytes, dim_t dilation_x, dim_t dilation_y) {
    const dim_t kernel_area = kernel_width * kernel_height;
    const dim_t x_e = top_left_x + kernel_width * dilation_x;
    const dim_t y_e = top_left_y + kernel_height * dilation_y;
    auto *out_ptr = reinterpret_cast<T *>(dst_base);

    dim_t d = 0;
    for (; d <= kernel_depth - 3; d += 3) {
        for (dim_t y = top_left_y; y < y_e; y += dilation_y) {
            if (y < 0 || y >= input_h) {
                for (dim_t x = top_left_x; x < x_e;
                        x += dilation_x, ++out_ptr) {
                    *(out_ptr + 0 * kernel_area) = T {};
                    *(out_ptr + 1 * kernel_area) = T {};
                    *(out_ptr + 2 * kernel_area) = T {};
                }
            } else {
                for (dim_t x = top_left_x; x < x_e;
                        x += dilation_x, ++out_ptr) {
                    if (x < 0 || x >= input_w) {
                        *(out_ptr + 0 * kernel_area) = T {};
                        *(out_ptr + 1 * kernel_area) = T {};
                        *(out_ptr + 2 * kernel_area) = T {};
                    } else {
                        const auto src_offset
                                = static_cast<size_t>(y) * input_stride_y_bytes
                                + static_cast<size_t>(x) * input_stride_x_bytes;
                        *(out_ptr + 0 * kernel_area)
                                = *reinterpret_cast<const T *>(src_base
                                        + static_cast<size_t>(d + 0)
                                                * input_stride_z_bytes
                                        + src_offset);
                        *(out_ptr + 1 * kernel_area)
                                = *reinterpret_cast<const T *>(src_base
                                        + static_cast<size_t>(d + 1)
                                                * input_stride_z_bytes
                                        + src_offset);
                        *(out_ptr + 2 * kernel_area)
                                = *reinterpret_cast<const T *>(src_base
                                        + static_cast<size_t>(d + 2)
                                                * input_stride_z_bytes
                                        + src_offset);
                    }
                }
            }
        }
        out_ptr += 2 * kernel_area;
    }

    for (; d < kernel_depth; ++d) {
        for (dim_t y = top_left_y; y < y_e; y += dilation_y) {
            if (y < 0 || y >= input_h) {
                for (dim_t x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                    *out_ptr = T {};
            } else {
                for (dim_t x = top_left_x; x < x_e;
                        x += dilation_x, ++out_ptr) {
                    if (x < 0 || x >= input_w) {
                        *out_ptr = T {};
                    } else {
                        *out_ptr = *reinterpret_cast<const T *>(src_base
                                + static_cast<size_t>(d) * input_stride_z_bytes
                                + static_cast<size_t>(y) * input_stride_y_bytes
                                + static_cast<size_t>(x)
                                        * input_stride_x_bytes);
                    }
                }
            }
        }
    }
}

template <typename T>
void linearize_volume_nchw_fixed_format(const char *src_base, char *dst_base,
        dim_t top_left_x, dim_t top_left_y, dim_t kernel_width,
        dim_t kernel_height, dim_t kernel_depth, dim_t input_w, dim_t input_h,
        size_t input_stride_x_bytes, size_t input_stride_y_bytes,
        size_t input_stride_z_bytes, dim_t dilation_x, dim_t dilation_y) {
    const dim_t x_e = top_left_x + kernel_width * dilation_x;
    const dim_t y_e = top_left_y + kernel_height * dilation_y;
    auto *out_ptr = reinterpret_cast<T *>(dst_base);

    for (dim_t y = top_left_y; y < y_e; y += dilation_y) {
        const bool y_in_bounds = 0 <= y && y < input_h;
        for (dim_t x = top_left_x; x < x_e; x += dilation_x) {
            if (!y_in_bounds || x < 0 || x >= input_w) {
                for (dim_t d = 0; d < kernel_depth; ++d)
                    *out_ptr++ = T {};
            } else {
                const auto src_offset
                        = static_cast<size_t>(y) * input_stride_y_bytes
                        + static_cast<size_t>(x) * input_stride_x_bytes;
                for (dim_t d = 0; d < kernel_depth; ++d) {
                    *out_ptr++ = *reinterpret_cast<const T *>(src_base
                            + static_cast<size_t>(d) * input_stride_z_bytes
                            + src_offset);
                }
            }
        }
    }
}

} // namespace

void kai_im2row_convolution_fwd_t::pd_t::book_datapath_scratchpad(
        memory_tracking::registrar_t &scratchpad, size_t src_dt_size) const {
    const size_t im2row_elems
            = static_cast<size_t>(MB()) * OH() * OW() * KH() * KW() * IC();
    scratchpad.book(memory_tracking::names::key_conv_gemm_col,
            im2row_elems * src_dt_size, 1, 64, 64);
}

status_t kai_im2row_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd
            = static_cast<const kai_im2row_convolution_fwd_t::pd_t &>(args.pd);
    const dim_t MB = pd.MB();
    const dim_t IC = pd.IC();
    const dim_t IH = pd.IH();
    const dim_t IW = pd.IW();
    const dim_t OH = pd.OH();
    const dim_t OW = pd.OW();
    const dim_t KH = pd.KH();
    const dim_t KW = pd.KW();
    const dim_t KSH = pd.KSH();
    const dim_t KSW = pd.KSW();
    const dim_t KDH = pd.KDH();
    const dim_t KDW = pd.KDW();
    const dim_t padT = pd.padT();
    const dim_t padL = pd.padL();
    const bool src_channels_last = pd.src_channels_last_;
    const bool fixed_format = pd.fixed_format_;

    const dim_t k_total = IC * KH * KW;
    const dim_t im2row_batch_stride = OH * OW * k_total;
    const size_t patch_bytes = static_cast<size_t>(IC) * args.src_dt_size;
    const size_t k_total_bytes
            = static_cast<size_t>(k_total) * args.src_dt_size;
    const size_t im2row_batch_stride_bytes
            = static_cast<size_t>(im2row_batch_stride) * args.src_dt_size;
    auto *im2row = args.scratchpad.get<char>(
            memory_tracking::names::key_conv_gemm_col);

    parallel_nd_ext(args.num_threads, MB, OH, OW,
            [&](int, int, dim_t mb, dim_t oh, dim_t ow) {
        const auto *src_batch
                = args.src_base + mb * args.src_batch_stride_bytes;
        auto *row = im2row + mb * im2row_batch_stride_bytes
                + static_cast<size_t>(oh * OW + ow) * k_total_bytes;

        if (src_channels_last) {
            size_t offset = 0;
            for (dim_t kh = 0; kh < KH; ++kh) {
                const dim_t ih = oh * KSH + kh * (KDH + 1) - padT;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    const dim_t iw = ow * KSW + kw * (KDW + 1) - padL;
                    auto *dst_patch = row + offset;
                    if (0 <= ih && ih < IH && 0 <= iw && iw < IW) {
                        const auto *src_patch = src_batch
                                + ((ih * IW + iw) * args.src_col_stride_bytes);
                        std::memcpy(dst_patch, src_patch, patch_bytes);
                    } else {
                        std::memset(dst_patch, 0, patch_bytes);
                    }
                    offset += patch_bytes;
                }
            }
        } else {
            const dim_t start_w = ow * KSW - padL;
            const dim_t start_h = oh * KSH - padT;
            const dim_t dilation_w = KDW + 1;
            const dim_t dilation_h = KDH + 1;
            if (fixed_format && args.src_dt_size == sizeof(uint32_t)) {
                linearize_volume_nchw_fixed_format<uint32_t>(src_batch, row,
                        start_w, start_h, KW, KH, IC, IW, IH, args.src_dt_size,
                        args.src_h_stride_bytes, args.src_channel_stride_bytes,
                        dilation_w, dilation_h);
            } else if (fixed_format && args.src_dt_size == sizeof(uint16_t)) {
                linearize_volume_nchw_fixed_format<uint16_t>(src_batch, row,
                        start_w, start_h, KW, KH, IC, IW, IH, args.src_dt_size,
                        args.src_h_stride_bytes, args.src_channel_stride_bytes,
                        dilation_w, dilation_h);
            } else if (args.src_dt_size == sizeof(uint32_t)) {
                linearize_volume_nchw<uint32_t>(src_batch, row, start_w,
                        start_h, KW, KH, IC, IW, IH, args.src_dt_size,
                        args.src_h_stride_bytes, args.src_channel_stride_bytes,
                        dilation_w, dilation_h);
            } else if (args.src_dt_size == sizeof(uint16_t)) {
                linearize_volume_nchw<uint16_t>(src_batch, row, start_w,
                        start_h, KW, KH, IC, IW, IH, args.src_dt_size,
                        args.src_h_stride_bytes, args.src_channel_stride_bytes,
                        dilation_w, dilation_h);
            }
        }
    });

    args.kernel.set_arrays_generic(im2row, static_cast<int>(k_total),
            static_cast<int>(im2row_batch_stride), 0, args.wei_base,
            args.ld_wei, 0, args.kernel_dst_base, args.ld_dst,
            args.dst_batch_stride, 0, args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
