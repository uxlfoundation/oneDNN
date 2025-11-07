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

#include "cpu/aarch64/kai_convolution.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace kai_utils;

namespace {

bool bias_ok(const cpu_convolution_fwd_pd_t &pd) {
    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type;
}

format_tag_t exact_src_dst_tag(const memory_desc_t &md) {
    using namespace format_tag;
    if (md.format_kind != format_kind::blocked || md.ndims != 4)
        return format_tag::undef;

    const auto &strides = md.format_desc.blocking.strides;
    const dim_t c = md.dims[1];
    const dim_t h = md.dims[2];
    const dim_t w = md.dims[3];
    if (strides[0] == c * h * w && strides[1] == h * w && strides[2] == w
            && strides[3] == 1)
        return nchw;
    if (strides[0] == h * w * c && strides[1] == 1 && strides[2] == w * c
            && strides[3] == c)
        return nhwc;

    return format_tag::undef;
}

format_tag_t src_dst_tag(const memory_desc_t &md) {
    using namespace format_tag;
    const auto exact_tag = exact_src_dst_tag(md);
    if (exact_tag != format_tag::undef) return exact_tag;
    return memory_desc_matches_one_of_tag(md, nhwc, nchw);
}

bool can_use_direct_1x1_src(const kai_convolution_fwd_t::pd_t &pd) {
    return !pd.indirect_ && exact_src_dst_tag(*pd.src_md()) == format_tag::nhwc
            && pd.KH() == 1 && pd.KW() == 1 && pd.KSH() == 1 && pd.KSW() == 1
            && pd.padT() == 0 && pd.padL() == 0 && pd.OH() == pd.IH()
            && pd.OW() == pd.IW();
}

int cap_threads_for_small_gemm(
        const kai_convolution_fwd_t::pd_t &pd, int requested_threads) {
    if (pd.KH() != 1 || pd.KW() != 1) return requested_threads;

    const dim_t m = pd.MB() * pd.OH() * pd.OW();
    const dim_t mn = m * pd.OC();
    int capped_threads = requested_threads;

    if (mn < 2048)
        capped_threads = 2;
    else if (mn < 8192)
        capped_threads = 8;
    else if (mn < 16384)
        capped_threads = 16;

    return std::max(1, std::min(requested_threads, capped_threads));
}

void build_indirect_input_table(const kai_convolution_fwd_t::pd_t &pd,
        const char *src_base, size_t src_batch_stride_bytes,
        size_t src_col_stride_bytes, char *buffer, size_t src_dt_size,
        int num_threads) {
    const size_t kernel_points = static_cast<size_t>(pd.KH() * pd.KW());
    const size_t output_points = static_cast<size_t>(pd.OH() * pd.OW());
    const size_t outer_count = static_cast<size_t>(pd.MB()) * kernel_points;
    const size_t inner_count = outer_count * output_points;
    const size_t outer_bytes = outer_count * sizeof(const void *const *);
    const size_t inner_bytes = inner_count * sizeof(const void *);
    const size_t pad_offset = utils::rnd_up(
            outer_bytes + inner_bytes, std::max<size_t>(1, src_dt_size));
    const size_t pad_bytes = static_cast<size_t>(pd.IC()) * src_dt_size;

    auto *outer = reinterpret_cast<const void *const **>(buffer);
    auto *inner = reinterpret_cast<const void **>(buffer + outer_bytes);
    auto *pad = buffer + pad_offset;
    std::memset(pad, 0, pad_bytes);

    parallel_nd_ext(num_threads, pd.MB(), pd.KH(), pd.KW(), pd.OH(),
            [&](int, int, dim_t mb, dim_t kh, dim_t kw, dim_t oh) {
                const auto *src_batch = src_base + mb * src_batch_stride_bytes;
                const size_t kernel_idx
                        = static_cast<size_t>(kh * pd.KW() + kw);
                const size_t outer_idx
                        = static_cast<size_t>(mb) * kernel_points + kernel_idx;
                const size_t inner_base = outer_idx * output_points;
                if (oh == 0) outer[outer_idx] = inner + inner_base;
                const dim_t ih
                        = oh * pd.KSH() + kh * (pd.KDH() + 1) - pd.padT();
                const bool ih_in_bounds = 0 <= ih && ih < pd.IH();
                const auto *src_row = ih_in_bounds ? src_batch
                                + static_cast<size_t>(ih * pd.IW())
                                        * src_col_stride_bytes
                                                   : nullptr;
                const size_t row_base
                        = inner_base + static_cast<size_t>(oh) * pd.OW();

                PRAGMA_OMP_SIMD()
                for (dim_t ow = 0; ow < pd.OW(); ++ow) {
                    const dim_t iw
                            = ow * pd.KSW() + kw * (pd.KDW() + 1) - pd.padL();
                    inner[row_base + static_cast<size_t>(ow)]
                            = (src_row && 0 <= iw && iw < pd.IW()) ? src_row
                                    + static_cast<size_t>(iw)
                                            * src_col_stride_bytes
                                                                   : pad;
                }
            });
}

void copy_contiguous_channels_to_strided(char *dst, const char *src,
        dim_t channels, size_t dst_stride_bytes, size_t dt_size) {
    if (dt_size == sizeof(uint32_t)) {
        const auto *src_u32 = reinterpret_cast<const uint32_t *>(src);
        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < channels; ++c) {
            *reinterpret_cast<uint32_t *>(
                    dst + static_cast<size_t>(c) * dst_stride_bytes)
                    = src_u32[c];
        }
    } else if (dt_size == sizeof(uint16_t)) {
        const auto *src_u16 = reinterpret_cast<const uint16_t *>(src);
        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < channels; ++c) {
            *reinterpret_cast<uint16_t *>(
                    dst + static_cast<size_t>(c) * dst_stride_bytes)
                    = src_u16[c];
        }
    } else {
        for (dim_t c = 0; c < channels; ++c) {
            std::memcpy(dst + static_cast<size_t>(c) * dst_stride_bytes,
                    src + static_cast<size_t>(c) * dt_size, dt_size);
        }
    }
}

void scatter_channels_last_dst(const cpu_convolution_fwd_pd_t &pd,
        const void *tmp_dst, void *dst, int num_threads) {
    constexpr int n_dim = 0;
    constexpr int c_dim = 1;
    constexpr int h_dim = 2;
    constexpr int w_dim = 3;
    const memory_desc_wrapper dst_d(pd.dst_md());
    const size_t dt_size = types::data_type_size(dst_d.data_type());
    const size_t n_stride
            = static_cast<size_t>(
                      pd.dst_md()->format_desc.blocking.strides[n_dim])
            * dt_size;
    const size_t c_stride
            = static_cast<size_t>(
                      pd.dst_md()->format_desc.blocking.strides[c_dim])
            * dt_size;
    const size_t h_stride
            = static_cast<size_t>(
                      pd.dst_md()->format_desc.blocking.strides[h_dim])
            * dt_size;
    const size_t w_stride
            = static_cast<size_t>(
                      pd.dst_md()->format_desc.blocking.strides[w_dim])
            * dt_size;
    const auto *src = static_cast<const char *>(tmp_dst);
    auto *dst_c = static_cast<char *>(dst);

    parallel_nd_ext(num_threads, pd.MB(), pd.OH(), pd.OW(),
            [&](int, int, dim_t mb, dim_t oh, dim_t ow) {
                const auto *src_row = src
                        + static_cast<size_t>(
                                  (mb * pd.OH() + oh) * pd.OW() + ow)
                                * pd.OC() * dt_size;
                auto *dst_row = dst_c + static_cast<size_t>(mb) * n_stride
                        + static_cast<size_t>(oh) * h_stride
                        + static_cast<size_t>(ow) * w_stride;
                copy_contiguous_channels_to_strided(
                        dst_row, src_row, pd.OC(), c_stride, dt_size);
            });
}

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

bool regular_swd_ok(const cpu_convolution_fwd_pd_t &pd) {
    const auto src_dt = pd.invariant_src_md()->data_type;
    const auto wei_dt = pd.invariant_wei_md()->data_type;
    const auto dst_dt = pd.invariant_dst_md()->data_type;

    return (src_dt == f32 && wei_dt == f32 && dst_dt == f32)
            || (src_dt == bf16 && wei_dt == bf16
                    && utils::one_of(dst_dt, bf16, f32))
            || (src_dt == f16 && wei_dt == f16
                    && utils::one_of(dst_dt, f16, f32));
}

} // namespace

bool kai_convolution_fwd_t::pd_t::swd_dt(
        data_type_t s, data_type_t w, data_type_t d) const {
    return src_md()->data_type == s && weights_md()->data_type == w
            && dst_md()->data_type == d;
}

bool kai_convolution_fwd_t::pd_t::set_default_formats() {
    using namespace format_tag;
    if (src_md_.format_kind == format_kind::any
            && dst_md_.format_kind == format_kind::any) {
        return set_default_formats_common(nhwc, hwio, nhwc);
    }

    const auto src_tag = src_dst_tag(src_md_);
    const auto dst_tag = src_dst_tag(dst_md_);
    if (src_md_.format_kind == format_kind::any && dst_tag != format_tag::undef)
        CHECK(memory_desc_init_by_tag(src_md_, dst_tag));
    else if (dst_md_.format_kind == format_kind::any
            && src_tag != format_tag::undef)
        CHECK(memory_desc_init_by_tag(dst_md_, src_tag));

    const auto resolved_src_tag = src_dst_tag(src_md_);
    const auto resolved_dst_tag = src_dst_tag(dst_md_);
    if (resolved_src_tag == format_tag::undef
            || resolved_dst_tag == format_tag::undef
            || resolved_src_tag != resolved_dst_tag)
        return false;

    if (weights_md_.format_kind == format_kind::any) {
        const format_tag_t wei_tag = resolved_src_tag == nchw ? ihwo : hwio;
        CHECK(memory_desc_init_by_tag(weights_md_, wei_tag));
    }

    if (with_bias() && bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    return true;
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_convolution_fwd_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(*args_, cfg_.get(), src_md()->data_type,
            weights_md()->data_type, dst_md()->data_type);
}

status_t kai_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;
    MAYBE_UNUSED(engine);

    const bool requested_fixed_format
            = weights_md_.format_kind == format_kind::any;

    bool ok = true && is_fwd()
            && set_default_alg_kind(alg_kind::convolution_direct)
            && ndims() == 4 && !with_groups() && regular_swd_ok(*this)
            && !has_zero_dim_memory() && !has_runtime_dims_or_strides()
            && attr()->has_default_values(primitive_mask_t::fpmath_mode
                            | primitive_mask_t::accumulation_mode,
                    dst_md()->data_type)
            && set_default_formats()
            && attr_.set_default_formats(dst_md()) == status::success;
    if (!ok) return status::unimplemented;

    const auto src_tag = src_dst_tag(src_md_);
    const auto dst_tag = src_dst_tag(dst_md_);
    const auto wei_tag = memory_desc_matches_one_of_tag(
            weights_md_, format_tag::hwio, format_tag::ihwo);

    VDISPATCH_CONV(src_md_.format_kind == format_kind::any
                    || utils::one_of(
                            src_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dst_md_.format_kind == format_kind::any
                    || utils::one_of(
                            dst_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(weights_md_.format_kind == format_kind::any
                    || wei_tag == format_tag::hwio
                    || wei_tag == format_tag::ihwo,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    fast_mode_ = use_bf16_fast_math(attr(), src_md()->data_type,
                         weights_md()->data_type, dst_md()->data_type)
            || (src_md()->data_type == data_type::f16
                    && utils::one_of(attr()->acc_mode_,
                            accumulation_mode::relaxed,
                            accumulation_mode::f16));
    src_channels_last_ = src_tag != format_tag::nchw;
    dst_channels_last_ = dst_tag != format_tag::nchw;
    const bool weights_hwio = wei_tag == format_tag::hwio;
    VDISPATCH_CONV(src_channels_last_ == weights_hwio || (KH() == 1 && KW() == 1),
            "kai convolution requires weights K dimension order to match src");
    wei_k_stride_dim_ = weights_hwio ? 1 : 3;
    const memory_desc_t original_weights_md = weights_md_;
    const size_t src_dt_size = types::data_type_size(src_md()->data_type);

    for (bool use_indirect : {true, false}) {
        indirect_ = use_indirect;
        if (indirect_) {
            name_ = "indirect_gemm:kai";
        } else if(can_use_direct_1x1_src(*this)) {
            name_ = "direct_1x1:kai";
        } else {
            name_ = "im2row:kai";
        }
        cfg_ = std::make_shared<kai::ops::GemmConfig>();
        args_.reset();
        fixed_format_ = false;
        run_weight_reorder_ = false;
        weights_md_ = original_weights_md;

        if (indirect_ && (!src_channels_last_ || !dst_channels_last_)) continue;

        const bool try_fixed_format = requested_fixed_format && !fast_mode_
                && (indirect_ || (KH() == 1 && KW() == 1));
        if (try_fixed_format) cfg_->weight_format = kai::ops::WeightFormat::ANY;

        const auto make_args = [&](bool fixed_format) {
            const unsigned int m = static_cast<unsigned int>(
                    (indirect_ ? 1 : MB()) * OH() * OW());
            const unsigned int k = static_cast<unsigned int>(
                    indirect_ ? IC() : IC() * KH() * KW());
            const unsigned int k_sections
                    = static_cast<unsigned int>(indirect_ ? KH() * KW() : 1);
            const unsigned int nbatches
                    = static_cast<unsigned int>(indirect_ ? MB() : 1);
            return std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), m, OC(),
                    k, k_sections, nbatches, 1, indirect_,
                    kai::ops::Activation {}, dnnl_get_current_num_threads(),
                    fixed_format, fast_mode_, false, cfg_.get());
        };

        args_ = make_args(try_fixed_format);
        std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
        if (!kernel) continue;

        cfg_ = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());
        fixed_format_
                = try_fixed_format && is_fixed_format(cfg_->weight_format);

        if (indirect_) {
            if (KH() == 1 && KW() == 1) continue;
        }

        args_ = make_args(fixed_format_);
        kernel = create_kai_gemm();
        if (!kernel) continue;

        if (fixed_format_) {
            constexpr dim_t O_dim = 0;
            constexpr dim_t I_dim = 1;
            constexpr dim_t H_dim = 2;
            constexpr dim_t W_dim = 3;
            weight_format_to_memory_desc(weights_md_, cfg_->weight_format,
                    I_dim, O_dim, {W_dim, H_dim});
        }

        run_weight_reorder_ = !fixed_format_ && kernel->B_is_pretransposed();

        auto scratchpad = scratchpad_registry().registrar();
        if (kernel->get_working_size() != 0) {
            scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                    kernel->get_working_size(), 1);
        }

        if (run_weight_reorder_) {
            scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                    kernel->get_B_pretransposed_array_size(), 1);
        }

        if (!dst_channels_last_) {
            const size_t tmp_dst_bytes = static_cast<size_t>(MB()) * OH() * OW()
                    * OC() * types::data_type_size(dst_md()->data_type);
            scratchpad.book(memory_tracking::names::key_conv_ncsp_dst,
                    tmp_dst_bytes, 1, 64, 64);
        }

        if (indirect_) {
            const size_t kernel_points = static_cast<size_t>(KH() * KW());
            const size_t output_points = static_cast<size_t>(OH() * OW());
            const size_t outer_count
                    = static_cast<size_t>(MB()) * kernel_points;
            const size_t inner_count = outer_count * output_points;
            const size_t outer_bytes
                    = outer_count * sizeof(const void *const *);
            const size_t inner_bytes = inner_count * sizeof(const void *);
            const size_t pad_offset = utils::rnd_up(outer_bytes + inner_bytes,
                    std::max<size_t>(1, src_dt_size));
            const size_t total_bytes
                    = pad_offset + static_cast<size_t>(IC()) * src_dt_size;

            scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                    total_bytes, 1, 64, 64);
        } else if (!can_use_direct_1x1_src(*this)) {
            const size_t im2row_elems = static_cast<size_t>(MB()) * OH() * OW()
                    * KH() * KW() * IC();
            scratchpad.book(memory_tracking::names::key_conv_gemm_col,
                    im2row_elems * src_dt_size, 1, 64, 64);
        }

        return status::success;
    }

    return status::unimplemented;
}

status_t kai_convolution_fwd_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
    return status::success;
}

status_t kai_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    std::unique_ptr<kai::ops::IGemmCommon> kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const kai::ops::ndrange_t window_size = kernel->get_window_size();
    const int num_windows = static_cast<int>(window_size.total_size());
    int num_threads = std::min(num_windows,
            cap_threads_for_small_gemm(*pd(), dnnl_get_current_num_threads()));

    unsigned int row_parts = num_threads;
    unsigned int col_parts = 1;
    if (window_size.get_size(1) > 1) {
        row_parts = split_window_2d(num_threads, window_size);
        col_parts = num_threads / row_parts;

        const unsigned int max_threads_2d
                = std::min(row_parts, window_size.get_size(0))
                * std::min(col_parts, window_size.get_size(1));
        if (max_threads_2d < static_cast<unsigned int>(num_threads)) {
            row_parts = std::min(row_parts, window_size.get_size(0));
            col_parts = std::min(col_parts, window_size.get_size(1));
            num_threads = static_cast<int>(max_threads_2d);
        }
    }

    kernel->set_nthreads(num_threads);

    const auto *src_base = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *wei_base = const_cast<void *>(raw_wei);
    if (pd()->run_weight_reorder_) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
    }

    void *dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    constexpr int src_n_dim = 0;
    constexpr int src_c_dim = 1;
    constexpr int src_h_dim = 2;
    constexpr int src_w_dim = 3;
    constexpr int dst_n_dim = 0;
    constexpr int dst_w_dim = 3;
    constexpr int wei_o_dim = 0;

    const int ld_src = static_cast<int>(
            pd()->src_md()->format_desc.blocking.strides[src_w_dim]);
    const int ld_dst = pd()->dst_channels_last_
            ? static_cast<int>(
                      pd()->dst_md()->format_desc.blocking.strides[dst_w_dim])
            : static_cast<int>(pd()->OC());
    const int ld_wei = static_cast<int>(pd()->fixed_format_
                    ? pd()->weights_md()
                              ->format_desc.blocking.strides[wei_o_dim]
                    : pd()->weights_md()
                              ->format_desc.blocking
                              .strides[pd()->wei_k_stride_dim_]);

    const int src_batch_stride = static_cast<int>(
            pd()->src_md()->format_desc.blocking.strides[src_n_dim]);
    const int src_channel_stride = static_cast<int>(
            pd()->src_md()->format_desc.blocking.strides[src_c_dim]);
    const int src_h_stride = static_cast<int>(
            pd()->src_md()->format_desc.blocking.strides[src_h_dim]);
    const int dst_batch_stride = pd()->dst_channels_last_
            ? static_cast<int>(
                      pd()->dst_md()->format_desc.blocking.strides[dst_n_dim])
            : static_cast<int>(pd()->OH() * pd()->OW() * pd()->OC());
    const size_t src_dt_size = types::data_type_size(pd()->src_md()->data_type);
    const size_t src_col_stride_bytes
            = static_cast<size_t>(ld_src) * src_dt_size;
    const size_t src_channel_stride_bytes
            = static_cast<size_t>(src_channel_stride) * src_dt_size;
    const size_t src_h_stride_bytes
            = static_cast<size_t>(src_h_stride) * src_dt_size;
    const size_t src_batch_stride_bytes
            = static_cast<size_t>(src_batch_stride) * src_dt_size;
    void *kernel_dst_base = pd()->dst_channels_last_
            ? dst_base
            : scratchpad.get<void>(memory_tracking::names::key_conv_ncsp_dst);

    if (pd()->run_weight_reorder_) {
        const unsigned int wsize = kernel->get_B_pretranspose_window_size();
        parallel(num_threads, [&](int ithr, int nthr) {
            const unsigned int start = (ithr * wsize) / nthr;
            const unsigned int end = ((ithr + 1) * wsize) / nthr;
            if (start < end) {
                kernel->pretranspose_B_array_part_generic(
                        wei_base, raw_wei, ld_wei, 0, false, start, end);
            }
        });
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    if (pd()->indirect_) {
        char *buffer = scratchpad.get<char>(
                memory_tracking::names::key_gemm_tmp_buffer);
        build_indirect_input_table(*pd(), src_base, src_batch_stride_bytes,
                src_col_stride_bytes, buffer, src_dt_size, num_threads);
        auto *outer = reinterpret_cast<const void *const **>(buffer);

        kernel->set_indirect_parameters_generic(pd()->IC(),
                reinterpret_cast<const void *const *const *>(outer));
        kernel->set_arrays_generic(nullptr, 0, 0, 0, wei_base, ld_wei, 0,
                kernel_dst_base, ld_dst, dst_batch_stride, 0, bias_base, 0);
    } else if (can_use_direct_1x1_src(*pd())) {
        kernel->set_arrays_generic(src_base, ld_src, 0, 0, wei_base, ld_wei, 0,
                kernel_dst_base, ld_dst, dst_batch_stride, 0, bias_base, 0);
    } else {
        const dim_t k_total = pd()->IC() * pd()->KH() * pd()->KW();
        const dim_t im2row_batch_stride = pd()->OH() * pd()->OW() * k_total;
        const size_t patch_bytes
                = static_cast<size_t>(pd()->IC()) * src_dt_size;
        const size_t k_total_bytes = static_cast<size_t>(k_total) * src_dt_size;
        const size_t im2row_batch_stride_bytes
                = static_cast<size_t>(im2row_batch_stride) * src_dt_size;
        auto *im2row = scratchpad.get<char>(
                memory_tracking::names::key_conv_gemm_col);

        parallel_nd_ext(num_threads, pd()->MB(), pd()->OH(), pd()->OW(),
                [&](int, int, dim_t mb, dim_t oh, dim_t ow) {
                    const auto *src_batch
                            = src_base + mb * src_batch_stride_bytes;
                    auto *row = im2row + mb * im2row_batch_stride_bytes
                            + static_cast<size_t>(oh * pd()->OW() + ow)
                                    * k_total_bytes;

                    if (pd()->src_channels_last_) {
                        size_t offset = 0;
                        for (dim_t kh = 0; kh < pd()->KH(); ++kh) {
                            const dim_t ih = oh * pd()->KSH()
                                    + kh * (pd()->KDH() + 1) - pd()->padT();
                            for (dim_t kw = 0; kw < pd()->KW(); ++kw) {
                                const dim_t iw = ow * pd()->KSW()
                                        + kw * (pd()->KDW() + 1) - pd()->padL();
                                auto *dst_patch = row + offset;
                                if (0 <= ih && ih < pd()->IH() && 0 <= iw
                                        && iw < pd()->IW()) {
                                    const auto *src_patch = src_batch
                                            + ((ih * pd()->IW() + iw)
                                                    * src_col_stride_bytes);
                                    std::memcpy(
                                            dst_patch, src_patch, patch_bytes);
                                } else {
                                    std::memset(dst_patch, 0, patch_bytes);
                                }
                                offset += patch_bytes;
                            }
                        }
                    } else {
                        const dim_t start_w = ow * pd()->KSW() - pd()->padL();
                        const dim_t start_h = oh * pd()->KSH() - pd()->padT();
                        const dim_t dilation_w = pd()->KDW() + 1;
                        const dim_t dilation_h = pd()->KDH() + 1;
                        if (src_dt_size == sizeof(uint32_t)) {
                            linearize_volume_nchw<uint32_t>(src_batch, row,
                                    start_w, start_h, pd()->KW(), pd()->KH(),
                                    pd()->IC(), pd()->IW(), pd()->IH(),
                                    src_dt_size, src_h_stride_bytes,
                                    src_channel_stride_bytes, dilation_w,
                                    dilation_h);
                        } else if (src_dt_size == sizeof(uint16_t)) {
                            linearize_volume_nchw<uint16_t>(src_batch, row,
                                    start_w, start_h, pd()->KW(), pd()->KH(),
                                    pd()->IC(), pd()->IW(), pd()->IH(),
                                    src_dt_size, src_h_stride_bytes,
                                    src_channel_stride_bytes, dilation_w,
                                    dilation_h);
                        }
                    }
                });

        kernel->set_arrays_generic(im2row, static_cast<int>(k_total),
                static_cast<int>(im2row_batch_stride), 0, wei_base, ld_wei, 0,
                kernel_dst_base, ld_dst, dst_batch_stride, 0, bias_base, 0);
    }

    parallel(num_threads, [&](int ithr, int nthr) {
        unsigned int row_start = 0;
        unsigned int row_end = window_size.get_size(0);
        unsigned int col_start = 0;
        unsigned int col_end = window_size.get_size(1);

        unsigned int thread_row = ithr;
        unsigned int thread_col = 0;
        if (col_parts > 1) {
            balance2D(nthr, ithr, window_size.get_size(0), row_start, row_end,
                    window_size.get_size(1), col_start, col_end, col_parts);
            thread_row = ithr % row_parts;
            thread_col = ithr / row_parts;
        } else {
            balance211(window_size.get_size(0), nthr, ithr, row_start, row_end);
        }

        kai::ops::ndcoord_t thread_locator {{thread_row, row_parts},
                {thread_col, col_parts}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
        kai::ops::ndcoord_t win {{row_start, row_end - row_start},
                {col_start, col_end - col_start}, {0, window_size.get_size(2)},
                {0, window_size.get_size(3)}, {0, window_size.get_size(4)},
                {0, window_size.get_size(5)}};

        kernel->execute(win, thread_locator, ithr);
    });

    if (!pd()->dst_channels_last_) {
        scatter_channels_last_dst(
                *pd(), kernel_dst_base, dst_base, num_threads);
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
