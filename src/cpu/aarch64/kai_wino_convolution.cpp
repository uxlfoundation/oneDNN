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

#include "cpu/aarch64/kai_wino_convolution.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "kai/ops/conv/winograd.hpp"
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

} // namespace

bool kai_wino_convolution_fwd_t::pd_t::set_default_formats() {
    using namespace format_tag;
    return set_default_formats_common(nhwc, hwio, nhwc);
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_wino_convolution_fwd_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(*wino_impl_->gemm_args, cfg_.get(),
            src_md()->data_type, weights_md()->data_type, dst_md()->data_type);
}

status_t kai_wino_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;
    MAYBE_UNUSED(engine);

    const auto src_dt = src_md()->data_type;
    const auto wei_dt = weights_md()->data_type;
    const auto dst_dt = dst_md()->data_type;

    bool ok = true && is_fwd()
            && utils::one_of(desc()->alg_kind, alg_kind::convolution_auto,
                    alg_kind::convolution_winograd)
            && set_default_alg_kind(alg_kind::convolution_winograd)
            && ndims() == 4 && !with_groups() && KSH() == 1 && KSW() == 1
            && KDH() == 0 && KDW() == 0
            && ((src_dt == data_type::f32 && wei_dt == data_type::f32
                        && dst_dt == data_type::f32)
                    || (src_dt == data_type::f16 && wei_dt == data_type::f16
                            && dst_dt == data_type::f16))
            && !has_zero_dim_memory() && !has_runtime_dims_or_strides()
            && attr()->has_default_values(primitive_mask_t::fpmath_mode
                            | primitive_mask_t::accumulation_mode,
                    dst_dt)
            && set_default_formats()
            && attr_.set_default_formats(dst_md()) == status::success;
    if (!ok) return status::unimplemented;

    VDISPATCH_CONV(utils::everyone_is(3, KH(), KW()),
            "unsupported KAI Winograd kernel size");
    VDISPATCH_CONV(padT() <= 1 && padL() <= 1 && padB() <= 1 && padR() <= 1,
            "unsupported KAI Winograd padding");

    const auto src_tag
            = memory_desc_matches_one_of_tag(src_md_, format_tag::nhwc);
    const auto dst_tag
            = memory_desc_matches_one_of_tag(dst_md_, format_tag::nhwc);
    const auto wei_tag
            = memory_desc_matches_one_of_tag(weights_md_, format_tag::hwio);

    VDISPATCH_CONV(src_md_.format_kind == format_kind::any
                    || src_tag == format_tag::nhwc,
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dst_md_.format_kind == format_kind::any
                    || dst_tag == format_tag::nhwc,
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(weights_md_.format_kind == format_kind::any
                    || wei_tag == format_tag::hwio,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    if (desc()->alg_kind == alg_kind::convolution_auto
            && (IH() > 112 || IW() > 112 || IC() < 64 || OC() < 64
                    || dnnl_get_max_threads() > 28)) {
        return status::unimplemented;
    }

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);

    cfg_ = std::make_shared<kai::ops::GemmConfig>();
    conv_args_ = std::make_shared<kai::ops::ConvolutionArgs>(MB(),
            kai::ops::Shape2D {static_cast<unsigned int>(IH()),
                    static_cast<unsigned int>(IW())},
            static_cast<unsigned int>(IC()), static_cast<unsigned int>(padT()),
            static_cast<unsigned int>(padL()),
            kai::ops::Shape2D {static_cast<unsigned int>(OH()),
                    static_cast<unsigned int>(OW())},
            static_cast<unsigned int>(OC()),
            kai::ops::Shape2D {static_cast<unsigned int>(KH()),
                    static_cast<unsigned int>(KW())});
    wino_impl_ = std::make_shared<kai::ops::winograd::WinogradImpl>();

    const int max_threads = dnnl_get_current_num_threads();
    kai::ops::winograd::WinogradConfig winograd_cfg;
    bool supported = false;
    if (src_dt == data_type::f32) {
        supported = kai::ops::winograd::get_implementation<float>(*wino_impl_,
                get_cpu_info(), *conv_args_, max_threads, false, &winograd_cfg,
                cfg_.get());
    } else if (src_dt == data_type::f16) {
        supported = kai::ops::winograd::get_implementation<__fp16>(*wino_impl_,
                get_cpu_info(), *conv_args_, max_threads, false, &winograd_cfg,
                cfg_.get());
    }
    VDISPATCH_CONV(supported, "unsupported KAI Winograd configuration");

    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    cfg_ = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());
    run_weight_reorder_ = kernel->B_is_pretransposed();

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << kernel->get_config().filter
                  << std::endl;
    }

    working_size_ = std::max({kernel->get_working_size(),
            wino_impl_->input_transform->get_working_space_size(
                    *conv_args_, max_threads),
            wino_impl_->output_transform->get_working_space_size(
                    *conv_args_, max_threads)});

    const auto &wds = wino_impl_->winograd_spec;
    auto scratchpad = scratchpad_registry().registrar();

    if (working_size_ != 0) {
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                working_size_, 1, 64, 64);
    }

    scratchpad.book(memory_tracking::names::key_conv_gemm_col,
            wds.input_matrix_size_bytes, 1, 64, 64);
    scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
            wds.output_matrix_size_bytes, 1, 64, 64);
    scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
            wds.weight_matrix_size_bytes, 1, 64, 64);

    if (run_weight_reorder_) {
        scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                kernel->get_B_pretransposed_array_size(), 1, 64, 64);
    }

    return status::success;
}

status_t kai_wino_convolution_fwd_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
    return status::success;
}

template <typename data_t>
status_t execute_wino_forward(const kai_wino_convolution_fwd_t::pd_t *pd, const exec_ctx_t &ctx) {
    std::unique_ptr<kai::ops::IGemmCommon> kernel = pd->create_kai_gemm();
    if (!kernel) return status::runtime_error;

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto &wds = pd->wino_impl_->winograd_spec;

    const auto *src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto *dst_base = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    const auto *bias_base = pd->with_bias()
            ? CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS)
            : nullptr;

    auto *wino_input
            = scratchpad.get<data_t>(memory_tracking::names::key_conv_gemm_col);
    auto *wino_output = scratchpad.get<data_t>(
            memory_tracking::names::key_gemm_tmp_buffer);
    auto *wino_weights = scratchpad.get<data_t>(
            memory_tracking::names::key_conv_permuted_weights);
    void *pretransposed_weights = pd->run_weight_reorder_
            ? scratchpad.get<void>(memory_tracking::names::key_matmul_wei_trans)
            : nullptr;
    void *working_space = pd->working_size_ != 0
            ? scratchpad.get<void>(
                      memory_tracking::names::key_gemm_asm_tmp_buffer)
            : nullptr;
    const kai::ops::ndrange_t window_size = kernel->get_window_size();
    const int num_windows = static_cast<int>(window_size.total_size());
    int num_threads = std::max(
            1, std::min(num_windows, dnnl_get_current_num_threads()));

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

    constexpr int n_dim = 0;
    constexpr int c_dim = 1;
    constexpr int h_dim = 2;
    constexpr int w_dim = 3;

    const size_t src_batch_stride = static_cast<size_t>(
            pd->src_md()->format_desc.blocking.strides[n_dim]);
    const size_t src_row_stride = static_cast<size_t>(
            pd->src_md()->format_desc.blocking.strides[h_dim]);
    const size_t src_col_stride = static_cast<size_t>(
            pd->src_md()->format_desc.blocking.strides[w_dim]);

    const size_t wei_row_stride = static_cast<size_t>(
            pd->weights_md()->format_desc.blocking.strides[h_dim]);
    const size_t wei_col_stride = static_cast<size_t>(
            pd->weights_md()->format_desc.blocking.strides[w_dim]);
    const size_t wei_ic_stride = static_cast<size_t>(
            pd->weights_md()->format_desc.blocking.strides[c_dim]);

    const size_t dst_batch_stride = static_cast<size_t>(
            pd->dst_md()->format_desc.blocking.strides[n_dim]);
    const size_t dst_row_stride = static_cast<size_t>(
            pd->dst_md()->format_desc.blocking.strides[h_dim]);
    const size_t dst_col_stride = static_cast<size_t>(
            pd->dst_md()->format_desc.blocking.strides[w_dim]);

    parallel(num_threads, [&](int ithr, int nthr) {
        pd->wino_impl_->weight_transform->execute(*pd->conv_args_, raw_wei,
                wei_row_stride, wei_col_stride, wei_ic_stride, wino_weights,
                wds, ithr, nthr);
    });

    if (pd->run_weight_reorder_) {
        const unsigned int wsize = kernel->get_B_pretranspose_window_size();
        parallel(num_threads, [&](int ithr, int nthr) {
            const unsigned int start = (ithr * wsize) / nthr;
            const unsigned int end = ((ithr + 1) * wsize) / nthr;
            if (start < end) {
                kernel->pretranspose_B_array_part_generic(pretransposed_weights,
                        wino_weights, static_cast<int>(wds.weight_ld_row),
                        static_cast<int>(wds.weight_ld_matrix), false, start,
                        end);
            }
        });
        kernel->set_pretransposed_B_data(pretransposed_weights);
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(working_space);
    }

    kernel->set_arrays_generic(wino_input, static_cast<int>(wds.input_ld_row),
            static_cast<int>(wds.input_ld_batch),
            static_cast<int>(wds.input_ld_matrix), wino_weights,
            static_cast<int>(wds.weight_ld_row),
            static_cast<int>(wds.weight_ld_matrix), wino_output,
            static_cast<int>(wds.output_ld_row),
            static_cast<int>(wds.output_ld_batch),
            static_cast<int>(wds.output_ld_matrix), nullptr, 0);

    parallel(num_threads, [&](int ithr, int nthr) {
        pd->wino_impl_->input_transform->execute(*pd->conv_args_, src_base,
                src_batch_stride, src_row_stride, src_col_stride, wino_input,
                wds, working_space, ithr, nthr);
    });

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

    parallel(num_threads, [&](int ithr, int nthr) {
        pd->wino_impl_->output_transform->execute(*pd->conv_args_,
                wino_output, wds, bias_base, dst_base, dst_batch_stride,
                dst_row_stride, dst_col_stride, working_space, ithr, nthr);
    });

    return status::success;
}

status_t kai_wino_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->src_md()->data_type == data_type::f32)
        return execute_wino_forward<float>(pd(), ctx);
    if (pd()->src_md()->data_type == data_type::f16)
        return execute_wino_forward<__fp16>(pd(), ctx);
    return status::runtime_error;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
