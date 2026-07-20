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
#include "cpu/aarch64/kai_wino_utils.hpp"

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
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
    if (fixed_format_) {
        if (src_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_md_, nhwc));
        if (dst_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_md_, nhwc));
        if (with_bias() && bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, x));
        return true;
    }
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

    fixed_format_ = weights_md_.format_kind == format_kind::any;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_winograd),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_CONV(KSH() == 1 && KSW() == 1, "only supports unit strides");
    VDISPATCH_CONV(KDH() == 0 && KDW() == 0, "only supports undilated kernels");
    VDISPATCH_CONV(src_dt == data_type::f32 && wei_dt == data_type::f32
                    && dst_dt == data_type::f32,
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_CONV(attr()->has_default_values(primitive_mask_t::fpmath_mode
                                   | primitive_mask_t::accumulation_mode
                                   | primitive_mask_t::post_ops,
                           dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_CONV(
            utils::everyone_is(3, KH(), KW()), "only supports 3x3 kernels");
    VDISPATCH_CONV(padT() <= 1 && padL() <= 1 && padB() <= 1 && padR() <= 1,
            "only supports padding <= 1");

    const auto src_tag = memory_desc_matches_one_of_tag(
            src_md_, format_tag::nhwc, format_tag::nchw);
    const auto dst_tag = memory_desc_matches_one_of_tag(
            dst_md_, format_tag::nhwc, format_tag::nchw);
    const auto wei_tag
            = memory_desc_matches_one_of_tag(weights_md_, format_tag::hwio);

    VDISPATCH_CONV(src_md_.format_kind == format_kind::any
                    || utils::one_of(
                            src_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dst_md_.format_kind == format_kind::any
                    || utils::one_of(
                            dst_tag, format_tag::nhwc, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(weights_md_.format_kind == format_kind::any
                    || wei_tag == format_tag::hwio,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    VDISPATCH_CONV(desc()->alg_kind != alg_kind::convolution_auto
                    || (IH() <= 112 && IW() <= 112 && IC() >= 64 && OC() >= 64
                            && dnnl_get_max_threads() <= 28),
            VERBOSE_IMPL_HEURISTIC_FAIL, "auto policy");

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);
    const bool fast_mode = kai_utils::use_fast_mode(*src_md(), *attr());
    CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md()));
    has_post_ops_fallback_ = attr_.post_ops_.len() > 0;

    src_channels_last_ = src_tag == format_tag::nhwc;
    dst_channels_last_ = dst_tag == format_tag::nhwc;
    use_src_reorder_ = !src_channels_last_;
    use_dst_reorder_ = !dst_channels_last_;
    VDISPATCH_CONV(!(use_dst_reorder_ && post_ops.has_sum()),
            "sum post-op is unsupported with staged dst");

    cfg_ = std::make_shared<kai::ops::GemmConfig>();
    if (fixed_format_) cfg_->weight_format = kai::ops::WeightFormat::ANY;
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
                get_cpu_info(), *conv_args_, max_threads, fast_mode,
                &winograd_cfg, cfg_.get());
    } else if (src_dt == data_type::f16) {
        supported = kai::ops::winograd::get_implementation<__fp16>(*wino_impl_,
                get_cpu_info(), *conv_args_, max_threads, fast_mode,
                &winograd_cfg, cfg_.get());
    }
    VDISPATCH_CONV(supported, "unsupported configuration");
    wino_impl_->gemm_args->_fixed_format = fixed_format_;

    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    const bool fixed_format_failed = fixed_format_
            && (!kernel
                    || !kai_utils::is_fixed_format(
                            kernel->get_config().weight_format));
    if (fixed_format_failed) {
        fixed_format_ = false;
        CHECK(memory_desc_init_by_tag(weights_md_, format_tag::hwio));
        cfg_ = std::make_shared<kai::ops::GemmConfig>();
        wino_impl_->gemm_args->_fixed_format = false;
        kernel = create_kai_gemm();
    }
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    cfg_ = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());
    // Some generated filters do not match the impl list, so it ends up rejecting
    // the second time around. This could be removed if this is fixed in KleidiAI
    cfg_->filter.clear();

    if (fixed_format_) {
        const auto &wt = *wino_impl_->weight_transform;
        kai_wino_utils::init_packed_weights_desc(weights_md_,
                cfg_->weight_format, static_cast<int>(KH()),
                static_cast<int>(wt.get_transformed_tile_rows()),
                static_cast<int>(IC()), static_cast<int>(OC()));
    }

    run_weight_reorder_ = !fixed_format_ && kernel->B_is_pretransposed();

    working_size_ = std::max({kernel->get_working_size(),
            wino_impl_->input_transform->get_working_space_size(
                    *conv_args_, max_threads),
            wino_impl_->output_transform->get_working_space_size(
                    *conv_args_, max_threads)});

    const auto &wds = wino_impl_->winograd_spec;
    auto scratchpad = scratchpad_registry().registrar();

    if (use_src_reorder_) {
        CHECK(memory_desc_init_by_tag(tmp_src_md_, src_md()->ndims,
                src_md()->dims, src_md()->data_type, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(src_reorder_pd_, engine,
                                  src_md(), &tmp_src_md_),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "src reorder");
        const memory_desc_wrapper tmp_src_d(&tmp_src_md_);
        scratchpad.book(memory_tracking::names::key_conv_ncsp_src,
                tmp_src_d.size(), 1, 64, 64);
        scratchpad.book(memory_tracking::names::key_nested_multiple,
                src_reorder_pd_->scratchpad_registry());
    }

    if (use_dst_reorder_) {
        CHECK(memory_desc_init_by_tag(tmp_dst_md_, dst_md()->ndims,
                dst_md()->dims, dst_md()->data_type, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(dst_reorder_pd_, engine,
                                  &tmp_dst_md_, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");
        const memory_desc_wrapper tmp_dst_d(&tmp_dst_md_);
        scratchpad.book(memory_tracking::names::key_conv_ncsp_dst,
                tmp_dst_d.size(), 1, 64, 64);
        scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
                dst_reorder_pd_->scratchpad_registry());
    }

    if (working_size_ != 0) {
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                working_size_, 1, 64, 64);
    }

    scratchpad.book(memory_tracking::names::key_conv_gemm_col,
            wds.input_matrix_size_bytes, 1, 64, 64);
    scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
            wds.output_matrix_size_bytes, 1, 64, 64);
    if (!fixed_format_) {
        scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                wds.weight_matrix_size_bytes, 1, 64, 64);
    }

    if (run_weight_reorder_) {
        scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                kernel->get_B_pretransposed_array_size(), 1, 64, 64);
    }

    if (post_ops.has_sum()) {
        const memory_desc_wrapper dst_d(dst_md());
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.size(),
                1, 64, 64);
    }
    post_ops.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_wino_convolution_fwd_t::init(engine_t *engine) {
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_, engine));
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_, engine));
    return status::success;
}

template <typename data_t>
status_t execute_wino_forward(const kai_wino_convolution_fwd_t::pd_t *pd,
        const exec_ctx_t &ctx, const std::shared_ptr<primitive_t> &src_reorder,
        const std::shared_ptr<primitive_t> &dst_reorder) {
    std::unique_ptr<kai::ops::IGemmCommon> kernel = pd->create_kai_gemm();
    if (!kernel) return status::runtime_error;

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << kernel->get_config().filter
                  << std::endl;
    }

    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto &wds = pd->wino_impl_->winograd_spec;

    const auto *src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto *dst_base = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    const auto *bias_base = pd->with_bias()
            ? CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS)
            : nullptr;

    if (pd->use_src_reorder_) {
        if (!src_reorder) return status::runtime_error;

        auto *tmp_src_base = scratchpad.get<data_t>(
                memory_tracking::names::key_conv_ncsp_src);
        auto *engine = ctx.stream()->engine();
        std::unique_ptr<memory_t, memory_deleter_t> src_mem(new memory_t(engine,
                pd->src_md(), use_runtime_ptr, const_cast<data_t *>(src_base)));
        std::unique_ptr<memory_t, memory_deleter_t> tmp_src_mem(new memory_t(
                engine, &pd->tmp_src_md_, use_runtime_ptr, tmp_src_base));

        exec_args_t reorder_args;
        reorder_args[DNNL_ARG_SRC] = {src_mem.get(), true};
        reorder_args[DNNL_ARG_DST] = {tmp_src_mem.get(), false};
        exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

        auto *nested_grantor = memory_tracking::create_nested_grantor(
                ctx.get_scratchpad_grantor(),
                memory_tracking::names::key_nested_multiple,
                src_reorder->pd()->scratchpad_registry());
        reorder_ctx.set_scratchpad_grantor(nested_grantor);
        CHECK(src_reorder->execute(reorder_ctx));
        src_base = tmp_src_base;
    }

    auto *kernel_dst_base = pd->use_dst_reorder_
            ? scratchpad.get<data_t>(memory_tracking::names::key_conv_ncsp_dst)
            : dst_base;
    auto *post_ops_src = pd->post_ops.has_sum()
            ? scratchpad.get<data_t>(memory_tracking::names::key_generic_acc)
            : kernel_dst_base;

    auto *wino_input
            = scratchpad.get<data_t>(memory_tracking::names::key_conv_gemm_col);
    auto *wino_output = scratchpad.get<data_t>(
            memory_tracking::names::key_gemm_tmp_buffer);
    data_t *scratch_wino_weights = pd->fixed_format_
            ? nullptr
            : scratchpad.get<data_t>(
                      memory_tracking::names::key_conv_permuted_weights);
    const data_t *wino_weights
            = pd->fixed_format_ ? raw_wei : scratch_wino_weights;
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

    const memory_desc_t *kernel_src_md
            = pd->use_src_reorder_ ? &pd->tmp_src_md_ : pd->src_md();
    const memory_desc_t *kernel_dst_md
            = pd->use_dst_reorder_ ? &pd->tmp_dst_md_ : pd->dst_md();

    const size_t src_batch_stride = static_cast<size_t>(
            kernel_src_md->format_desc.blocking.strides[n_dim]);
    const size_t src_row_stride = static_cast<size_t>(
            kernel_src_md->format_desc.blocking.strides[h_dim]);
    const size_t src_col_stride = static_cast<size_t>(
            kernel_src_md->format_desc.blocking.strides[w_dim]);

    size_t wei_row_stride = 0;
    size_t wei_col_stride = 0;
    size_t wei_ic_stride = 0;
    if (!pd->fixed_format_) {
        wei_row_stride = static_cast<size_t>(
                pd->weights_md()->format_desc.blocking.strides[h_dim]);
        wei_col_stride = static_cast<size_t>(
                pd->weights_md()->format_desc.blocking.strides[w_dim]);
        wei_ic_stride = static_cast<size_t>(
                pd->weights_md()->format_desc.blocking.strides[c_dim]);
    }

    const size_t dst_batch_stride = static_cast<size_t>(
            kernel_dst_md->format_desc.blocking.strides[n_dim]);
    const size_t dst_row_stride = static_cast<size_t>(
            kernel_dst_md->format_desc.blocking.strides[h_dim]);
    const size_t dst_col_stride = static_cast<size_t>(
            kernel_dst_md->format_desc.blocking.strides[w_dim]);

    if (!pd->fixed_format_) {
        parallel(num_threads, [&](int ithr, int nthr) {
            pd->wino_impl_->weight_transform->execute(*pd->conv_args_, raw_wei,
                    wei_row_stride, wei_col_stride, wei_ic_stride,
                    scratch_wino_weights, wds, ithr, nthr);
        });
    }

    if (pd->run_weight_reorder_) {
        const unsigned int wsize = kernel->get_B_pretranspose_window_size();
        parallel(num_threads, [&](int ithr, int nthr) {
            const unsigned int start = (ithr * wsize) / nthr;
            const unsigned int end = ((ithr + 1) * wsize) / nthr;
            if (start < end) {
                kernel->pretranspose_B_array_part_generic(pretransposed_weights,
                        scratch_wino_weights,
                        static_cast<int>(wds.weight_ld_row),
                        static_cast<int>(wds.weight_ld_matrix), false, start,
                        end);
            }
        });
        kernel->set_pretransposed_B_data(pretransposed_weights);
    }

    if (kernel->get_working_size() != 0) {
        kernel->set_working_space(working_space);
    }

    const int weight_ld_row = pd->fixed_format_
            ? kai_wino_utils::packed_ld(*pd->weights_md())
            : static_cast<int>(wds.weight_ld_row);
    const int weight_ld_matrix = pd->fixed_format_
            ? kai_wino_utils::packed_multi_stride(*pd->weights_md())
            : static_cast<int>(wds.weight_ld_matrix);

    kernel->set_arrays_generic(wino_input, static_cast<int>(wds.input_ld_row),
            static_cast<int>(wds.input_ld_batch),
            static_cast<int>(wds.input_ld_matrix), wino_weights, weight_ld_row,
            weight_ld_matrix, wino_output, static_cast<int>(wds.output_ld_row),
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
        pd->wino_impl_->output_transform->execute(*pd->conv_args_, wino_output,
                wds, bias_base, post_ops_src, dst_batch_stride, dst_row_stride,
                dst_col_stride, working_space, ithr, nthr);
    });

    if (pd->use_dst_reorder_) {
        if (!dst_reorder) return status::runtime_error;

        auto *engine = ctx.stream()->engine();
        std::unique_ptr<memory_t, memory_deleter_t> tmp_dst_mem(new memory_t(
                engine, &pd->tmp_dst_md_, use_runtime_ptr, post_ops_src));
        std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
                new memory_t(engine, pd->dst_md(), use_runtime_ptr, dst_base));

        exec_args_t reorder_args;
        reorder_args[DNNL_ARG_SRC] = {tmp_dst_mem.get(), true};
        reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
        exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

        auto *nested_grantor = memory_tracking::create_nested_grantor(
                ctx.get_scratchpad_grantor(),
                memory_tracking::names::key_nested_multiple + 1,
                dst_reorder->pd()->scratchpad_registry());
        reorder_ctx.set_scratchpad_grantor(nested_grantor);
        CHECK(dst_reorder->execute(reorder_ctx));
        post_ops_src = dst_base;
    }

    if (pd->has_post_ops_fallback_) {
        if (pd->post_ops.has_sum())
            CHECK(pd->post_ops.execute(ctx, post_ops_src, dst_base));
        else
            CHECK(pd->post_ops.execute(ctx, post_ops_src));
    }

    return status::success;
}

status_t kai_wino_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->src_md()->data_type == data_type::f32)
        return execute_wino_forward<float>(
                pd(), ctx, src_reorder_, dst_reorder_);
    if (pd()->src_md()->data_type == data_type::f16)
        return execute_wino_forward<__fp16>(
                pd(), ctx, src_reorder_, dst_reorder_);
    return status::runtime_error;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
