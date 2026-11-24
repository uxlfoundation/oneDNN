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

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_utils.hpp"
#include "cpu/aarch64/kai_wino_utils.hpp"

#include "kai/ops/conv/winograd.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

DEFINE_PD_CREATE(kai_wino_convolution_fwd_t::pd_t)

using namespace data_type;
using namespace kai_utils;

namespace {

bool dense_layout(const memory_desc_t &md, format_tag_t tag) {
    const memory_desc_wrapper mdw(md);
    return mdw.is_plain() && mdw.is_dense() && mdw.offset0() == 0
            && mdw.matches_tag(tag);
}

double kernel_execute_work(const kai::ops::GemmArgs &args) {
    return static_cast<double>(args._Msize) * args._Nsize * args._Ksize
            * args._Ksections * args._nbatches * args._nmulti;
}

} // namespace

const kai::ops::GemmArgs &kai_wino_convolution_fwd_t::pd_t::gemm_args() const {
    return *wino_impl_->gemm_args;
}

kai::ops::ConvolutionArgs
kai_wino_convolution_fwd_t::pd_t::convolution_args() const {
    return {static_cast<unsigned int>(MB()),
            {static_cast<unsigned int>(IH()), static_cast<unsigned int>(IW())},
            static_cast<unsigned int>(IC()), static_cast<unsigned int>(padT()),
            static_cast<unsigned int>(padL()),
            {static_cast<unsigned int>(OH()), static_cast<unsigned int>(OW())},
            static_cast<unsigned int>(OC()),
            {static_cast<unsigned int>(KH()), static_cast<unsigned int>(KW())}};
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_wino_convolution_fwd_t::pd_t::create_kai_gemm() const {
    return kai_utils::create_kai_gemm(gemm_args(), src_md()->data_type,
            weights_md()->data_type, dst_md()->data_type);
}

status_t kai_wino_convolution_fwd_t::pd_t::init(const engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;
    const bool is_auto = desc()->alg_kind == alg_kind::convolution_auto;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_winograd),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(ndims() == 4, VERBOSE_BAD_NDIMS, "src", ndims());
    VDISPATCH_CONV(!with_groups(), VERBOSE_UNSUPPORTED_FEATURE, "groups");
    VDISPATCH_CONV(KSH() == 1 && KSW() == 1, "only supports unit strides");
    VDISPATCH_CONV(KDH() == 0 && KDW() == 0, "only supports undilated kernels");
    const auto src_dt = src_md()->data_type;
    VDISPATCH_CONV(utils::one_of(src_dt, f32, f16)
                    && utils::everyone_is(src_dt, weights_md()->data_type,
                            dst_md()->data_type)
                    && (!with_bias()
                            || invariant_bia_md()->data_type == src_dt),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_CONV(src_dt != f16 || mayiuse_f16(), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    const auto supported_attrs = primitive_mask_t::fpmath_mode
            | primitive_mask_t::accumulation_mode | primitive_mask_t::post_ops;
    VDISPATCH_CONV(
            attr()->has_default_values(supported_attrs, dst_md()->data_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(utils::one_of(attr()->acc_mode_, accumulation_mode::strict,
                           accumulation_mode::relaxed, accumulation_mode::any,
                           accumulation_mode::f32)
                    || (src_dt == f16
                            && attr()->acc_mode_ == accumulation_mode::f16),
            "accumulation mode is not valid for the data type combination");

    // Resolve layouts once; leave weights `any` until GEMM selection determines
    // their final Winograd fixed format or plain HWIO layout.
    if (src_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md_, format_tag::nhwc));
    if (dst_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md_, format_tag::nhwc));
    if (with_bias() && bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    VDISPATCH_CONV_SC(attr_.set_default_formats(dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(dense_layout(src_md_, format_tag::nhwc)
                    || dense_layout(src_md_, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(dense_layout(dst_md_, format_tag::nhwc)
                    || dense_layout(dst_md_, format_tag::nchw),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(weights_md_.format_kind == format_kind::any
                    || dense_layout(weights_md_, format_tag::hwio),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_CONV(!with_bias() || dense_layout(bias_md_, format_tag::x),
            VERBOSE_UNSUPPORTED_TAG_S, "bias");

    VDISPATCH_CONV(
            utils::everyone_is(3, KH(), KW()), "only supports 3x3 kernels");
    VDISPATCH_CONV(padT() <= 1 && padL() <= 1 && padB() <= 1 && padR() <= 1,
            "only supports padding <= 1");
    VDISPATCH_CONV(!is_auto
                    || (IH() <= 112 && IW() <= 112 && IC() >= 64 && OC() >= 64
                            && dnnl_get_max_threads() <= 28),
            VERBOSE_IMPL_HEURISTIC_FAIL, "auto policy");
    VDISPATCH_CONV(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md()));

    const auto conv_args = convolution_args();
    wino_impl_ = std::make_shared<kai::ops::winograd::WinogradImpl>();
    // Keep f32 transforms and GEMM in f32 even when bf16 fpmath is allowed.
    const kai::ops::winograd::WinogradConfig winograd_cfg;
    const int max_threads = dnnl_get_current_num_threads();
    const bool supported = src_dt == f32
            ? kai::ops::winograd::get_implementation<float>(*wino_impl_,
                      get_cpu_info(), conv_args, max_threads, false,
                      &winograd_cfg, nullptr)
            : kai::ops::winograd::get_implementation<__fp16>(*wino_impl_,
                      get_cpu_info(), conv_args, max_threads,
                      use_fast_mode(*src_md(), *attr()), &winograd_cfg,
                      nullptr);
    VDISPATCH_CONV(supported, "unsupported configuration");

    auto &args = *wino_impl_->gemm_args;
    args._maxthreads = threads_for_kernel_execute(
            kernel_execute_work(args), args._maxthreads);
    args._fixed_format = weights_md_.format_kind == format_kind::any;
    std::unique_ptr<kai::ops::IGemmCommon> kernel = create_kai_gemm();
    const bool fixed_format_failed = args._fixed_format
            && (!kernel
                    || !is_fixed_format(kernel->get_config().weight_format));
    if (fixed_format_failed) {
        args._fixed_format = false;
        kernel = create_kai_gemm();
    }
    VDISPATCH_CONV(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    // Inspect the selected format without feeding the returned configuration
    // back into GemmArgs. Recreate the same GEMM in execute, as in KAI conv.
    const auto kernel_cfg = kernel->get_config();
    if (args._fixed_format) {
        kai_wino_utils::init_packed_weights_desc(weights_md_,
                kernel_cfg.weight_format, static_cast<int>(KH()),
                wino_impl_->weight_transform->get_transformed_tile_rows(),
                static_cast<int>(IC()), static_cast<int>(OC()));
    } else if (weights_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md_, format_tag::hwio));
    }

    auto scratchpad = scratchpad_registry().registrar();
    if (!memory_desc_matches_tag(*src_md(), format_tag::nhwc)) {
        memory_desc_t tmp_src_md;
        CHECK(memory_desc_init_by_tag(tmp_src_md, src_md()->ndims,
                src_md()->dims, src_dt, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(src_reorder_pd_, engine,
                                  src_md(), &tmp_src_md),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "src reorder");
        scratchpad.book<char>(memory_tracking::names::key_conv_ncsp_src,
                memory_desc_wrapper(tmp_src_md).size());
        scratchpad.book(memory_tracking::names::key_nested_multiple
                        + attr_.post_ops_.len(),
                src_reorder_pd_->scratchpad_registry());
    }
    if (!memory_desc_matches_tag(*dst_md(), format_tag::nhwc)) {
        memory_desc_t tmp_dst_md;
        CHECK(memory_desc_init_by_tag(tmp_dst_md, dst_md()->ndims,
                dst_md()->dims, dst_md()->data_type, format_tag::nhwc));
        VDISPATCH_CONV_SC(reorder_primitive_desc_create(dst_reorder_pd_, engine,
                                  &tmp_dst_md, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");
        scratchpad.book<char>(memory_tracking::names::key_conv_ncsp_dst,
                memory_desc_wrapper(tmp_dst_md).size());
        scratchpad.book(memory_tracking::names::key_nested,
                dst_reorder_pd_->scratchpad_registry());
    }

    const size_t working_size = std::max({kernel->get_working_size(),
            wino_impl_->input_transform->get_working_space_size(
                    conv_args, args._maxthreads),
            wino_impl_->output_transform->get_working_space_size(
                    conv_args, args._maxthreads)});
    if (working_size != 0)
        scratchpad.book<char>(memory_tracking::names::key_gemm_asm_tmp_buffer,
                working_size, PAGE_4K);

    const auto &wds = wino_impl_->winograd_spec;
    scratchpad.book<char>(memory_tracking::names::key_conv_gemm_col,
            wds.input_matrix_size_bytes);
    scratchpad.book<char>(memory_tracking::names::key_gemm_tmp_buffer,
            wds.output_matrix_size_bytes);
    if (!args._fixed_format)
        scratchpad.book<char>(memory_tracking::names::key_conv_permuted_weights,
                wds.weight_matrix_size_bytes);
    if (!args._fixed_format && kernel->B_is_pretransposed())
        scratchpad.book<char>(memory_tracking::names::key_matmul_wei_trans,
                kernel->get_B_pretransposed_array_size());
    if (post_ops.has_sum())
        scratchpad.book<char>(memory_tracking::names::key_generic_acc,
                memory_desc_wrapper(dst_md()).size());
    post_ops.init_scratchpad(scratchpad);
    return status::success;
}

status_t kai_wino_convolution_fwd_t::init(engine_t *engine) {
    if (pd()->src_reorder_pd_)
        CHECK(pd()->src_reorder_pd_->create_primitive(src_reorder_, engine));
    if (pd()->dst_reorder_pd_)
        CHECK(pd()->dst_reorder_pd_->create_primitive(dst_reorder_, engine));
    post_ops_ = pd()->post_ops;
    CHECK(post_ops_.init_primitives(engine));
    return status::success;
}

status_t kai_wino_convolution_fwd_t::execute_reorder(const exec_ctx_t &ctx,
        const primitive_t &reorder, const void *src, void *dst,
        memory_tracking::key_t nested_key) {
    auto *engine = ctx.stream()->engine();
    std::unique_ptr<memory_t, memory_deleter_t> src_mem(new memory_t(engine,
            reorder.pd()->src_md(), use_runtime_ptr, const_cast<void *>(src)));
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
            new memory_t(engine, reorder.pd()->dst_md(), use_runtime_ptr, dst));
    exec_args_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = {src_mem.get(), true};
    reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
    exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));
    auto *nested_grantor = memory_tracking::create_nested_grantor(
            ctx.get_scratchpad_grantor(), nested_key,
            reorder.pd()->scratchpad_registry());
    reorder_ctx.set_scratchpad_grantor(nested_grantor);
    return reorder.execute(reorder_ctx);
}

status_t kai_wino_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    // Recreate the GEMM with the descriptor's unchanged arguments so that its
    // selected weight format and workspace match initialization.
    auto kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;
    const auto &args = pd()->gemm_args();
    const auto conv_args = pd()->convolution_args();
    const auto &wds = pd()->wino_impl_->winograd_spec;
    const int num_threads
            = std::min(args._maxthreads, dnnl_get_current_num_threads());
    const auto window_size = kernel->get_window_size();
    const auto thread_partition
            = make_thread_partition(num_threads, window_size);
    kernel->set_nthreads(thread_partition.nthr);

    if (get_verbose(verbose_t::profile_externals))
        std::cout << "profile_externals: " << kernel->get_config().filter
                  << std::endl;

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const void *src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    if (src_reorder_) {
        auto *tmp_src = scratchpad.get<void>(
                memory_tracking::names::key_conv_ncsp_src);
        CHECK(execute_reorder(ctx, *src_reorder_, src_base, tmp_src,
                memory_tracking::names::key_nested_multiple
                        + pd()->attr()->post_ops_.len()));
        src_base = tmp_src;
    }
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto *dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const auto *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;
    void *kernel_dst_base = dst_reorder_
            ? scratchpad.get<void>(memory_tracking::names::key_conv_ncsp_dst)
            : post_ops_.has_sum()
            ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
            : dst_base;
    auto *wino_input
            = scratchpad.get<void>(memory_tracking::names::key_conv_gemm_col);
    auto *wino_output
            = scratchpad.get<void>(memory_tracking::names::key_gemm_tmp_buffer);
    void *working_space = scratchpad.get<void>(
            memory_tracking::names::key_gemm_asm_tmp_buffer);

    const void *wei_base = raw_wei;
    if (!args._fixed_format) {
        auto *wino_weights = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
        const auto &strides = pd()->weights_md()->format_desc.blocking.strides;
        parallel(thread_partition.team_nthr, [&](int ithr, int nthr) {
            pd()->wino_impl_->weight_transform->execute(conv_args, raw_wei,
                    strides[2], strides[3], strides[1], wino_weights, wds, ithr,
                    nthr);
        });
        wei_base = wino_weights;
        if (kernel->B_is_pretransposed()) {
            auto *packed_weights = scratchpad.get<void>(
                    memory_tracking::names::key_matmul_wei_trans);
            parallel_pretranspose_B_array(*kernel, packed_weights, wino_weights,
                    wds.weight_ld_row, wds.weight_ld_matrix, false,
                    thread_partition.team_nthr);
            wei_base = packed_weights;
        }
    }
    if (kernel->get_working_size() != 0)
        kernel->set_working_space(working_space);

    const int ld_wei = args._fixed_format
            ? kai_wino_utils::packed_ld(*pd()->weights_md())
            : static_cast<int>(wds.weight_ld_row);
    const int wei_multi_stride = args._fixed_format
            ? kai_wino_utils::packed_multi_stride(*pd()->weights_md())
            : static_cast<int>(wds.weight_ld_matrix);
    kernel->set_arrays_generic(wino_input, wds.input_ld_row, wds.input_ld_batch,
            wds.input_ld_matrix, wei_base, ld_wei, wei_multi_stride,
            wino_output, wds.output_ld_row, wds.output_ld_batch,
            wds.output_ld_matrix, nullptr, 0);

    const auto *src_md
            = src_reorder_ ? pd()->src_reorder_pd_->dst_md() : pd()->src_md();
    const auto &src_strides = src_md->format_desc.blocking.strides;
    parallel(thread_partition.team_nthr, [&](int ithr, int nthr) {
        pd()->wino_impl_->input_transform->execute(conv_args, src_base,
                src_strides[0], src_strides[2], src_strides[3], wino_input, wds,
                working_space, ithr, nthr);
    });
    parallel_execute(*kernel, window_size, thread_partition);

    const auto *dst_md
            = dst_reorder_ ? pd()->dst_reorder_pd_->src_md() : pd()->dst_md();
    const auto &dst_strides = dst_md->format_desc.blocking.strides;
    parallel(thread_partition.team_nthr, [&](int ithr, int nthr) {
        pd()->wino_impl_->output_transform->execute(conv_args, wino_output, wds,
                bias_base, kernel_dst_base, dst_strides[0], dst_strides[2],
                dst_strides[3], working_space, ithr, nthr);
    });

    void *post_ops_src = kernel_dst_base;
    if (dst_reorder_) {
        post_ops_src = post_ops_.has_sum()
                ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
                : dst_base;
        CHECK(execute_reorder(ctx, *dst_reorder_, kernel_dst_base, post_ops_src,
                memory_tracking::names::key_nested));
    }
    if (post_ops_.len() > 0)
        CHECK(post_ops_.execute(
                ctx, post_ops_src, post_ops_.has_sum() ? dst_base : nullptr));
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
