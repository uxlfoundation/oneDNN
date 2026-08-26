/*******************************************************************************
* Copyright 2021-2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/kai_matmul.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/gemm_based_common.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"

#include <functional>
#include <memory>
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace data_type;
using namespace kai_utils;

namespace {

bool batch_dims_have_default_order(const memory_desc_wrapper &mdw) {
    assert(mdw.is_blocking_desc());

    if (mdw.ndims() <= 2) return true;

    const auto &dims = mdw.dims();
    const auto ndims = mdw.ndims();
    const auto &strides = mdw.strides();

    dim_t expected_stride = dims[ndims - 1] * dims[ndims - 2];
    for (int i = ndims - 3; i >= 0; --i) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= dims[i];
    }

    return true;
}

bool batch_dims_match(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {
    if (lhs.ndims() != rhs.ndims()) return false;

    for (int i = 0; i < lhs.ndims() - 2; ++i) {
        if (lhs.dims()[i] != rhs.dims()[i]) return false;
    }

    return true;
}

bool batch_dims_are_all_one(const memory_desc_wrapper &mdw) {
    for (int i = 0; i < mdw.ndims() - 2; ++i) {
        if (mdw.dims()[i] != 1) return false;
    }

    return true;
}

int get_innermost_batch_stride(const memory_desc_t *md) {
    return md->ndims > 2 ? md->format_desc.blocking.strides[md->ndims - 3] : 0;
}

double weight_reorder_work(const kai_matmul_t::pd_t &pd) {
    return static_cast<double>(pd.N()) * pd.K() * pd._ag_nmulti;
}

double kernel_execute_work(const kai_matmul_t::pd_t &pd) {
    return static_cast<double>(pd.M()) * pd.N() * pd.K() * pd._ag_nbatches
            * pd._ag_nmulti;
}

enum class ab_reorder_direction { user_to_ab, ab_to_user };

status_t init_ab_reorder(const engine_t *engine, const memory_desc_t *user_md,
        memory_desc_t &ab_md, std::shared_ptr<primitive_desc_t> &reorder_pd,
        memory_tracking::registrar_t &scratchpad,
        memory_tracking::key_t tmp_key, memory_tracking::key_t nested_key,
        ab_reorder_direction direction) {
    // Copy user md for dimensions, then select default dense ab strides.
    ab_md = *user_md;
    CHECK(memory_desc_init_by_strides(ab_md, nullptr));

    const memory_desc_t *reorder_src_md
            = direction == ab_reorder_direction::ab_to_user ? &ab_md : user_md;
    const memory_desc_t *reorder_dst_md
            = direction == ab_reorder_direction::ab_to_user ? user_md : &ab_md;
    CHECK(reorder_primitive_desc_create(
            reorder_pd, engine, reorder_src_md, reorder_dst_md));

    const memory_desc_wrapper ab_d(&ab_md);
    scratchpad.book(tmp_key, ab_d.size(), 1, 64, 64);
    scratchpad.book(nested_key, reorder_pd->scratchpad_registry());

    return status::success;
}

status_t execute_reorder(const exec_ctx_t &ctx,
        const std::shared_ptr<primitive_t> &reorder,
        const memory_desc_t *src_md, const memory_desc_t *dst_md,
        const void *src, void *dst, memory_tracking::key_t nested_key) {
    if (!reorder) return status::runtime_error;

    auto *engine = ctx.stream()->engine();
    std::unique_ptr<memory_t, memory_deleter_t> src_mem(new memory_t(
            engine, src_md, use_runtime_ptr, const_cast<void *>(src)));
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem(
            new memory_t(engine, dst_md, use_runtime_ptr, dst));

    exec_args_t reorder_args;
    reorder_args[DNNL_ARG_SRC] = {src_mem.get(), true};
    reorder_args[DNNL_ARG_DST] = {dst_mem.get(), false};
    exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));

    auto *nested_grantor = memory_tracking::create_nested_grantor(
            ctx.get_scratchpad_grantor(), nested_key,
            reorder->pd()->scratchpad_registry());
    reorder_ctx.set_scratchpad_grantor(nested_grantor);
    return reorder->execute(reorder_ctx);
}

} //namespace

memory_tracking::key_t kai_matmul_t::pd_t::reorder_nested_key(
        nested_reorder_t reorder) const {
    return memory_tracking::names::key_nested_multiple
            + _reorder_nested_scratchpad_base
            + static_cast<memory_tracking::key_t>(reorder);
}

std::unique_ptr<kai::ops::IGemmCommon>
kai_matmul_t::pd_t::create_kai_gemm_dequant(
        const kai::ops::DequantizeFloat &dequant, int max_threads) const {
    return kai_utils::create_kai_gemm_dequant(*_args, _cfg.get(), _kai_src_dt,
            _kai_weights_dt, _kai_dst_dt, dequant, max_threads);
}

std::unique_ptr<kai::ops::IGemmCommon> kai_matmul_t::pd_t::create_kai_gemm(
        int max_threads) const {
    return kai_utils::create_kai_gemm(*_args, _cfg.get(), _kai_src_dt,
            _kai_weights_dt, _kai_dst_dt, max_threads);
}

status_t kai_matmul_t::pd_t::init(const engine_t *engine) {

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper wei_d(weights_md());
    const memory_desc_wrapper dst_d(dst_md());
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    const bool weights_md_was_any
            = weights_md()->format_kind == format_kind::any;
    const bool weights_md_may_be_fixed_format
            = weights_md()->format_kind == format_kind::blocked
            && weights_md()->format_desc.blocking.inner_nblks > 0;

    auto sdt = src_md()->data_type;
    auto wdt = weights_md()->data_type;
    auto ddt = dst_md()->data_type;
    const bool fast_mode = use_fast_mode(*src_md(), *attr());

    _cfg = std::make_shared<kai::ops::GemmConfig>();
    _kai_src_dt = sdt;
    _kai_weights_dt = wdt;
    _kai_dst_dt = ddt;

    if (types::is_integral_dt(sdt) && types::is_integral_dt(wdt)
            && !types::is_integral_dt(ddt)) {
        kai_gemm_type_ = kai_gemm_type::dequant;
    } else {
        kai_gemm_type_ = kai_gemm_type::noquant;
    }

    // Quant workflows do not yet support fixed format, in this case set_default_formats() will set
    // wtag, and we will do a reorder in execute
    if ((weights_md_was_any || weights_md_may_be_fixed_format)
            && kai_gemm_type_ == kai_gemm_type::noquant) {
        _cfg->weight_format = kai::ops::WeightFormat::ANY;
        _fixed_format = true;
    }

    // Note that this may change formats
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    if (with_bias() && bias_md_.format_kind == format_kind::any) {
        VDISPATCH_MATMUL_SC(memory_desc_init_by_strides(bias_md_, nullptr),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    if (_fixed_format && fast_mode
            && utils::everyone_is(data_type::f32, sdt, ddt)
            && wdt == data_type::bf16) {
        _kai_weights_dt = data_type::f32;
    }

    const memory_desc_wrapper bia_d(weights_md(1));
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_input_format(*src_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_MATMUL(
            cpu::matmul::gemm_based::check_gemm_output_format(*dst_md())
                    || cpu::matmul::gemm_based::check_gemm_input_format(
                            *dst_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_MATMUL(_fixed_format
                    || cpu::matmul::gemm_based::check_gemm_input_format(
                            *weights_md()),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");
    VDISPATCH_MATMUL(batch_dims_have_default_order(src_d),
            "src batch dimensions must be in order");
    VDISPATCH_MATMUL(batch_dims_have_default_order(wei_d),
            "weights batch dimensions must be in order");
    VDISPATCH_MATMUL(batch_dims_have_default_order(dst_d),
            "dst batch dimensions must be in order");
    VDISPATCH_MATMUL(
            IMPLICATION(with_bias(),
                    is_bias_1xN()
                            && cpu::matmul::gemm_based::
                                    check_gemm_output_format(*weights_md(1))
                            && bia_d.data_type() == ddt),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    if (kai_gemm_type_ == kai_gemm_type::noquant) {
        using smask_t = primitive_attr_t::skip_mask_t;
        VDISPATCH_MATMUL(
                attr()->has_default_values(smask_t::fpmath_mode
                        | smask_t::accumulation_mode | smask_t::post_ops),
                VERBOSE_UNSUPPORTED_ATTR);
    } else {
        using smask_t = primitive_attr_t::skip_mask_t;
        // TODO: implement zero_points
        VDISPATCH_MATMUL(utils::one_of(ddt, f32), VERBOSE_UNSUPPORTED_DT_CFG);
        VDISPATCH_MATMUL(
                attr()->has_default_values(smask_t::scales | smask_t::post_ops,
                        dst_md()->data_type),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_SRC)
                        || attr()->scales_.get_mask(DNNL_ARG_SRC) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
                        || attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_MATMUL(attr()->scales_.has_default_values(DNNL_ARG_DST),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    batch_mode_ = batch_mode::none;
    _ag_nbatches = 1;
    _ag_nmulti = 1;
    _src_broadcast_batch_dims = false;

    if (helper.batched()) {
        const bool src_matches_dst = batch_dims_match(src_d, dst_d);
        const bool wei_matches_dst = batch_dims_match(wei_d, dst_d);
        const bool src_all_one = batch_dims_are_all_one(src_d);
        const bool wei_all_one = batch_dims_are_all_one(wei_d);

        const bool can_use_batches = wei_all_one && src_matches_dst;
        const bool can_use_multis
                = wei_matches_dst && (src_matches_dst || src_all_one);

        VDISPATCH_MATMUL(can_use_batches || can_use_multis,
                "only supports batch dims that are fully shared or "
                "fully varying");

        if (can_use_batches) {
            batch_mode_ = batch_mode::batches;
            _ag_nbatches = static_cast<unsigned int>(helper.batch());
        } else {
            batch_mode_ = batch_mode::multis;
            _ag_nmulti = static_cast<unsigned int>(helper.batch());
            _src_broadcast_batch_dims = src_all_one;
        }
    }

    auto scratchpad = scratchpad_registry().registrar();

    unsigned int sections = 1;
    bool indirect = false;

    _reorder_src_ba_to_ab = helper.transA() == 'T';
    _reorder_dst_ab_to_ba = helper.transC() == 'T';
    _reorder_nested_scratchpad_base = attr_.post_ops_.len();

    if (types::is_integral_dt(ddt)) {
        VDISPATCH_MATMUL(attr_.post_ops_.len() == 0,
                "no post op support for integral dt");
    }
    VDISPATCH_MATMUL(num_sum_post_ops(attr_.post_ops_) <= 1,
            "supports at most one sum post op");
    const post_ops_fusion_t post_ops_fusion = create_post_ops_fusion(
            attr_.post_ops_, !with_bias() && !_reorder_dst_ab_to_ba);
    _has_post_ops_fallback = post_ops_fusion.has_fallback(attr_.post_ops_);

    // Strict/F32 mode applies post-ops before the final low-precision cast.
    const bool strict_f32_post_ops = utils::one_of(attr()->acc_mode_,
            accumulation_mode::strict, accumulation_mode::f32);
    // A fallback post-op runs in oneDNN after the KAI kernel.
    const bool fallback_rounds_low_precision_dst = _has_post_ops_fallback
            && utils::one_of(ddt, data_type::bf16, data_type::f16);
    // This combination needs F32 storage to avoid rounding before the post-op.
    const bool needs_f32_post_ops_intermediate
            = strict_f32_post_ops && fallback_rounds_low_precision_dst;

    // Unsupported strict low-precision cases use another implementation.
    const bool can_use_f32_post_ops_intermediate
            = desc()->accum_data_type == data_type::f32
            && utils::everyone_is(ddt, sdt, wdt) && !with_bias()
            && num_sum_post_ops(attr_.post_ops_) == 0 && !_reorder_dst_ab_to_ba;
    VDISPATCH_MATMUL(!needs_f32_post_ops_intermediate
                    || can_use_f32_post_ops_intermediate,
            "unsupported f32 fallback intermediate configuration");

    _use_f32_post_ops_intermediate = needs_f32_post_ops_intermediate;
    if (_use_f32_post_ops_intermediate) {
        // Make KAI write its unrounded result to an F32 temporary buffer.
        _kai_dst_dt = data_type::f32;
        // Keep the user's destination shape and layout, changing only its type.
        dst_f32_md_ = *dst_md();
        dst_f32_md_.data_type = data_type::f32;
        CHECK(post_ops.init(engine, attr_.post_ops_, dst_f32_md_,
                post_ops_fusion.fallback_start_index));
        // Prepare the one final cast from F32 to the user's BF16/F16 output.
        VDISPATCH_MATMUL_SC(
                reorder_primitive_desc_create(dst_f32_to_user_reorder_pd_,
                        engine, &dst_f32_md_, dst_md()),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "f32 dst reorder");
    } else {
        CHECK(post_ops.init(engine, attr_.post_ops_, *dst_md(),
                post_ops_fusion.fallback_start_index));
    }

    const int max_threads = dnnl_get_current_num_threads();
    const int num_threads = threads_for_kernel_execute(
            kernel_execute_work(*this), max_threads);
    _args = std::make_shared<kai::ops::GemmArgs>(get_cpu_info(), M(), N(), K(),
            sections, _ag_nbatches, _ag_nmulti, indirect,
            post_ops_fusion.activation, num_threads, _fixed_format, fast_mode,
            post_ops_fusion.accumulate, _cfg.get());

    std::unique_ptr<kai::ops::IGemmCommon> kernel = nullptr;

    // Create an kai object, this is where we enforce the datatype combination
    if (is_dequant()) {
        // Non-trivial placeholder value, because the value is only provided at runtime
        kai::ops::DequantizeFloat dequant(0.5);
        kernel = create_kai_gemm_dequant(dequant);
    } else {
        kernel = create_kai_gemm();
    }
    VDISPATCH_MATMUL(kernel, VERBOSE_UNSUPPORTED_DT_CFG);

    const bool weights_are_transposed
            = !_fixed_format && helper.transB() == 'T';
    _pack_weights = !_fixed_format && kernel->B_is_pretransposed();
    _pack_transposed_weights = weights_are_transposed && _pack_weights
            && kernel->B_pretranspose_supports_transpose();
    _reorder_weights_ba_to_ab
            = weights_are_transposed && !_pack_transposed_weights;

    // Copy the resulting config object constructed from kernel
    _cfg = std::make_shared<kai::ops::GemmConfig>(kernel->get_config());
    // Some generated filters do not match the impl list, so it ends up rejecting
    // the second time around. This could be removed if this is fixed in KleidiAI
    _cfg->filter.clear();

    if (_fixed_format) {
        // Logical dimension indices
        dim_t innermost_dim = weights_md_.ndims - 1;
        dim_t N_dim = innermost_dim;
        dim_t K_dim = innermost_dim - 1;

        // The logical indices of dimensions related to the batch, ordered from
        // innermost to outermost
        std::vector<dim_t> batch_dims = {};
        for (dim_t i = K_dim - 1; i >= 0; --i)
            batch_dims.push_back(i);

        VDISPATCH_MATMUL(kai_utils::is_fixed_format(_cfg->weight_format),
                "KAI did not select a fixed weights format");
        if (weights_md_was_any) {
            weight_format_to_memory_desc(weights_md_, _cfg->weight_format,
                    K_dim, N_dim, {}, batch_dims);
        } else {
            VDISPATCH_MATMUL(
                    memory_desc_matches_weight_format(weights_md_,
                            _cfg->weight_format, K_dim, N_dim, {}, batch_dims),
                    VERBOSE_UNSUPPORTED_TAG_S, "weights");
        }
    }

    if (kernel->get_working_size() != 0)
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                kernel->get_working_size(), 1);

    if (_reorder_src_ba_to_ab)
        VDISPATCH_MATMUL_SC(
                init_ab_reorder(engine, src_md(), src_ab_md_,
                        src_ba_to_ab_reorder_pd_, scratchpad,
                        memory_tracking::names::key_matmul_src_trans,
                        reorder_nested_key(nested_reorder_t::src),
                        ab_reorder_direction::user_to_ab),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "src reorder");

    if (_reorder_weights_ba_to_ab)
        VDISPATCH_MATMUL_SC(
                init_ab_reorder(engine, weights_md(), weights_ab_md_,
                        weights_ba_to_ab_reorder_pd_, scratchpad,
                        memory_tracking::names::key_matmul_pack_space,
                        reorder_nested_key(nested_reorder_t::weights),
                        ab_reorder_direction::user_to_ab),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "weights reorder");

    if (_reorder_dst_ab_to_ba)
        VDISPATCH_MATMUL_SC(
                init_ab_reorder(engine, dst_md(), dst_ab_md_,
                        dst_ab_to_ba_reorder_pd_, scratchpad,
                        memory_tracking::names::key_matmul_dst_trans,
                        reorder_nested_key(nested_reorder_t::dst),
                        ab_reorder_direction::ab_to_user),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "dst reorder");

    // KleidiAI names its B packing transform "pretranspose". Keep the oneDNN
    // flag named as packing so it is distinct from layout reorders such as
    // ba -> ab. Some packers consume transposed user weights directly; in
    // that case _reorder_weights_ba_to_ab stays false and the packer receives
    // the original transposed layout.
    if (_pack_weights)
        scratchpad.book(memory_tracking::names::key_matmul_wei_trans,
                kernel->get_B_pretransposed_array_size(), 1);

    if (_use_f32_post_ops_intermediate || post_ops.has_sum()) {
        // Reserve temp dst, the new path sizes it as F32.
        const memory_desc_wrapper tmp_dst_d(
                _use_f32_post_ops_intermediate ? &dst_f32_md_ : dst_md());
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                tmp_dst_d.size(), 1, 64, 64);
    }
    if (_use_f32_post_ops_intermediate) {
        // Reserve any temporary memory needed by the final cast.
        scratchpad.book(reorder_nested_key(nested_reorder_t::dst_f32_to_user),
                dst_f32_to_user_reorder_pd_->scratchpad_registry());
    }
    post_ops.init_scratchpad(scratchpad);

    return status::success;
}

status_t kai_matmul_t::init(engine_t *engine) {
    if (pd()->src_ba_to_ab_reorder_pd_)
        CHECK(pd()->src_ba_to_ab_reorder_pd_->create_primitive(
                src_ba_to_ab_reorder_, engine));
    if (pd()->weights_ba_to_ab_reorder_pd_)
        CHECK(pd()->weights_ba_to_ab_reorder_pd_->create_primitive(
                weights_ba_to_ab_reorder_, engine));
    if (pd()->dst_ab_to_ba_reorder_pd_)
        CHECK(pd()->dst_ab_to_ba_reorder_pd_->create_primitive(
                dst_ab_to_ba_reorder_, engine));
    if (pd()->dst_f32_to_user_reorder_pd_)
        CHECK(pd()->dst_f32_to_user_reorder_pd_->create_primitive(
                dst_f32_to_user_reorder_, engine));
    post_ops_ = pd()->post_ops;
    CHECK(post_ops_.init_primitives(engine));
    return status::success;
}

status_t kai_matmul_t::execute(const exec_ctx_t &ctx) const {

    const int max_threads = dnnl_get_current_num_threads();
    int num_threads = threads_for_kernel_execute(
            kernel_execute_work(*pd()), max_threads);

    std::unique_ptr<kai::ops::IGemmCommon> _kernel = nullptr;
    if (pd()->is_dequant()) {
        DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
        DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
        kai::ops::DequantizeFloat dequant(src_scale[0] * wei_scale[0]);
        _kernel = pd()->create_kai_gemm_dequant(dequant, num_threads);
    } else {
        _kernel = pd()->create_kai_gemm(num_threads);
    }
    if (!_kernel) return status::runtime_error;

    if (get_verbose(verbose_t::profile_externals)) {
        std::cout << "profile_externals: " << _kernel->get_config().filter
                  << std::endl;
    }

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const auto &post_ops = post_ops_;

    const kai::ops::ndrange_t window_size = _kernel->get_window_size();
    const auto thread_partition
            = make_thread_partition(num_threads, window_size);
    num_threads = thread_partition.nthr;

    _kernel->set_nthreads(num_threads);

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);

    auto raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *wei_base = const_cast<void *>(raw_wei);

    auto dst_arg = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    // The guarded path makes KAI and fallback post-ops share the F32 buffer.
    auto dst_base = pd()->_reorder_dst_ab_to_ba
            ? scratchpad.get<void>(memory_tracking::names::key_matmul_dst_trans)
            : (pd()->_use_f32_post_ops_intermediate || post_ops.has_sum()
                              ? scratchpad.get<void>(memory_tracking::names::
                                                key_matmul_dst_in_acc_dt)
                              : dst_arg);
    const void *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    if (pd()->_reorder_src_ba_to_ab) {
        auto *tmp_src_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_src_trans);
        CHECK(execute_reorder(ctx, src_ba_to_ab_reorder_, pd()->src_md(),
                &pd()->src_ab_md_, src_base, tmp_src_base,
                pd()->reorder_nested_key(pd_t::nested_reorder_t::src)));
        src_base = tmp_src_base;
    }

    // If KleidiAI's packer can consume transposed weights, keep raw_wei pointing at
    // the user buffer and pass _pack_transposed_weights to pretranspose_B.
    // Otherwise, normalize weights to ab once with oneDNN reorder.
    if (pd()->_reorder_weights_ba_to_ab) {
        auto *tmp_weights_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_pack_space);
        CHECK(execute_reorder(ctx, weights_ba_to_ab_reorder_,
                pd()->weights_md(), &pd()->weights_ab_md_, raw_wei,
                tmp_weights_base,
                pd()->reorder_nested_key(pd_t::nested_reorder_t::weights)));
        raw_wei = tmp_weights_base;
        wei_base = tmp_weights_base;
    }

    if (pd()->_pack_weights) {
        wei_base = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
    }

    const memory_desc_t *kernel_src_md
            = pd()->_reorder_src_ba_to_ab ? &pd()->src_ab_md_ : pd()->src_md();
    const memory_desc_t *kernel_wei_md = pd()->_reorder_weights_ba_to_ab
            ? &pd()->weights_ab_md_
            : pd()->weights_md();
    // Describe the temporary output to KAI as F32 as well.
    const memory_desc_t *kernel_dst_md = pd()->_use_f32_post_ops_intermediate
            ? &pd()->dst_f32_md_
            : (pd()->_reorder_dst_ab_to_ba ? &pd()->dst_ab_md_
                                           : pd()->dst_md());

    const memory_desc_wrapper src_d(kernel_src_md);
    const memory_desc_wrapper wei_d(kernel_wei_md);
    const memory_desc_wrapper dst_d(kernel_dst_md);
    cpu::matmul::matmul_helper_t helper(src_d, wei_d, dst_d);

    // Leading dimensions of our matrices are the strides of the first non-dense dimensions (second logical)
    auto ld_src = kernel_src_md->format_desc.blocking
                          .strides[kernel_src_md->ndims - 2];
    auto ld_dst = kernel_dst_md->format_desc.blocking
                          .strides[kernel_dst_md->ndims - 2];

    // With fixed format, weights are already packed in kai's expected
    // layout, so the row stride follows the innermost logical dimension.
    // Otherwise rely on matmul_helper_t for the kernel-facing descriptor.
    auto ld_wei = pd()->_fixed_format
            ? kernel_wei_md->format_desc.blocking
                      .strides[kernel_wei_md->ndims - 1]
            : helper.ldb();

    const int src_batch_stride = get_innermost_batch_stride(kernel_src_md);
    const int wei_batch_stride = get_innermost_batch_stride(kernel_wei_md);
    const int dst_batch_stride = get_innermost_batch_stride(kernel_dst_md);

    const int batch_stride_a = pd()->is_batches() ? src_batch_stride : 0;
    const int multi_stride_a = pd()->is_multis()
            ? (pd()->_src_broadcast_batch_dims ? 0 : src_batch_stride)
            : 0;
    const int multi_stride_b = pd()->is_multis() ? wei_batch_stride : 0;
    const int batch_stride_c = pd()->is_batches() ? dst_batch_stride : 0;
    const int multi_stride_c = pd()->is_multis() ? dst_batch_stride : 0;

    if (pd()->_pack_weights) {
        const int reorder_num_threads = threads_for_weight_reorder(
                weight_reorder_work(*pd()), num_threads);
        parallel_pretranspose_B_array(*_kernel, wei_base, raw_wei, ld_wei,
                multi_stride_b, pd()->_pack_transposed_weights,
                reorder_num_threads);
    }

    if (_kernel->get_working_size() != 0) {
        _kernel->set_working_space(scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer));
    }

    _kernel->set_arrays_generic(src_base, ld_src, batch_stride_a,
            multi_stride_a, wei_base, ld_wei, multi_stride_b, dst_base, ld_dst,
            batch_stride_c, multi_stride_c, bias_base, 0);

    parallel_execute(*_kernel, window_size, thread_partition);

    if (pd()->_reorder_dst_ab_to_ba) {
        auto *reordered_dst = post_ops.has_sum()
                ? scratchpad.get<void>(
                          memory_tracking::names::key_matmul_dst_in_acc_dt)
                : dst_arg;
        CHECK(execute_reorder(ctx, dst_ab_to_ba_reorder_, &pd()->dst_ab_md_,
                pd()->dst_md(), dst_base, reordered_dst,
                pd()->reorder_nested_key(pd_t::nested_reorder_t::dst)));
        dst_base = reordered_dst;
    }

    if (pd()->_has_post_ops_fallback) {
        if (post_ops.has_sum())
            CHECK(post_ops.execute(ctx, dst_base, dst_arg));
        else
            CHECK(post_ops.execute(ctx, dst_base));
    }

    if (pd()->_use_f32_post_ops_intermediate) {
        // Post-ops are complete, so cast once into the user's BF16/F16 buffer.
        CHECK(execute_reorder(ctx, dst_f32_to_user_reorder_, &pd()->dst_f32_md_,
                pd()->dst_md(), dst_base, dst_arg,
                pd()->reorder_nested_key(
                        pd_t::nested_reorder_t::dst_f32_to_user)));
    }

    return status::success;
}

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
