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

#include "gpu/intel/matmul/with_post_ops.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/intel/gemm/host_scalars.hpp"
#include "gpu/intel/gemm/jit/jit_gemm_pd.hpp"
#include "gpu/intel/utils.hpp"

// jit_gemm_pd.hpp defines VDISPATCH_JIT_GEMM referencing this->info(engine);
// that helper is on jit_gemm_pd_t, not gpu_matmul_pd_t, so override here.
#undef VDISPATCH_JIT_GEMM

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

#define VDISPATCH_JIT_GEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, "%s," msg, "ocl:with_po:any", ##__VA_ARGS__)

#define VDISPATCH_JIT_GEMM_IC(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, matmul, (cond), \
            status::unimplemented, "%s," msg, "ocl:with_po:any", ##__VA_ARGS__)

status_t with_post_ops_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    const auto a_type = src_md(0)->data_type;
    const auto b_type = weights_md(0)->data_type;
    const auto c_type = dst_md(0)->data_type;

    using smask_t = primitive_attr_t::skip_mask_t;
    const auto attr_skip_mask = smask_t::scales_data_type
            | smask_t::scales_groups | smask_t::post_ops
            | smask_t::accumulation_mode | smask_t::fpmath_mode
            | smask_t::zero_points_data_type | smask_t::dropout;

    bool wei_decomp = (utils::one_of(c_type, f32, f16, bf16)
                              && utils::one_of(a_type, u8, s8, u4, s4)
                              && utils::one_of(b_type, f16, f32, bf16))
            && attr()->mayiconvert(a_type, f32);
    VDISPATCH_JIT_GEMM(
            ndims() <= 6, VERBOSE_UNSUPPORTED_MD_FLAG, "c_desc.ndims");
    VDISPATCH_JIT_GEMM(!has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_JIT_GEMM(attr()->has_default_values(attr_skip_mask),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_JIT_GEMM(!utils::one_of(c_type, u4, s4), VERBOSE_UNSUPPORTED_DT);

    // gemm_post_ops kernel supports only dst zero-point (incl. host scalar).
    const auto &zps = attr()->zero_points_;
    VDISPATCH_JIT_GEMM(!(zps.get(kA).is_host_scalar()
                               || zps.get(kB).is_host_scalar()),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    const primitive_attr_t *attributes_with_po = attr();
    for (int arg : {kA, kB, kC}) {
        if (attr()->scales_.has_default_values(arg)) continue;

        const auto &mask = attr()->scales_.get_mask(arg);
        if (arg == kB && !wei_decomp) {
            VDISPATCH_JIT_GEMM(
                    (mask == 0 || mask == (1 << (dst_md()->ndims - 1))),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        } else if (arg == kC && attr()->scales_.get(arg).is_dynamic()) {
            VDISPATCH_JIT_GEMM(utils::one_of(a_type, f4_e2m1, f8_e5m2, f8_e4m3)
                            && utils::one_of(
                                    b_type, f4_e2m1, f8_e5m2, f8_e4m3),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        } else
            VDISPATCH_JIT_GEMM((mask == 0), VERBOSE_UNSUPPORTED_SCALES_CFG);
    }
    attr_info_ = attr_info_t::create(attributes_with_po);
    requires_user_scales_ = attr_info_.with_src_scales
            || attr_info_.with_wei_scales || attr_info_.with_dst_scales;

    const auto &po = attributes_with_po->post_ops_;
    for (auto i = 0; i < po.len(); ++i)
        VDISPATCH_JIT_GEMM(!po.entry_[i].is_binary_with_ternary_op(),
                VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_JIT_GEMM(
            !with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "bias reduction");

    subbyte_pack_ = utils::one_of(c_type, f4_e2m1, f4_e3m0);
    if (subbyte_pack_) {
        using namespace dnnl::impl::memory_tracking::names;
        const memory_desc_wrapper dst_mdw(dst_md(0));
        const auto &padded_dims = dst_mdw.padded_dims();
        const dim_t nd = dst_mdw.ndims();
        const dim_t nelems = utils::array_product(padded_dims, nd);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_pack_space, nelems,
                sizeof(char), OCL_BUFFER_ALIGNMENT);
    }

    dynamic_scales_ = attr()->scales_.get(kC).is_dynamic();
    if (dynamic_scales_) {
        using namespace dnnl::impl::memory_tracking::names;
        const memory_desc_wrapper dst_mdw(dst_md(0));
        const auto &padded_dims = dst_mdw.padded_dims();
        const dim_t nd = dst_mdw.ndims();
        const dim_t nelems = utils::array_product(padded_dims, nd);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_dyn_scale_space,
                nelems, sizeof(float), OCL_BUFFER_ALIGNMENT);
    }

    const auto impl_list = engine->get_implementation_list(op_desc());
    int current_impl_idx
            = impl_list_item_t::find<with_post_ops_t::pd_t>(impl_list);

    primitive_desc_iterator_t it_with_po(engine, op_desc(), attributes_with_po,
            nullptr, current_impl_idx /* skip implementation */);
    if (!it_with_po.is_initialized()) return status::invalid_arguments;
    pd_ = *(++it_with_po);
    // Exit if the inner gemm kernel already fuses post-ops natively.
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto arch = intel_engine->device_info()->gpu_arch();
    bool is_xe_hp = arch >= compute::gpu_arch_t::xe_hp;
    auto skip_impl = is_xe_hp ? "ocl" : "ref";
    VDISPATCH_JIT_GEMM(!(pd_ && strstr(pd_->name(), skip_impl) == nullptr),
            VERBOSE_SKIP_PRIMITIVE_IMPL);

    matmul_desc_t desc_copy = *this->desc();
    dst_type_ = desc_copy.dst_desc.data_type;
    const data_type_t intermediate_c_type
            = (engine->mayiuse_f16_accumulator_with_f16()
                      && utils::one_of(data_type::f16,
                              desc_copy.src_desc.data_type,
                              desc_copy.weights_desc.data_type))
            ? data_type::f32
            : desc_copy.accum_data_type;
    desc_copy.dst_desc.data_type = intermediate_c_type;
    acc_type_ = intermediate_c_type;
    use_reorder = dst_md(0)->data_type != intermediate_c_type;
    desc_copy.bias_desc = glob_zero_md;

    // Setup empty attributes but keep zero points for gemm.
    primitive_attr_t attributes_without_po = *attr();
    CHECK(attributes_without_po.set_post_ops(post_ops_t()));
    attributes_without_po.scales_ = scales_t();
    attributes_without_po.zero_points_ = zero_points_t();
    attributes_without_po.dropout_ = dropout_t();
    const auto &zp = attributes_with_po->zero_points_;
    // Copy the full quant_entry_t for each user-supplied zp slot. The 2-arg
    // `set(arg, mask)` overload resets data_type to the default (s32) and
    // clears group_ndims/group_dims, silently dropping any user-supplied
    // dtype (e.g. f32/s8) or grouped-zp metadata. The entry-copy overload
    // preserves mask, data_type, group_ndims/group_dims, is_host_scalar,
    // and qmode. (DST zp is intentionally NOT forwarded to the inner; the
    // post-op worker applies it on the user-dtype dst — see line ~66.)
    if (!zp.has_default_values(kA)) {
        CHECK(attributes_without_po.zero_points_.set(kA, zp.get(kA)));
    }
    if (!zp.has_default_values(kB)) {
        CHECK(attributes_without_po.zero_points_.set(kB, zp.get(kB)));
    }

    primitive_desc_iterator_t it_without_po(engine,
            reinterpret_cast<const op_desc_t *>(&desc_copy),
            &attributes_without_po, nullptr,
            current_impl_idx /* skip implementation */);
    if (!it_without_po.is_initialized()) return status::invalid_arguments;
    pd_ = *(++it_without_po);
    VDISPATCH_JIT_GEMM(!(!pd_ || strstr(pd_->name(), skip_impl) != nullptr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, pd_ ? pd_->name() : "");

    // Take the inner pd's resolved layouts as our own kernel-view layouts.
    // The post-op worker reads pd_->dst_md(0) (intermediate dt) and writes
    // dst_md(0) (final dt), so the inner's C md drives both shapes.
    src_md_ = *pd_->src_md(0);
    weights_md_ = *pd_->weights_md(0);
    memory_desc_t c = *pd_->dst_md(0);
    c.data_type = dst_type_;
    dst_md_ = c;
    // Keep our own bias_md_ (we cleared inner's bias in desc_copy).
    desc_.accum_data_type = intermediate_c_type;

    CHECK(attr_.set_default_formats(dst_md(0)));
    VDISPATCH_JIT_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    use_scratchpad_with_post_op_worker = use_reorder
            || attributes_with_po->post_ops_.find(primitive_kind_t::dnnl_sum)
                    != -1;
    with_dropout = !attr()->dropout_.has_default_values();
    if (with_dropout) {
        dropout_use_host_scalars = attr()->dropout_.use_host_scalars_;
        dropout_use_offset = attr()->dropout_.use_offset_;
        dropout_has_output_mask = attr()->dropout_.has_output_mask();
        assert(memory_desc_wrapper(dst_md(0)).format_kind()
                == format_kind::blocked);
        using namespace format_tag;
        VDISPATCH_JIT_GEMM_IC(
                memory_desc_matches_one_of_tag(*dst_md(0), ncdhw, nchw, ncw, nc)
                        && IMPLICATION(dropout_has_output_mask,
                                memory_desc_wrapper(dst_md(0)).similar_to(
                                        attr()->dropout_.dropout_desc_, true,
                                        false)),
                VERBOSE_UNSUPPORTED_DROPOUT);
    }
    auto nd = pd_->dst_md()->ndims;
    dispatch_ = intel_engine->create_dispatch(pd_->dst_md());
    dispatch_.define_dim("D0", 0, pd_->dst_md()->padded_dims[0]);
    dispatch_.define_dim("D1", 1, pd_->dst_md()->padded_dims[1]);
    dispatch_.define_dim("D2", nd > 2 ? 2 : 0,
            nd > 2 ? pd_->dst_md()->padded_dims[2] : 1);
    dispatch_.define_dim("D3", nd > 3 ? 3 : 0,
            nd > 3 ? pd_->dst_md()->padded_dims[3] : 1);
    dispatch_.define_dim("D4", nd > 4 ? 4 : 0,
            nd > 4 ? pd_->dst_md()->padded_dims[4] : 1);
    dispatch_.define_dim("D5", nd > 5 ? 5 : 0,
            nd > 5 ? pd_->dst_md()->padded_dims[5] : 1);
    dispatch_.generate();

    init_scratchpad();

    return status::success;
}

status_t with_post_ops_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    auto c_type = dst_md(0)->data_type;
    const auto src_info = memory_desc_info_t::create(pd_->dst_md(0));
    const auto bias_info = [&]() {
        // If no bias, default to dst layout (just a placeholder).
        auto info = memory_desc_info_t::create(
                matmul_pd_t::with_bias() ? weights_md(1) : dst_md(0));
        if (info.data_type == data_type::undef) info.data_type = data_type::f32;
        return info;
    }();

    def_memory_desc_info(kernel_ctx, src_info, "SRC");
    def_memory_desc_info(kernel_ctx, bias_info, "BIAS");
    if (dynamic_scales_) {
        dnnl_memory_desc d_md(*dst_md(0));
        d_md.data_type = acc_type_;
        memory_desc_wrapper d_mdw(d_md);
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(d_mdw), "DST");
    } else {
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(dst_md(0)), "DST");
    }

    int nd = src_info.ndims;
    kernel_ctx.set_data_type(dynamic_scales_ ? acc_type_ : c_type);
    kernel_ctx.require_stateless_addressing(has_large_buffers());

    const bool with_src_scales = attr_info_.with_src_scales;
    const bool with_wei_scales = attr_info_.with_wei_scales;
    const bool with_dst_scales = attr_info_.with_dst_scales;
    auto is_int_type = [](data_type_t t) {
        return utils::one_of(t, data_type::s8, data_type::u8, data_type::s32);
    };
    data_type_t acc_t = desc_.accum_data_type;
    if (desc_.accum_data_type == data_type::s32) {
        if (with_src_scales || with_wei_scales
                || !is_int_type(bias_info.data_type)
                || !is_int_type(dst_md(0)->data_type)) {
            acc_t = data_type::f32;
        }
    }
    def_data_type(kernel_ctx, acc_t, "ACC");

    kernel_ctx.define_int("NDIMS", nd);
    CHECK(def_attr_info(
            kernel_ctx, attr_info_, attr()->post_ops_, *pd_->dst_md()));
    kernel_ctx.define_int("A_SCALES", with_src_scales);
    kernel_ctx.define_int("B_SCALES", with_wei_scales);
    kernel_ctx.define_int("C_SCALES", with_dst_scales);
    // Per-OC wei scale varies along DST's last dim (the N axis in matmul
    // convention). The inner gen_t's swap_ab_ is a kernel-internal flag — it
    // does not rotate the matmul-convention dst buffer the post-op worker
    // iterates over.
    kernel_ctx.define_int("B_SCALE_AXIS", nd - 1);
    kernel_ctx.define_int("DST_ZERO_POINT", attr_info_.with_dst_zpoints);
    kernel_ctx.define_int("WITH_DROPOUT", with_dropout);
    kernel_ctx.define_int("DROPOUT_USE_HOST_SCALARS", dropout_use_host_scalars);
    kernel_ctx.define_int("DROPOUT_USE_OFFSET", dropout_use_offset);
    kernel_ctx.define_int("DROPOUT_HAS_OUTPUT_MASK", dropout_has_output_mask);
    def_dispatch(kernel_ctx, dispatch_);
    return status::success;
}

void with_post_ops_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (use_scratchpad_with_post_op_worker) {
        memory_desc_wrapper dst_mdw(dst_md());
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                dst_mdw.nelems(/*with_padding=*/true),
                types::data_type_size(desc_.accum_data_type));
    }
    scratchpad.book(memory_tracking::names::key_nested_multiple,
            pd_->scratchpad_registry());
}

status_t with_post_ops_t::execute(const impl::exec_ctx_t &ctx) const {
    gpu_assert(!pd()->requires_user_scales_ || !ctx.args().empty());
    std::unique_ptr<memory_t, memory_deleter_t> c_mem_before_po_worker;
    impl::exec_args_t nested_args(ctx.args());

    if (pd()->use_scratchpad()) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_tmp_buffer);
        auto tmp_md = *(pd()->dst_md(0));
        tmp_md.data_type = pd()->desc()->accum_data_type;
        CHECK(safe_ptr_assign(c_mem_before_po_worker,
                new memory_t(ctx.stream()->engine(), &tmp_md,
                        std::move(scratchpad))));

        nested_args[DNNL_ARG_DST]
                = memory_arg_t(c_mem_before_po_worker.get(), false);
    }

    impl::exec_ctx_t nested_ctx(ctx, std::move(nested_args));
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested_multiple,
            prim_->pd()->scratchpad_registry());
    nested_ctx.set_scratchpad_grantor(nested_grantor);

    CHECK(prim_->execute(nested_ctx));

    const bool subbyte_pack = pd()->subbyte_pack_;
    const bool dyn_scales = pd()->dynamic_scales_;

    auto tmp = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_pack_space);

    auto tmp_ds = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_dyn_scale_space);

    auto &c_user = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &bias_user = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &c_zp_user = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    // Read the user dst-scales storage by probing ctx.args() directly rather
    // than via CTX_IN_STORAGE. Dynamic dst scales (dst:mx / dst:dynamic_fp)
    // are registered as arg_usage_t::output by the framework, and
    // ctx.input() asserts on non-const args. The args-map read bypasses
    // input/output discrimination — matches the user_scale lambda below.
    auto user_dst_scales
            = [&ctx]() -> const memory_storage_t & {
        const auto &args_map = ctx.args();
        const auto it = args_map.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
        if (it == args_map.end() || !it->second.mem())
            return memory_storage_t::empty_storage();
        const auto *s = it->second.mem()->memory_storage();
        return s ? *s : memory_storage_t::empty_storage();
    };
    auto &c_scales_user = user_dst_scales();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0,
            pd()->use_scratchpad() ? *c_mem_before_po_worker->memory_storage()
                                   : c_user);
    arg_list.set(1, bias_user);
    arg_list.set(2,
            dyn_scales             ? *tmp_ds
                    : subbyte_pack ? *tmp
                                   : c_user);
    int idx = append_post_ops_to_arg_list_base(ctx.args(), arg_list, 3,
            pd()->attr()->post_ops_, *pd()->dst_md());
    const auto user_scale = [&ctx](int user_arg) -> const memory_storage_t & {
        auto it = ctx.args().find(DNNL_ARG_ATTR_SCALES | user_arg);
        if (it != ctx.args().end() && it->second.mem()) {
            const auto *s = it->second.mem()->memory_storage();
            if (s) return *s;
        }
        return memory_storage_t::empty_storage();
    };
    // CONTRACT (enforced at pd init, see validator):
    //   - user SRC mask == 0 (kernel reads a_scales[0] only).
    //   - user WEIGHTS mask is 0 or per-OC (1 << (dst_ndims - 1)).
    //   - wei_scales_mask captured pre-swap; consumed below as a boolean.
    // The a_scales / b_scales slots are USER-keyed (matmul SRC/WEIGHTS);
    // the gemm-internal attr_.scales_ entries are post-swap by execute time
    // and must NOT be read here.
    arg_list.set(idx++, user_scale(DNNL_ARG_SRC));
    arg_list.set(idx++, user_scale(DNNL_ARG_WEIGHTS));
    arg_list.set(idx++, c_scales_user);
    arg_list.set(idx++, pd()->attr_info_.wei_scales_mask > 0 ? 1 : 0);
    arg_list.set(idx, c_zp_user);
    if (pd()->with_dropout) {
        const auto mem_dropout_seed
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_SEED);
        const auto mem_dropout_offset
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_OFFSET);
        const auto mem_dropout_prob
                = &CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
        idx++;
        arg_list.set(idx++, CTX_OUT_STORAGE(DNNL_ARG_ATTR_DROPOUT_MASK));
        if (pd()->dropout_use_host_scalars) {
            int64_t scalar_dropout_seed = 0;
            int64_t scalar_dropout_offset = 0;
            float scalar_dropout_prob = 0.f;
            CHECK(gemm::maybe_get_host_scalar_value(
                    *mem_dropout_seed, scalar_dropout_seed));
            if (pd()->dropout_use_offset) {
                CHECK(gemm::maybe_get_host_scalar_value(
                        *mem_dropout_offset, scalar_dropout_offset));
            }
            CHECK(gemm::maybe_get_host_scalar_value(
                    *mem_dropout_prob, scalar_dropout_prob));
            arg_list.set(idx++, scalar_dropout_seed);
            arg_list.set(idx++, scalar_dropout_offset);
            arg_list.set(idx, scalar_dropout_prob);
        } else {
            arg_list.set(idx++, *mem_dropout_seed);
            arg_list.set(idx++, *mem_dropout_offset);
            arg_list.set(idx, *mem_dropout_prob);
        }
    }
    auto nd_range = pd()->dispatch_.nd_range();
    CHECK(parallel_for(ctx, nd_range, kernels_[0], arg_list));

    if (dyn_scales) {
        const auto group_size = pd()->attr()->scales_.get_group(pd_t::kC, -1);
        const auto c_d = nested_ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
        const int last = c_d.ndims() - 1;
        const dim_t D3 = c_d.ndims() > 5 ? c_d.dims()[last - 5] : 1;
        const dim_t D2 = c_d.ndims() > 4 ? c_d.dims()[last - 4] : 1;
        const dim_t D1 = c_d.ndims() > 3 ? c_d.dims()[last - 3] : 1;
        const dim_t D0 = c_d.ndims() > 2 ? c_d.dims()[last - 2] : 1;
        const dim_t M = c_d.dims()[last - 1];
        const dim_t N = c_d.dims()[last];
        dnnl_dims_t c_stride {0};
        const auto &c_strides = c_d.blocking_desc().strides;
        for (int i = 0; i < c_d.ndims(); i++)
            if (c_d.dims()[last - i] > 1) { c_stride[i] = c_strides[last - i]; }

        compute::kernel_arg_list_t arg_list_2;
        int arg_idx = 0;
        arg_list_2.set(arg_idx++, *tmp_ds);
        arg_list_2.set(arg_idx++, subbyte_pack ? *tmp : c_user);
        arg_list_2.set(arg_idx++, c_scales_user);
        arg_list_2.set(arg_idx++, group_size);
        arg_list_2.set(arg_idx++, D0);
        arg_list_2.set(arg_idx++, D1);
        arg_list_2.set(arg_idx++, D2);
        arg_list_2.set(arg_idx++, c_stride[5]);
        arg_list_2.set(arg_idx++, c_stride[4]);
        arg_list_2.set(arg_idx++, c_stride[3]);
        arg_list_2.set(arg_idx++, c_stride[2]);
        arg_list_2.set(arg_idx++, c_stride[1]);
        arg_list_2.set(arg_idx++, c_stride[0]);
        compute::range_t gws({(size_t)M, (size_t)N / group_size,
                (size_t)(D0 * D1 * D2 * D3)});
        compute::nd_range_t nd_range_2(gws);
        CHECK(parallel_for(ctx, nd_range_2, kernels_[1], arg_list_2));
    }
    if (!subbyte_pack) return status_t::dnnl_success;
    memory_desc_wrapper dst_mdw(pd()->dst_md(0));
    const dim_t nelems = dst_mdw.nelems();
    compute::kernel_arg_list_t repack_arg_list;
    repack_arg_list.set(0, *tmp);
    repack_arg_list.set(1, c_user);
    repack_arg_list.set(2, into<dim_t>(nelems));
    repack_arg_list.set(3, 4);
    compute::range_t repack_gws((nelems * 4 + 7) / 8);
    compute::nd_range_t repack_nd_range(repack_gws);
    return large_parallel_for(impl::exec_ctx_t(ctx.stream()), repack_nd_range,
            kernels_[2], repack_arg_list, 4);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
