/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "common/host_scalar_memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/sdpa/brgemm_sdpa.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

DEFINE_PD_CREATE(brgemm_sdpa_fwd_t::pd_t)

status_t brgemm_sdpa_fwd_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace status;

    UNUSED(engine);
    VDISPATCH_SDPA(is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_SDPA(desc()->prop_kind == prop_kind::forward_inference,
            "training-forward (softmax stats output) is not supported");

    VDISPATCH_SDPA(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);

    const memory_desc_wrapper qry_mdw(desc()->qry_md());
    const memory_desc_wrapper key_mdw(desc()->key_md());
    const memory_desc_wrapper val_mdw(desc()->val_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    VDISPATCH_SDPA(utils::everyone_is(sdpa_pd_t::ndims, qry_mdw.ndims(),
                           key_mdw.ndims(), val_mdw.ndims(), dst_mdw.ndims()),
            VERBOSE_SHAPE_RESTRICTION
            ": qry(%d) key(%d) val(%d) and dst(%d) must be 4d",
            qry_mdw.ndims(), key_mdw.ndims(), val_mdw.ndims(), dst_mdw.ndims());
    VDISPATCH_SDPA(
            utils::everyone_is(true, qry_mdw.is_plain(), key_mdw.is_plain(),
                    val_mdw.is_plain(), dst_mdw.is_plain()),
            VERBOSE_UNSUPPORTED_TAG);

    for (int d = 0; d < sdpa_pd_t::ndims; ++d)
        VDISPATCH_SDPA(dst_mdw.dims()[d] == 1 || dst_mdw.strides()[d] != 0,
                "dst must not have zero strides on non-unit dimensions");

    VDISPATCH_SDPA(qry_mdw.strides()[sdpa_pd_t::ndims - 1] == 1,
            "the query head-size axis must be contiguous");
    VDISPATCH_SDPA(key_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
                    || key_mdw.strides()[sdpa_pd_t::ndims - 2] == 1,
            "one key matrix axis must be contiguous");
    VDISPATCH_SDPA(val_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
                    || val_mdw.strides()[sdpa_pd_t::ndims - 2] == 1,
            "one value matrix axis must be contiguous");

    const data_type_t dt = qry_mdw.data_type();
    VDISPATCH_SDPA(utils::one_of(dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_SDPA(utils::everyone_is(dt, key_mdw.data_type(),
                           val_mdw.data_type(), dst_mdw.data_type()),
            "query, key, value and dst data types must match");

    VDISPATCH_SDPA(kq_acc_dt() == f32 && vs_acc_dt() == f32,
            "only f32 accumulation is currently supported");
    VDISPATCH_SDPA(!with_query_scales() && !with_key_scales()
                    && !with_value_scales() && !with_key_zp()
                    && !with_value_zp() && !with_probs_quant_scales()
                    && !with_probs_dequant_scales(),
            "quantization is not yet supported");
    VDISPATCH_SDPA(!with_causal_mask(),
            "an implicit causal mask is not yet supported");
    VDISPATCH_SDPA(IMPLICATION(with_attn_mask(),
                           with_buffer_mask() || with_select_mask()),
            "only explicit buffer or select attention masks are supported");
    VDISPATCH_SDPA(
            utils::one_of(desc()->softmax_alg, alg_kind::softmax_accurate,
                    alg_kind::softmax_accurate_inf_as_zero),
            "unsupported softmax algorithm");
    VDISPATCH_SDPA(desc()->num_kv_heads() > 0
                    && desc()->num_q_heads() % desc()->num_kv_heads() == 0,
            "the number of query heads must be a multiple of the number of "
            "key/value heads");

    if (with_attn_mask()) {
        const memory_desc_wrapper mask_mdw(desc()->attn_mask_md());
        VDISPATCH_SDPA(mask_mdw.ndims() == sdpa_pd_t::ndims,
                VERBOSE_SHAPE_RESTRICTION ": attn_mask(%d) must be 4d",
                mask_mdw.ndims());
        VDISPATCH_SDPA(mask_mdw.is_plain(), VERBOSE_UNSUPPORTED_TAG);
        VDISPATCH_SDPA(utils::one_of(mask_mdw.dims()[mask_q_index],
                               desc()->queries(), 1),
                VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
        VDISPATCH_SDPA(mask_mdw.dims()[mask_k_index] == desc()->keys(),
                VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
        if (with_select_mask()) {
            VDISPATCH_SDPA(utils::one_of(mask_mdw.data_type(), s8, u8),
                    "the select condition data type must be s8 or u8");
            VDISPATCH_SDPA(mask_mdw.dims()[mask_k_index] == 1
                            || mask_mdw.strides()[mask_k_index] == 1,
                    "the select condition key axis must be contiguous");

            const memory_desc_wrapper fill_mdw(desc()->fill_md());
            VDISPATCH_SDPA(fill_mdw.ndims() == 1 && fill_mdw.dims()[0] == 1
                            && utils::one_of(
                                    fill_mdw.data_type(), f32, bf16, f16),
                    "the select attention mask fill must be a scalar f32, "
                    "bf16, or f16");
        } else
            VDISPATCH_SDPA(
                    mask_mdw.data_type() == dt || mask_mdw.data_type() == f32,
                    "the attention mask data type must match qry/dst or be "
                    "f32");
    }

    if (with_attn_scale()) {
        const memory_desc_wrapper scale_mdw(desc()->scale_md());
        VDISPATCH_SDPA(scale_mdw.ndims() == 1 && scale_mdw.dims()[0] == 1
                        && utils::one_of(scale_mdw.data_type(), f32, bf16, f16),
                "the attention scale must be a scalar f32, bf16, or f16");
    }

    VDISPATCH_SDPA(
            sdpa_fwd_pd_t::set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    // ONEDNN_SDPA_IMPL={auto,full_softmax,online_softmax}: auto and unset
    // use full softmax; online softmax must be explicitly requested.
    const std::string forced = getenv_string_user("SDPA_IMPL");
    VDISPATCH_SDPA(
            utils::one_of(forced, std::string(), std::string("auto"),
                    std::string("full_softmax"), std::string("online_softmax")),
            "unknown ONEDNN_SDPA_IMPL value '%s'", forced.c_str());
    if (forced == "online_softmax") {
        // Online reads K and V in place, so their last axes must be contiguous.
        const bool online_capable = dt == f32 && !with_buffer_mask()
                && key_mdw.strides()[sdpa_pd_t::ndims - 1] == 1
                && val_mdw.strides()[sdpa_pd_t::ndims - 1] == 1;
        VDISPATCH_SDPA(online_capable,
                "the online-softmax SDPA strategy was forced via "
                "ONEDNN_SDPA_IMPL but "
                "does not support this data type, attention mask or layout");
        kind_ = sdpa_impl_kind_t::online_softmax;
    } else {
        kind_ = sdpa_impl_kind_t::full_softmax;
    }

    auto scratchpad = scratchpad_registry().registrar();

    if (kind_ == sdpa_impl_kind_t::online_softmax) {
        VDISPATCH_SDPA_SC(sdpa_online_softmax::configure(*this, online_conf_),
                "sdpa_online_softmax configuration failed");
        nthr_ = online_conf_.nthr;
        scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
                online_conf_.scratch_total(nthr_), 1, 64);
        return status::success;
    }

    // Debugging/perf overrides of the full-softmax strategies:
    // ONEDNN_SDPA_SCHEDULE={per_head,kv_head_group,split_kv,auto},
    // ONEDNN_SDPA_SPLIT_KV_CHUNKS=<n> (seq_kv chunks per split_kv work item;
    // 0 or unset lets the cost model choose),
    // ONEDNN_SDPA_KV_PACKING={ahead,per_tile,auto} and
    // ONEDNN_SDPA_STACK_HEADS={0,1,auto} (1 and auto allow stacking).
    sdpa_full_softmax_knobs_t knobs;
    const std::string schedule = getenv_string_user("SDPA_SCHEDULE");
    if (schedule == "per_head")
        knobs.schedule = sdpa_full_softmax_schedule_t::per_head;
    else if (schedule == "kv_head_group")
        knobs.schedule = sdpa_full_softmax_schedule_t::kv_head_group;
    else if (schedule == "split_kv")
        knobs.schedule = sdpa_full_softmax_schedule_t::split_kv;
    else
        VDISPATCH_SDPA(
                utils::one_of(schedule, std::string(), std::string("auto")),
                "unknown ONEDNN_SDPA_SCHEDULE value '%s'", schedule.c_str());
    knobs.split_kv_chunks = getenv_int_user("SDPA_SPLIT_KV_CHUNKS", 0);
    VDISPATCH_SDPA(knobs.split_kv_chunks >= 0,
            "invalid ONEDNN_SDPA_SPLIT_KV_CHUNKS value %lld",
            static_cast<long long>(knobs.split_kv_chunks));
    const std::string kv_packing = getenv_string_user("SDPA_KV_PACKING");
    if (kv_packing == "ahead")
        knobs.kv_packing = sdpa_full_softmax_kv_packing_t::ahead;
    else if (kv_packing == "per_tile")
        knobs.kv_packing = sdpa_full_softmax_kv_packing_t::per_tile;
    else
        VDISPATCH_SDPA(
                utils::one_of(kv_packing, std::string(), std::string("auto")),
                "unknown ONEDNN_SDPA_KV_PACKING value '%s'",
                kv_packing.c_str());
    const std::string stack_heads = getenv_string_user("SDPA_STACK_HEADS");
    VDISPATCH_SDPA(utils::one_of(stack_heads, std::string(), std::string("0"),
                           std::string("1"), std::string("auto")),
            "unknown ONEDNN_SDPA_STACK_HEADS value '%s'", stack_heads.c_str());
    knobs.allow_stack_heads = stack_heads != "0";

    VDISPATCH_SDPA_SC(sdpa_full_softmax::configure(*this, knobs, full_conf_),
            "sdpa_full_softmax configuration failed");
    nthr_ = full_conf_.nthr;
    scratchpad.book(memory_tracking::names::key_sdpa_brgemm_buffer,
            full_conf_.scratch_total(nthr_), 1, 64);

    return status::success;
}

status_t brgemm_sdpa_fwd_t::init(engine_t *engine) {
    const auto isa_for_ir = mayiuse(avx512_core) ? avx512_core : avx2;

    if (pd()->impl_kind() == sdpa_impl_kind_t::online_softmax) {
        return sdpa_online_softmax::create_kernels(isa_for_ir, *pd(),
                pd()->online_softmax_conf(), engine, online_kernels_);
    } else {
        return sdpa_full_softmax::create_kernels(isa_for_ir, *pd(),
                pd()->full_softmax_conf(), engine, full_kernels_);
    }
    assert(!"Unknown SDPA implementation kind");
    return status::unimplemented;
}

namespace {
status_t get_scalar_value(
        const exec_ctx_t &ctx, const memory_desc_t &md, int arg, float &value) {
    if (md.data_type == data_type::f32) {
        value = *CTX_IN_MEM(const float *, arg);
    } else if (md.data_type == data_type::bf16) {
        value = *CTX_IN_MEM(const bfloat16_t *, arg);
    } else if (md.data_type == data_type::f16) {
        value = *CTX_IN_MEM(const float16_t *, arg);
    } else {
        return status::unimplemented;
    }
    return status::success;
}

status_t get_scalar_value(
        const exec_ctx_t &ctx, const sdpa_pd_t &pd, int arg, float &value) {
    const auto &storage = CTX_IN_STORAGE(arg);
    if (storage.is_host_scalar()) {
        const auto *host_storage
                = utils::downcast<const host_scalar_memory_storage_t *>(
                        &storage);
        return host_storage->get_scalar_value(&value, sizeof(value));
    } else {
        const auto &md = *pd.arg_md(arg);
        return get_scalar_value(ctx, md, arg, value);
    }
}
} // namespace

status_t brgemm_sdpa_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto *scratch = ctx.get_scratchpad_grantor().get<char>(
            memory_tracking::names::key_sdpa_brgemm_buffer);

    auto *q = CTX_IN_MEM(const void *, DNNL_ARG_QUERIES);
    auto *k = CTX_IN_MEM(const void *, DNNL_ARG_KEYS);
    auto *v = CTX_IN_MEM(const void *, DNNL_ARG_VALUES);
    auto *out = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    float scale_val = 1.0f;
    if (pd()->with_attn_scale()) {
        CHECK(get_scalar_value(ctx, *pd(), DNNL_ARG_SCALE, scale_val));
        if (pd()->desc()->invert_scale) scale_val = 1.0f / scale_val;
    }

    const bool with_select_mask = pd()->with_select_mask();
    const void *mask_data = with_select_mask || pd()->with_buffer_mask()
            ? CTX_IN_MEM(const void *, DNNL_ARG_ATTN_MASK)
            : nullptr;
    float fill_val = 0.0f;
    if (with_select_mask) {
        CHECK(get_scalar_value(ctx, *pd(), DNNL_ARG_ATTN_MASK_FILL, fill_val));
    }

    if (pd()->impl_kind() == sdpa_impl_kind_t::online_softmax) {
        sdpa_online_softmax_run_args_t args;
        args.q = q;
        args.k = k;
        args.v = v;
        args.cond = with_select_mask ? mask_data : nullptr;
        args.out = out;
        args.scale = scale_val;
        args.fill = fill_val;
        return sdpa_online_softmax::execute(*pd(), pd()->online_softmax_conf(),
                online_kernels_, args, scratch, pd()->nthr());
    }

    sdpa_full_softmax_run_args_t args;
    args.q = static_cast<const char *>(q);
    args.k = static_cast<const char *>(k);
    args.v = static_cast<const char *>(v);
    args.mask_data = static_cast<const char *>(mask_data);
    args.out = static_cast<char *>(out);
    args.scratch = scratch;
    args.scale = scale_val;
    args.fill = fill_val;
    return sdpa_full_softmax::execute(*pd(), pd()->full_softmax_conf(),
            full_kernels_, args, pd()->nthr());
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
