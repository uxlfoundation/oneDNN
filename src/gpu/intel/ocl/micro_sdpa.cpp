/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/ocl/micro_sdpa.hpp"
#include "gpu/intel/ocl/micro_sdpa_configs.hpp"

#include "common/c_types_map.hpp"
#include "common/sdpa_utils.hpp"
#include "common/type_helpers.hpp"
#include "gemmstone/microkernel_provider.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

namespace {

using namespace gemmstone;

/// Returns true if a common quantization value is used for each slice of the
/// tensor operation. For 4D case it's when the mask's two first bits are on
/// and two last bits are off.
/// Examples:
///   | mask      | result  |
///   |-----------+---------|
///   |  0 (0000) | true    |
///   | 12 (0011) | false   |
///   |  3 (1100) | true    |
///   |  1 (1000) | true    |
///   |  8 (0001) | false   |
bool with_quantize_common(const quant_entry_t &entry) {
    return !entry.has_default_values() && ((entry.get_mask() & 12) == 0);
}

} /* anonymous namespace */

status_t update_config_from_devenv_values(
        sdpa_config_t *config, bool quantized) {
    std::string q_config_str
            = gpu_utils::dev_getenv("QUANTIZED_SDPA_CONFIG", std::string(""));
    std::string config_str
            = gpu_utils::dev_getenv("SDPA_CONFIG", std::string(""));
    if ((!config_str.empty() && !quantized)
            || (!q_config_str.empty() && quantized)) {
        std::array<int, 8> config_values;
        int i;
        int num_values = 0;
        if (!q_config_str.empty() && quantized)
            config_str = std::move(q_config_str);

        std::stringstream ss(config_str);
        while (ss >> i) {
            config_values[num_values++] = i;
            if (ss.peek() == ',') ss.ignore();
        }
        VCHECK_SDPA_COND(num_values == 8,
                "(QUANTIZED_)SDPA_CONFIG(%s) is invalid. Must be 8 integers "
                "separate by a comma: "
                "<unroll_m_kq>,<unroll_n_kq>,<unroll_m_vs>,<unroll_n_vs>,<wg_m_"
                "kq>,<wg_n_kq>,<wg_m_vs>,<wg_n_vs>",
                config_str.c_str());
        if (num_values == 8) {
            config->unroll_m_kq = config_values[0];
            config->unroll_n_kq = config_values[1];
            config->unroll_m_vs = config_values[2];
            config->unroll_n_vs = config_values[3];
            config->wg_m_kq = config_values[4];
            config->wg_n_kq = config_values[5];
            config->wg_m_vs = config_values[6];
            config->wg_n_vs = config_values[7];
        }
    }
    return status::success;
}

status_t micro_sdpa_t::pd_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto *dev_info = compute_engine->device_info();
    arch_ = dev_info->gpu_arch();
    auto *d = desc();

    VCHECK_SDPA_COND(compute::mayiuse_microkernels(compute_engine),
            "Microkernels not supported by the OpenCL driver.");

    /* Retrieve pre-tuned kernel configuration */
    sdpa_config_t *config = nullptr;
    bool thin_q = (d->queries() <= 16);
    bool quantized = with_key_scales() || with_key_zp() || with_value_scales()
            || with_value_zp();
    bool is_integrated = compute_engine->device_info()->is_integrated();
    use_systolic_ukernel_ = compute_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    switch (arch_) {
        case arch_t::xe_hpg:
            if (!use_systolic_ukernel_) {
                config = choose_config_xehpg_fma(
                        d->head_size(), d->keys(), thin_q, quantized);
            } else {
                config = choose_config_xehpg(
                        d->head_size(), d->keys(), thin_q, quantized);
            }
            break;
        case arch_t::xe_hpc:
            config = choose_config_xehpc(d->head_size(), d->keys(), thin_q,
                    quantized, is_integrated);
            break;
        case arch_t::xe2:
        case arch_t::xe3:
            config = choose_config_xe2(d->head_size(), d->keys(), thin_q,
                    quantized, is_integrated);
        default: break;
    }

    if (!config) return status::unimplemented;

    auto status = update_config_from_devenv_values(config, quantized);
    if (status != status::success) return status;

    VDEBUGINFO(4, primitive, sdpa,
            "D=%d,K=%d,%s%s%s"
            "kq_tile(%d, %d): unroll_m=%d unroll_n=%d wg_m=%d wg_n=%d,"
            "vs_tile(%d, %d): unroll_m=%d unroll_n=%d wg_m=%d wg_n=%d",
            static_cast<int>(d->head_size()), static_cast<int>(d->keys()),
            thin_q ? "thin_q," : "", quantized ? "quant," : "",
            is_integrated ? "integrated" : "",
            config->unroll_m_kq * config->wg_m_kq,
            config->unroll_n_kq * config->wg_n_kq, config->unroll_m_kq,
            config->unroll_n_kq, config->wg_m_kq, config->wg_n_kq,
            config->unroll_m_vs * config->wg_m_vs,
            config->unroll_n_vs * config->wg_n_vs, config->unroll_m_vs,
            config->unroll_n_vs, config->wg_m_vs, config->wg_n_vs);

    VCHECK_SDPA_COND(config->unroll_n_kq * config->wg_n_kq
                            == config->unroll_n_vs * config->wg_n_vs
                    && config->unroll_n_kq % config->unroll_n_vs == 0,
            "[CONFIG] The config KQ work_group tile N(%d) axis must equal "
            "VS work_group tile N(%d) axis and KQ subgroup tile N(%d) axis "
            "must be divisible by VS subgroup tile N(%d) axis",
            config->unroll_n_kq * config->wg_n_kq,
            config->unroll_n_vs * config->wg_n_vs, config->unroll_n_kq,
            config->unroll_n_vs);

    VCHECK_SDPA_COND(config->unroll_m_vs * config->wg_m_vs >= d->head_size(),
            "The vs matmul config work_group tile M(%d*%d=%d) axis must be "
            "greater than or equal to head size(%ld)",
            config->unroll_m_vs, config->wg_m_vs,
            config->unroll_m_vs * config->wg_m_vs,
            static_cast<long int>(d->head_size()));

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel_;

    if (hw_info.gmdid == 0) return status::unimplemented;

    sg_size_ = dev_info->min_subgroup_size();

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    bool kq_common_scales = with_quantize_common(d->kq_scales);
    bool kq_common_zp = with_quantize_common(d->kq_zero_points);

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    GEMMProblem problem;
    problem.Ta_ext = jit::convert_dnnl_to_kernel_type(key_md()->data_type);
    problem.Tb_ext = jit::convert_dnnl_to_kernel_type(qry_md()->data_type);
    if (qry_md()->data_type == data_type::f16) {
        problem.Ta = problem.Tb = Type::f16;
    } else if (qry_md()->data_type == data_type::bf16) {
        problem.Ta = problem.Tb = Type::bf16;
    } else {
        VCHECK_SDPA_COND(utils::one_of(qry_md()->data_type, data_type::f16,
                                 data_type::bf16),
                "Q tensor's data type must be bf16 or f16");
    }
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.A.layout = convert_dnnl_to_kernel_layout(key_md());

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;

    if (with_key_scales() && !kq_common_scales) {
        auto scale_dt = key_scales_dt();
        problem_kq.Ta_scale = jit::convert_dnnl_to_kernel_type(scale_dt);
        problem_kq.A_scale.setAlignment(
                int8_t(d->keys() * types::data_type_size(scale_dt)));
        problem_kq.A_scale.layout = MatrixLayout::N;
        problem_kq.asPtrDims = 2;
    }
    if (with_key_zp()) {
        auto zp_dt = key_zp_dt();
        problem_kq.Tao = jit::convert_dnnl_to_kernel_type(zp_dt);
        problem_kq.AO.setAlignment(
                int8_t(d->keys() * types::data_type_size(zp_dt)));
        problem_kq.AO.layout = MatrixLayout::N;
        problem_kq.aoPtrDims = kq_common_zp ? 0 : 2;
        problem_kq.aOffset = ABOffset::Calc;
    }

    if (with_key_scales() || with_key_zp()) {
        problem_kq.aqGroupM = 1;
        problem_kq.aqGroupK
                = (kq_common_scales || kq_common_zp) ? 1 : key_group_size();
    }
    opts_kq.scaleA = with_key_scales() && !kq_common_scales;
    opts_kq.offsetA = with_key_zp();

    problem_kq.B.layout = MatrixLayout::Pr;
    problem_kq.C.layout = MatrixLayout::T;
    const memory_desc_wrapper key_mdw(key_md());
    auto ldk = static_cast<int>(
            gemm_desc_t::get_ld(*key_md()) * key_mdw.data_type_size());
    problem_kq.A.setAlignment(alignmentForLD(ldk));
    problem_kq.B.setAlignment(64); // Q is packed in VNNI format in SLM
    if (use_systolic_ukernel()) {
        problem_kq.B.crosspack = 2;
        problem_kq.B.tileR = into<uint16_t>(d_max());
        problem_kq.B.tileC = into<uint16_t>(sg_size_);
    }

    /* Set up problem size information */
    SizeParams sizes;
    sizes.m = d->keys();
    sizes.n = d->queries();
    sizes.k = d->head_size();
    sizes.batch = d->batch_size();

    /* Set up microkernel strategy */
    std::vector<StrategyRequirement> reqs_kq;
    reqs_kq.push_back(StrategyRequirement::UnrollM == config->unroll_m_kq);
    reqs_kq.push_back(StrategyRequirement::UnrollN == config->unroll_n_kq);
    reqs_kq.push_back(StrategyRequirement::WGM == config->wg_m_kq);
    reqs_kq.push_back(StrategyRequirement::WGN == config->wg_n_kq);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq_ = selectGEMMMicrokernel(
                opts_kq, hw_info, sizes, problem_kq, reqs_kq);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_kq microkernel generation failure with message: %s",
                ex.what());
    }

    bool vs_common_scales = with_quantize_common(d->vs_scales);
    bool vs_common_zp = with_quantize_common(d->vs_zero_points);

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;

    /* Update for second GEMM: V*S */
    auto problem_vs = std::move(problem);
    problem_vs.Ta_ext = jit::convert_dnnl_to_kernel_type(val_md()->data_type);
    problem_vs.A.layout = convert_dnnl_to_kernel_layout(val_md());
    if (with_value_scales() && !vs_common_scales) {
        auto scale_dt = value_scales_dt();
        problem_vs.Ta_scale = jit::convert_dnnl_to_kernel_type(scale_dt);
        problem_vs.A_scale.setAlignment(uint8_t(d->head_size()
                / value_group_size() * types::data_type_size(scale_dt)));
        problem_vs.A_scale.layout = MatrixLayout::N;
        problem_vs.asPtrDims = 2;
    }
    if (with_value_zp()) {
        auto zp_dt = value_zp_dt();
        problem_vs.Tao = jit::convert_dnnl_to_kernel_type(zp_dt);
        problem_vs.AO.setAlignment(uint8_t(d->head_size() / value_group_size()
                * types::data_type_size(zp_dt)));
        problem_vs.AO.layout = MatrixLayout::N;
        problem_vs.aoPtrDims = vs_common_zp ? 0 : 2;
        problem_vs.aOffset = ABOffset::Calc;
    }
    if (with_value_scales() || with_value_zp()) {
        problem_vs.aqGroupM = (vs_common_scales || vs_common_zp)
                ? 1
                : utils::rnd_up_pow2(value_group_size());
        problem_vs.aqGroupK = 1;
    }
    opts_vs.scaleA = with_value_scales() && !vs_common_scales;
    opts_vs.offsetA = with_value_zp();

    problem_vs.B.layout = MatrixLayout::Pr;
    problem_vs.C.layout = MatrixLayout::N;
    const memory_desc_wrapper val_mdw(val_md());
    auto ldv = static_cast<int>(
            gemm_desc_t::get_ld(*val_md()) * val_mdw.data_type_size());
    problem_vs.A.setAlignment(alignmentForLD(ldv));
    problem_vs.B.setAlignment(64); // S is packed in SLM
    if (use_systolic_ukernel()) { problem_vs.B.crosspack = 16; }
    sizes.m = d->values();
    sizes.n = gemm_kq_.getSetting("wg_tile_n");
    sizes.k = gemm_kq_.getSetting("wg_tile_m");

    /* Set up microkernel strategy */
    std::vector<StrategyRequirement> reqs_vs;
    reqs_vs.push_back(StrategyRequirement::UnrollM == config->unroll_m_vs);
    reqs_vs.push_back(StrategyRequirement::UnrollN == config->unroll_n_vs);
    reqs_vs.push_back(StrategyRequirement::WGM == config->wg_m_vs);
    reqs_vs.push_back(StrategyRequirement::WGN == config->wg_n_vs);

    /* Ask microkernel provider for microkernel */
    try {
        if (use_systolic_ukernel()) {
            auto adjust_vs = [](GEMMStrategy &strategy) {
                /* Enable dpasw */
                strategy.dpasw |= strategy.fused;
            };
            gemm_vs_ = selectGEMMMicrokernel(
                    opts_vs, hw_info, sizes, problem_vs, reqs_vs, adjust_vs);
        } else {
            gemm_vs_ = selectGEMMMicrokernel(
                    opts_vs, hw_info, sizes, problem_vs, reqs_vs);
        }
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_vs microkernel generation failure with message: %s",
                ex.what());
    }
    VDEBUGINFO(4, primitive, sdpa, "kq_gemm: %s, vs_gemm: %s,",
            problem_kq.toString().c_str(), problem_vs.toString().c_str());

    return status::success;
}

status_t micro_sdpa_t::init(impl::engine_t *engine) {
    using namespace micro;

    compute::kernel_ctx_t kernel_ctx;

    auto *d = pd()->desc();

    kernel_ctx.set_data_type(pd()->dst_md()->data_type);

    int ndims = 4;
    const memory_desc_wrapper qry_mdw(pd()->qry_md());
    const memory_desc_wrapper key_mdw(pd()->key_md());
    const memory_desc_wrapper val_mdw(pd()->val_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper msk_mdw(pd()->attn_mask_md());
    using offset_t = decltype(offsets_t().src_off);
    offset_t qry_off, key_off, val_off, dst_off, msk_off;
    set_offsets(qry_mdw, qry_off);
    set_offsets(key_mdw, key_off);
    set_offsets(val_mdw, val_off);
    set_offsets(dst_mdw, dst_off);
    set_offsets(msk_mdw, msk_off);
    def_offsets(qry_off, kernel_ctx, "QRY", ndims);
    def_offsets(key_off, kernel_ctx, "KEY", ndims);
    def_offsets(val_off, kernel_ctx, "VAL", ndims);
    def_offsets(dst_off, kernel_ctx, "DST", ndims);
    if (pd()->with_attn_mask()) {
        def_offsets(msk_off, kernel_ctx, "MSK", ndims);
    }
    kernel_ctx.define_int("NDIMS", ndims);

    def_data_type(kernel_ctx, key_mdw.data_type(), "KEY");
    def_data_type(kernel_ctx, qry_mdw.data_type(), "QRY");
    def_data_type(kernel_ctx, val_mdw.data_type(), "VAL");
    def_data_type(kernel_ctx, dst_mdw.data_type(), "DST");
    if (pd()->with_attn_mask()) {
        def_data_type(kernel_ctx, msk_mdw.data_type(), "MSK");
    }

    def_data_type(kernel_ctx, pd()->key_scales_dt(), "KEY_ATTR_SCALES");
    def_data_type(kernel_ctx, pd()->value_scales_dt(), "VAL_ATTR_SCALES");

    def_data_type(kernel_ctx, pd()->key_zp_dt(), "KEY_ATTR_ZP");
    def_data_type(kernel_ctx, pd()->value_zp_dt(), "VAL_ATTR_ZP");

    auto Q_num_heads_dim = qry_mdw.dims()[1];
    kernel_ctx.define_int("KV_GROUP_SIZE", Q_num_heads_dim / d->kv_head_number);

    auto ldq = gemm_desc_t::get_ld(*pd()->qry_md()) * qry_mdw.data_type_size();
    auto ldk = gemm_desc_t::get_ld(*pd()->key_md()) * key_mdw.data_type_size();
    auto ldv = gemm_desc_t::get_ld(*pd()->val_md()) * val_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*pd()->dst_md()) * dst_mdw.data_type_size();
    auto ldmsk = pd()->with_attn_mask()
            ? msk_mdw.dims()[3] * msk_mdw.data_type_size()
            : 0;
    kernel_ctx.define_int("Q_ALIGN", alignmentForLD(int(ldq)));
    kernel_ctx.define_int("K_ALIGN", alignmentForLD(int(ldk)));
    kernel_ctx.define_int("V_ALIGN", alignmentForLD(int(ldv)));
    kernel_ctx.define_int("A_ALIGN", alignmentForLD(int(lda)));

    kernel_ctx.define_int("TRANSPOSE_K",
            gemm_desc_t::get_trans(*pd()->key_md()) == dnnl_trans);

    int kq_scale_mask = (static_cast<int>(pd()->with_key_scales()) << 1)
            | static_cast<int>(with_quantize_common(d->kq_scales));
    kernel_ctx.define_int("KEY_SCALES", kq_scale_mask);

    int vs_scale_mask = (static_cast<int>(pd()->with_value_scales()) << 1)
            | static_cast<int>(with_quantize_common(d->vs_scales));
    kernel_ctx.define_int("VAL_SCALES", vs_scale_mask);

    int kq_zp_mask = (static_cast<int>(pd()->with_key_zp()) << 1)
            | static_cast<int>(with_quantize_common(d->kq_zero_points));
    kernel_ctx.define_int("KEY_ZERO_POINTS", kq_zp_mask);

    int vs_zp_mask = (static_cast<int>(pd()->with_value_zp()) << 1)
            | static_cast<int>(with_quantize_common(d->vs_zero_points));
    kernel_ctx.define_int("VAL_ZERO_POINTS", vs_zp_mask);

    using namespace data_type;
    auto elems_per_byte = [](data_type_t dt) {
        switch (dt) {
            case u4:
            case s4: return 2;
            default: return 1;
        }
    };
    kernel_ctx.define_int(
            "KEY_ELEMENTS_PER_BYTE", elems_per_byte(key_mdw.data_type()));
    kernel_ctx.define_int(
            "KEY_ZP_ELEMENTS_PER_BYTE", elems_per_byte(pd()->key_zp_dt()));
    kernel_ctx.define_int(
            "VAL_ELEMENTS_PER_BYTE", elems_per_byte(val_mdw.data_type()));
    kernel_ctx.define_int(
            "VAL_ZP_ELEMENTS_PER_BYTE", elems_per_byte(pd()->value_zp_dt()));

    if (pd()->with_key_scales() || pd()->with_key_zp())
        kernel_ctx.define_int("KEY_GROUP_SIZE", pd()->key_group_size());
    if (pd()->with_value_scales() || pd()->with_value_zp())
        kernel_ctx.define_int("VAL_GROUP_SIZE", pd()->value_group_size());

    def_data_type(kernel_ctx, d->scale_dt, "SCALE");
    kernel_ctx.define_int("INVERT_SCALE", d->invert_scale);
    kernel_ctx.define_int("WITH_ATTN_SCALE", pd()->with_attn_scale());
    kernel_ctx.define_int("ATTN_MASK_UNDEF", attn_mask_type::undef);
    kernel_ctx.define_int("ATTN_MASK_BUFFER", attn_mask_type::buffer);
    kernel_ctx.define_int("ATTN_MASK_TOP_LEFT", attn_mask_type::top_left);
    kernel_ctx.define_int(
            "ATTN_MASK_BOTTOM_RIGHT", attn_mask_type::bottom_right);

    kernel_ctx.define_int("WITH_ATTN_MASK",
            pd()->with_attn_mask() && !pd()->with_causal_mask());
    kernel_ctx.define_int(
            "BROADCAST_MASK_Q", msk_mdw.dims()[pd_t::mask_q_index] == 1);

    kernel_ctx.define_int("WITH_CAUSAL_MASK", pd()->with_causal_mask());

    kernel_ctx.define_int("SUBGROUP_SIZE", pd()->sg_size());
    kernel_ctx.define_int("D_MAX", pd()->d_max());

    int tile_k = pd()->gemm_kq().getSetting("wg_tile_m");
    int tile_q = pd()->gemm_kq().getSetting("wg_tile_n");
    int tile_v = pd()->gemm_vs().getSetting("wg_tile_m");

    bool d_full = (d->head_size() == pd()->d_max());
    bool v_full = (d->head_size() == tile_v);
    bool k_full = ((d->keys() % tile_k) == 0);

    kernel_ctx.define_int("REMAINDER_K", !k_full);

    auto &gemm_vs = pd()->gemm_vs();
    int unroll_m_vs = gemm_vs.getSetting("sg_tile_m");
    if (d_full) {
        if (ldq % 4 == 0) kernel_ctx.define_int("BLOCK_Q", 1);
        if (lda % 4 == 0 && v_full) kernel_ctx.define_int("BLOCK_A", 1);
        if (ldmsk % 4 == 0) kernel_ctx.define_int("BLOCK_MSK", 1);
        kernel_ctx.define_int("REMAINDER_Q", (d->queries() % tile_q) != 0);
    } else if (pd()->arch() >= compute::gpu_arch_t::xe_hpc
            && unroll_m_vs < 64) {
        auto vbytes = d->values() * val_mdw.data_type_size();
        if (lda % 16 == 0 && vbytes % 4 == 0)
            kernel_ctx.define_int("BLOCK_2D_A", 1);
    }

    if (pd()->arch() >= compute::gpu_arch_t::xe_hpc) {
        kernel_ctx.define_int("PREFETCH_MASK", 1);
        kernel_ctx.define_int("PREFETCH_K0", 1);
        kernel_ctx.define_int("PREFETCH_K", 1);
        kernel_ctx.define_int("PREFETCH_V", 1);
        bool no_rem = d_full && v_full && (d->keys() % tile_k == 0);
        kernel_ctx.define_int("PREFETCH_REMAINDER", !no_rem);
        kernel_ctx.define_int("PREFETCH_D_MAX", nstl::min(pd()->d_max(), 64));
    }

    kernel_ctx.define_int("Q_ARRIVE_AWAIT_BARRIER", d->queries() > 1);

    kernel_ctx.define_int("SOFTMAX_INF_AS_ZERO",
            d->softmax_alg == alg_kind::softmax_accurate_inf_as_zero);

    kernel_ctx.define_int("USE_SYSTOLIC_UKERNEL", pd()->use_systolic_ukernel());
    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = pd()->sg_size();
    shimOptions.useTileOps = true;
    shimOptions.decorator = "kq";

    kernel_ctx.add_custom_header("gemm_kq.h",
            micro::generateShim(
                    pd()->gemm_kq(), HostLanguage::OpenCL_C, shimOptions));

    shimOptions.microkernelID++;
    shimOptions.decorator = "vs";

    kernel_ctx.add_custom_header("gemm_vs.h",
            micro::generateShim(
                    pd()->gemm_vs(), HostLanguage::OpenCL_C, shimOptions));

    if (pd()->gemm_kq().grfMin > 128 || pd()->gemm_vs().grfMin > 128)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    CHECK(create_kernel(engine, &kernel_, "micro_sdpa", kernel_ctx));
    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t micro_sdpa_t::execute(const exec_ctx_t &ctx) const {
    const auto &qry = CTX_IN_STORAGE(DNNL_ARG_QUERIES);
    const auto &key = CTX_IN_STORAGE(DNNL_ARG_KEYS);
    const auto &val = CTX_IN_STORAGE(DNNL_ARG_VALUES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    const auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    const auto &attn_mask = CTX_IN_STORAGE(DNNL_ARG_ATTN_MASK);

    const auto &key_scales
            = CTX_IN_STORAGE(DNNL_ARG_KEYS | DNNL_ARG_ATTR_SCALES);
    const auto &key_zp
            = CTX_IN_STORAGE(DNNL_ARG_KEYS | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &value_scales
            = CTX_IN_STORAGE(DNNL_ARG_VALUES | DNNL_ARG_ATTR_SCALES);
    const auto &value_zp
            = CTX_IN_STORAGE(DNNL_ARG_VALUES | DNNL_ARG_ATTR_ZERO_POINTS);
    const dim_t Q = pd()->desc()->queries();
    const dim_t K = pd()->desc()->keys();
    const dim_t D = pd()->desc()->head_size();

    auto &gemm_kq = pd()->gemm_kq();
    auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
    auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m")
            * gemm_kq.getSetting("sg_per_wg_n");

    int mask_type = static_cast<int>(pd()->desc()->mask_type);
    compute::kernel_arg_list_t arg_list;
    arg_list.append(key);
    arg_list.append(qry);
    arg_list.append(val);
    arg_list.append(dst);
    arg_list.append(scale);
    arg_list.append((int)D);
    arg_list.append((int)K);
    arg_list.append((int)Q);
    arg_list.append(key_scales);
    arg_list.append(key_zp);
    arg_list.append(value_scales);
    arg_list.append(value_zp);
    arg_list.append(mask_type);
    if (pd()->with_attn_mask()) arg_list.append(attn_mask);

    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(Q, wg_tile_q);
    gws[1] *= pd()->dst_md()->dims[1];
    gws[2] *= pd()->dst_md()->dims[0];

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
