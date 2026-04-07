/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/sdpa/micro.hpp"
#include "gpu/intel/sdpa/configs.hpp"

#include "common/c_types_map.hpp"
#include "common/sdpa_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/utils.hpp"

#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

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
        fwd_config_t *config, bool quantized) {
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

        stringstream_t ss(config_str);
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

status_t update_config_from_devenv_values(bwd_config_t *config) {
    std::string bwd_config_str
            = gpu_utils::dev_getenv("BWD_SDPA_CONFIG", std::string(""));
    if (!bwd_config_str.empty()) {
        std::array<int, 12> config_values;
        int i;
        int num_values = 0;

        stringstream_t ss(bwd_config_str);
        while (ss >> i) {
            config_values[num_values++] = i;
            if (ss.peek() == ',') ss.ignore();
        }
        VCHECK_SDPA_COND(num_values == 12,
                "BWD_SDPA_CONFIG(%s) is invalid. Must be 12 integers "
                "separate by a comma: "
                "<unroll_m_BcBr>,<unroll_n_BcBr>,"
                "<unroll_m_DBc>,<unroll_n_DBc>,"
                "<unroll_m_DBr>,<unroll_n_DBr>,"
                "<wg_m_BcBr>,<wg_n_BcBr>,"
                "<wg_m_DBc>,<wg_n_DBc>,"
                "<wg_m_DBr>,<wg_n_DBr>",
                bwd_config_str.c_str());
        if (num_values == 12) {
            config->unroll_m_BcBr = config_values[0];
            config->unroll_n_BcBr = config_values[1];
            config->unroll_m_DBc = config_values[2];
            config->unroll_n_DBc = config_values[3];
            config->unroll_m_DBr = config_values[4];
            config->unroll_n_DBr = config_values[5];
            config->wg_m_BcBr = config_values[6];
            config->wg_n_BcBr = config_values[7];
            config->wg_m_DBc = config_values[8];
            config->wg_n_DBc = config_values[9];
            config->wg_m_DBr = config_values[10];
            config->wg_n_DBr = config_values[11];
        }
    }
    return status::success;
}

status_t micro_fwd_t::pd_t::init_conf_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    arch_ = dev_info->gpu_arch();
    auto *d = desc();

    VCHECK_SDPA_COND(compute::mayiuse_microkernels(intel_engine),
            "Microkernels not supported by the OpenCL driver.");

    /* Retrieve pre-tuned kernel configuration */
    fwd_config_t *config = nullptr;
    const dim_t thin_q_threshold = 16;
    auto queries = d->queries();
    if (queries == 1) { queries = (d->q_desc.dims[1] / d->num_kv_heads()); }

    bool thin_q = (queries <= thin_q_threshold);
    bool quantized = with_key_scales() || with_key_zp() || with_value_scales()
            || with_value_zp();
    bool is_integrated = intel_engine->device_info()->is_integrated();
    bool is_f32 = (desc()->qry_md()->data_type == data_type::f32);
    use_systolic_ukernel_
            = intel_engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            && !is_f32; // f32 -> non-systolic kernel only

    bool use_fma_config = !use_systolic_ukernel_;
    bool is_f16_accumulate_gemm = (kq_acc_dt() == data_type::f16)
            || (vs_acc_dt() == data_type::f16);
    VCHECK_SDPA_COND(
            IMPLICATION(is_f16_accumulate_gemm, !use_systolic_ukernel_),
            "f16 accumulate only available with FMA matmul."); //TODO: update once matmul primitive supports systolic f16 accumulate for testing
    config = choose_config(arch_, d->head_size(), d->keys(), thin_q, quantized,
            is_integrated, use_fma_config, is_f32, is_f16_accumulate_gemm);

    VCHECK_SDPA_COND(config != nullptr,
            "No suitable kernel configuration found for the given problem "
            "size and attributes.");

    CHECK(update_config_from_devenv_values(config, quantized));

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

    // serializable minimal set of configuration params for ukernels
    // will be used to generate shim ukernels in reusable kernel_ctx
    micro_fwd_ukernel_params_t ukernel_params;
    ukernel_params.unroll_m_kq = config->unroll_m_kq;
    ukernel_params.unroll_n_kq = config->unroll_n_kq;
    ukernel_params.unroll_m_vs = config->unroll_m_vs;
    ukernel_params.unroll_n_vs = config->unroll_n_vs;

    ukernel_params.wg_m_kq = config->wg_m_kq;
    ukernel_params.wg_n_kq = config->wg_n_kq;
    ukernel_params.wg_m_vs = config->wg_m_vs;
    ukernel_params.wg_n_vs = config->wg_n_vs;

    /* Get device information */
    micro::HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel_;

    VDISPATCH_SDPA(
            hw_info.gmdid != 0, "gmdid is 0, microkernels not supported.");

    ukernel_params.hwinfo = {hw_info};

    sg_size_ = dev_info->min_subgroup_size();

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    bool kq_common_scales = with_quantize_common(d->kq_scales);
    bool kq_common_zp = with_quantize_common(d->kq_zero_points);

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(desc()->key_md()->data_type);
    problem.Tb_ext = convert_dnnl_to_kernel_type(desc()->qry_md()->data_type);
    if (desc()->qry_md()->data_type == data_type::f16) {
        problem.Ta = problem.Tb = Type::f16;
    } else if (desc()->qry_md()->data_type == data_type::bf16) {
        problem.Ta = problem.Tb = Type::bf16;
    } else if (desc()->qry_md()->data_type == data_type::f32) {
        problem.Ta = problem.Tb = Type::f32;
    } else {
        VCHECK_SDPA_COND(utils::one_of(desc()->qry_md()->data_type,
                                 data_type::f16, data_type::bf16),
                "Q tensor's data type must be bf16 or f16");
    }
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;

    problem_kq.Tc = problem_kq.Ts
            = (kq_acc_dt() == data_type::f16) ? Type::f16 : Type::f32;

    problem_kq.A.layout = convert_dnnl_to_kernel_layout(desc()->key_md());

    if (with_key_scales() && !kq_common_scales) {
        auto scale_dt = key_scales_dt();
        problem_kq.Ta_scale = convert_dnnl_to_kernel_type(scale_dt);
        problem_kq.A_scale.setAlignment(
                int8_t(d->keys() * types::data_type_size(scale_dt)));
        problem_kq.A_scale.layout = MatrixLayout::N;
        const int matrix_scale = 2;
        problem_kq.asPtrDims = matrix_scale;
    }
    if (with_key_zp()) {
        auto zp_dt = key_zp_dt();
        problem_kq.Tao = convert_dnnl_to_kernel_type(zp_dt);
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

    problem_kq.B.layout = MatrixLayout::Pr;
    problem_kq.C.layout = MatrixLayout::T;
    const memory_desc_wrapper key_mdw(desc()->key_md());
    auto ldk = static_cast<int>(
            gemm_desc_t::get_ld(*desc()->key_md()) * key_mdw.data_type_size());
    problem_kq.A.setAlignment(micro::alignmentForLD(int(ldk)));
    problem_kq.B.setAlignment(64); // Q is packed in VNNI format in SLM
    if (use_systolic_ukernel()) {
        problem_kq.B.crosspack = 2;
        problem_kq.B.tileR = into<uint16_t>(d_max());
        problem_kq.B.tileC = into<uint16_t>(sg_size_);
    }

    ukernel_params.problem_kq = {problem_kq};

    /* Set up microkernel options */
    micro::GEMMOptions opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;
    opts_kq.scaleA = with_key_scales() && !kq_common_scales;
    opts_kq.offsetA = with_key_zp();

    ukernel_params.opts_kq = {opts_kq};

    /* Set up problem size information */
    SizeParams heuristic_sizes;
    // quantizing sizes to large intervals allows kernel
    // selection search while avoiding recompilation for every new size
    heuristic_sizes.m = nearest_conf_seq_interval(arch_, d->head_size(),
            d->keys(), thin_q, quantized, is_integrated, use_fma_config, is_f32,
            is_f16_accumulate_gemm);
    // query size is only tuned to thin_q/non-thin_q cases
    heuristic_sizes.n = (queries <= thin_q_threshold)
            ? thin_q_threshold
            : utils::rnd_up_pow2(queries);
    heuristic_sizes.k
            = d->head_size(); // baked into kernel regardless, no quantization
    heuristic_sizes.batch = utils::rnd_up_pow2(d->batch() * d->num_q_heads());

    ukernel_params.sizes_kq = {heuristic_sizes};

    /* Set up GEMMProblem structure for second GEMM: V * S  */
    auto problem_vs = std::move(problem);
    problem_vs.Tc = problem_vs.Ts
            = (vs_acc_dt() == data_type::f16) ? Type::f16 : Type::f32;

    bool vs_common_scales = with_quantize_common(d->vs_scales);
    bool vs_common_zp = with_quantize_common(d->vs_zero_points);

    problem_vs.Ta_ext
            = convert_dnnl_to_kernel_type(desc()->val_md()->data_type);
    problem_vs.A.layout = convert_dnnl_to_kernel_layout(desc()->val_md());
    if (with_value_scales() && !vs_common_scales) {
        auto scale_dt = value_scales_dt();
        problem_vs.Ta_scale = convert_dnnl_to_kernel_type(scale_dt);
        problem_vs.A_scale.setAlignment(uint8_t(d->head_size()
                / value_group_size() * types::data_type_size(scale_dt)));
        problem_vs.A_scale.layout = MatrixLayout::N;
        const int matrix_scale = 2;
        problem_vs.asPtrDims = matrix_scale;
    }
    if (with_value_zp()) {
        auto zp_dt = value_zp_dt();
        problem_vs.Tao = convert_dnnl_to_kernel_type(zp_dt);
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

    problem_vs.B.layout = MatrixLayout::Pr;
    problem_vs.C.layout = MatrixLayout::N;
    const memory_desc_wrapper val_mdw(desc()->val_md());
    auto ldv = static_cast<int>(
            gemm_desc_t::get_ld(*desc()->val_md()) * val_mdw.data_type_size());
    problem_vs.A.setAlignment(micro::alignmentForLD(int(ldv)));
    problem_vs.B.setAlignment(64); // S is packed in SLM
    if (use_systolic_ukernel()) { problem_vs.B.crosspack = 16; }

    ukernel_params.problem_vs = {problem_vs};

    // directly tied to config, will recompile w/head size and config updates
    // no need for interval quantization
    heuristic_sizes.m = d->values();
    const int wg_tile_n = config->wg_n_kq * config->unroll_n_kq;
    const int wg_tile_m = config->wg_m_kq * config->unroll_m_kq;
    heuristic_sizes.n = wg_tile_n;
    heuristic_sizes.k = wg_tile_m;

    ukernel_params.sizes_vs = {heuristic_sizes};

    /* Set up microkernel options */
    micro::GEMMOptions opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;
    opts_vs.scaleA = with_value_scales() && !vs_common_scales;
    opts_vs.offsetA = with_value_zp();

    ukernel_params.opts_vs = {opts_vs};

    conf.ukernel_config = ukernel_params;

    return status::success;
}

status_t micro_bwd_t::pd_t::init_conf_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    arch_ = dev_info->gpu_arch();
    auto *d = desc();

    VCHECK_SDPA_COND(compute::mayiuse_microkernels(intel_engine),
            "Microkernels not supported by the OpenCL driver.");

    /* Retrieve pre-tuned kernel configuration */
    bwd_config_t *config = nullptr;
    const dim_t thin_q_threshold = 16;
    auto queries = d->queries();
    // TODO: q=1 batch group optimizations
    // if (queries == 1) { queries = (d->q_desc.dims[1] / d->num_kv_heads()); }

    bool thin_q = (queries <= thin_q_threshold);
    bool quantized = false;
    bool is_integrated = intel_engine->device_info()->is_integrated();
    bool is_f32 = (desc()->qry_md()->data_type == data_type::f32);
    use_systolic_ukernel_
            = intel_engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            && !is_f32; // f32 -> non-systolic kernel only

    bool use_fma_config = !use_systolic_ukernel_;
    config = choose_bwd_config(arch_, d->head_size(), d->queries(), d->keys(),
            thin_q, quantized, is_integrated, use_fma_config, is_f32);

    VCHECK_SDPA_COND(config != nullptr,
            "No suitable kernel configuration found for the given problem "
            "size and attributes.");

    CHECK(update_config_from_devenv_values(config));

    VDEBUGINFO(4, primitive, sdpa,
            "D=%d,K=%d,%s%s%s"
            "BcBr_tile(%d, %d): unroll_m=%d unroll_n=%d wg_m=%d wg_n=%d,"
            "DBc_tile(%d, %d): unroll_m=%d unroll_n=%d wg_m=%d wg_n=%d"
            "DBr_tile(%d, %d): unroll_m=%d unroll_n=%d wg_m=%d wg_n=%d",
            static_cast<int>(d->head_size()), static_cast<int>(d->keys()),
            thin_q ? "thin_q," : "", quantized ? "quant," : "",
            is_integrated ? "integrated" : "",
            config->unroll_m_BcBr * config->wg_m_BcBr,
            config->unroll_n_BcBr * config->wg_n_BcBr, config->unroll_m_BcBr,
            config->unroll_n_BcBr, config->wg_m_BcBr, config->wg_n_BcBr,
            config->unroll_m_DBc * config->wg_m_DBc,
            config->unroll_n_DBc * config->wg_n_DBc, config->unroll_m_DBc,
            config->unroll_n_DBc, config->wg_m_DBc, config->wg_n_DBc,
            config->unroll_m_DBr * config->wg_m_DBr,
            config->unroll_n_DBr * config->wg_n_DBr, config->unroll_m_DBr,
            config->unroll_n_DBr, config->wg_m_DBr, config->wg_n_DBr);

    // Bc(Br) == (D)Bc
    VCHECK_SDPA_COND(
            ((config->unroll_m_BcBr * config->wg_m_BcBr
                     == config->unroll_n_DBc * config->wg_n_DBc)
                    && ((config->wg_m_DBc * config->wg_n_DBc)
                            <= (config->wg_m_BcBr * config->wg_n_BcBr))),
            "[CONFIG] The config BcBr work_group tile M(%d) axis must equal "
            "DBc work_group tile N(%d) axis and number of total subgroups "
            "should be less than BcBr subgroups (%d ?<= %d)",
            config->unroll_m_BcBr * config->wg_m_BcBr,
            config->unroll_n_DBc * config->wg_n_DBc,
            config->wg_m_DBc * config->wg_n_DBc,
            config->wg_m_BcBr * config->wg_n_BcBr);

    // D(Bc) >= head size
    VCHECK_SDPA_COND(config->unroll_m_DBc * config->wg_m_DBc >= d->head_size(),
            "The DBc matmul config work_group tile N(%d*%d=%d) axis must be "
            "greater than or equal to head size(%ld)",
            config->unroll_m_DBc, config->wg_m_DBc,
            config->unroll_m_DBc * config->wg_m_DBc,
            static_cast<long int>(d->head_size()));

    // (Bc)Br == (D)Br, ngroups <= BcBr ngroups
    VCHECK_SDPA_COND(((config->unroll_n_BcBr * config->wg_n_BcBr
                              == config->unroll_n_DBr * config->wg_n_DBr)
                             && (config->wg_m_DBr * config->wg_n_DBr
                                     <= config->wg_m_BcBr * config->wg_n_BcBr)),
            "[CONFIG] The config BcBr work_group tile N(%d) axis must equal "
            "DBr work_group tile N(%d) axis and number of total subgroups "
            "should be less than BcBr subgroups (%d ?<= %d)",
            config->unroll_n_BcBr * config->wg_n_BcBr,
            config->unroll_n_DBr * config->wg_n_DBr,
            config->wg_m_DBr * config->wg_n_DBr,
            config->wg_m_BcBr * config->wg_n_BcBr);

    // D(Br) >= head size
    VCHECK_SDPA_COND(config->unroll_m_DBr * config->wg_m_DBr >= d->head_size(),
            "The DBr matmul config work_group tile M(%d*%d=%d) axis must be "
            "greater than or equal to head size(%ld)",
            config->unroll_m_DBr, config->wg_m_DBr,
            config->unroll_m_DBr * config->wg_m_DBr,
            static_cast<long int>(d->head_size()));

    // serializable minimal set of configuration params for ukernels
    // will be used to generate shim ukernels in reusable kernel_ctx
    micro_bwd_ukernel_params_t ukernel_params;

    ukernel_params.unroll_m_BcBr = config->unroll_m_BcBr;
    ukernel_params.unroll_n_BcBr = config->unroll_n_BcBr;

    ukernel_params.unroll_m_DBc = config->unroll_m_DBc;
    ukernel_params.unroll_n_DBc = config->unroll_n_DBc;

    ukernel_params.unroll_m_DBr = config->unroll_m_DBr;
    ukernel_params.unroll_n_DBr = config->unroll_n_DBr;

    ukernel_params.wg_m_BcBr = config->wg_m_BcBr;
    ukernel_params.wg_n_BcBr = config->wg_n_BcBr;

    ukernel_params.wg_m_DBc = config->wg_m_DBc;
    ukernel_params.wg_n_DBc = config->wg_n_DBc;

    ukernel_params.wg_m_DBr = config->wg_m_DBr;
    ukernel_params.wg_n_DBr = config->wg_n_DBr;

    /* Get device information */
    micro::HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel_;

    VDISPATCH_SDPA(
            hw_info.gmdid != 0, "gmdid is 0, microkernels not supported.");

    ukernel_params.hwinfo = {hw_info};

    sg_size_ = dev_info->min_subgroup_size();

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };
    auto transpose_layout = [](const gemmstone::MatrixLayout l) {
        switch (l) {
            case MatrixLayout::N: return MatrixLayout::T;
            case MatrixLayout::T: return MatrixLayout::N;
            case MatrixLayout::Pr: return MatrixLayout::Pc;
            case MatrixLayout::Pc: return MatrixLayout::Pr;
            default: return l;
        }
    };

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(desc()->key_md()->data_type);
    problem.Tb_ext = convert_dnnl_to_kernel_type(desc()->qry_md()->data_type);
    if (desc()->qry_md()->data_type == data_type::f16) {
        problem.Ta = problem.Tb = Type::f16;
    } else if (desc()->qry_md()->data_type == data_type::bf16) {
        problem.Ta = problem.Tb = Type::bf16;
    } else if (desc()->qry_md()->data_type == data_type::f32) {
        problem.Ta = problem.Tb = Type::f32;
    } else {
        VCHECK_SDPA_COND(
                utils::one_of(desc()->qry_md()->data_type, data_type::f16,
                        data_type::bf16, data_type::f32),
                "Q tensor's data type must be bf16, f16, or f32");
    }
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;

    const int wg_tile_m_BcBr = config->wg_m_BcBr * config->unroll_m_BcBr;
    const int wg_tile_n_BcBr = config->wg_n_BcBr * config->unroll_n_BcBr;

    auto problem_kq = problem;

    problem_kq.A.layout = MatrixLayout::Pc;
    problem_kq.B.layout = MatrixLayout::N;
    problem_kq.C.layout = MatrixLayout::N;
    const memory_desc_wrapper key_mdw(desc()->key_md());
    const memory_desc_wrapper qry_mdw(desc()->qry_md());
    auto ldk = static_cast<int>(
            gemm_desc_t::get_ld(*desc()->key_md()) * key_mdw.data_type_size());
    auto ldq = static_cast<int>(
            gemm_desc_t::get_ld(*desc()->qry_md()) * qry_mdw.data_type_size());
    problem_kq.A.setAlignment(64); // Q is packed in VNNI format in SLM
    if (use_systolic_ukernel()) {
        problem_kq.A.crosspack = 2;
        problem_kq.A.tileR = into<uint16_t>(sg_size_);
        problem_kq.A.tileC = into<uint16_t>(d_max());
    }
    problem_kq.B.setAlignment(micro::alignmentForLD(int(ldq)));

    ukernel_params.problem_kq = {problem_kq};

    /* Set up microkernel options */
    micro::GEMMOptions opts_kq;
    opts_kq.localA = true;
    opts_kq.slmPtr = true;
    opts_kq.scaleA = false;
    opts_kq.offsetA = false;

    ukernel_params.opts_kq = {opts_kq};

    /* Set up problem size information */
    SizeParams heuristic_sizes;
    heuristic_sizes.m = wg_tile_m_BcBr;
    heuristic_sizes.n = wg_tile_n_BcBr;
    heuristic_sizes.k = d->head_size();
    heuristic_sizes.batch = 1;

    ukernel_params.sizes_kq = {heuristic_sizes};

    /* Set up GEMMProblem structure for second GEMM: V * S  */
    auto problem_vs = problem;
    problem_vs.Tc = problem_vs.Ts
            = (vs_acc_dt() == data_type::f16) ? Type::f16 : Type::f32;

    problem_vs.Ta_ext
            = convert_dnnl_to_kernel_type(desc()->val_md()->data_type);
    problem_vs.A.layout = convert_dnnl_to_kernel_layout(diff_dst_md());
    problem_vs.B.layout = MatrixLayout::Pr;
    problem_vs.C.layout = MatrixLayout::N;
    const memory_desc_wrapper diff_dst_mdw(diff_dst_md());
    auto lda = static_cast<int>(gemm_desc_t::get_ld(*diff_dst_md())
            * diff_dst_mdw.data_type_size());
    problem_vs.A.setAlignment(micro::alignmentForLD(int(lda)));
    problem_vs.B.setAlignment(64); // S is packed in SLM
    if (use_systolic_ukernel()) { problem_vs.B.crosspack = 16; }

    ukernel_params.problem_vs = {problem_vs};

    // directly tied to config, will recompile w/head size and config updates
    // no need for interval quantization
    heuristic_sizes.m = d->head_size();
    heuristic_sizes.n = wg_tile_m_BcBr;
    heuristic_sizes.k = wg_tile_n_BcBr;

    ukernel_params.sizes_vs = {heuristic_sizes};

    /* Set up microkernel options */
    micro::GEMMOptions opts_vs;
    opts_vs.localA = false;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;

    ukernel_params.opts_vs = {opts_vs};

    //////// Vt * dA
    auto problem_vtdA = problem;
    problem_vtdA.Ta_ext
            = convert_dnnl_to_kernel_type(desc()->val_md()->data_type);

    problem_vtdA.A.layout
            = transpose_layout(convert_dnnl_to_kernel_layout(desc()->val_md()));
    problem_vtdA.B.layout = convert_dnnl_to_kernel_layout(diff_dst_md());
    problem_vtdA.C.layout = MatrixLayout::N;
    const memory_desc_wrapper val_mdw(desc()->val_md());
    auto ldv
            = gemm_desc_t::get_ld(*desc()->val_md()) * val_mdw.data_type_size();
    problem_vtdA.A.setAlignment(micro::alignmentForLD(int(ldv)));
    problem_vtdA.B.setAlignment(micro::alignmentForLD(int(lda)));

    ukernel_params.problem_vtdA = {problem_vtdA};

    heuristic_sizes.m = wg_tile_m_BcBr;
    heuristic_sizes.n = wg_tile_n_BcBr;
    heuristic_sizes.k = d->head_size();

    ukernel_params.sizes_vtdA = {heuristic_sizes};

    /* Set up microkernel options */
    micro::GEMMOptions opts_vtdA;
    opts_vtdA.localA = false;
    opts_vtdA.localB = false;
    opts_vtdA.slmPtr = true;
    ukernel_params.opts_vtdA = {opts_vtdA};

    //////// Q * dS^t
    auto problem_qdSt = problem;
    problem_qdSt.Ta_ext
            = convert_dnnl_to_kernel_type(desc()->qry_md()->data_type);
    problem_qdSt.A.layout = MatrixLayout::Pc;
    problem_qdSt.B.layout
            = transpose_layout(convert_dnnl_to_kernel_layout(desc()->qry_md()));
    problem_qdSt.C.layout = MatrixLayout::N;

    problem_qdSt.A.setAlignment(64);
    problem_qdSt.B.setAlignment(micro::alignmentForLD(int(ldq)));
    if (use_systolic_ukernel()) {
        problem_qdSt.A.crosspack = 2;
        problem_qdSt.A.tileR = into<uint16_t>(
                sg_size_); // tile will be transposed (dS^t -> n x m)
        problem_qdSt.A.tileC = into<uint16_t>(wg_tile_n_BcBr);
    }

    ukernel_params.problem_qdSt = {problem_qdSt};

    heuristic_sizes.m = wg_tile_m_BcBr;
    heuristic_sizes.n = d->values();
    heuristic_sizes.k = wg_tile_n_BcBr;

    ukernel_params.sizes_qdSt = {heuristic_sizes};

    /* Set up microkernel options */
    micro::GEMMOptions opts_qdSt;
    opts_qdSt.localA = true;
    opts_qdSt.localB = false;
    opts_qdSt.slmPtr = true;
    ukernel_params.opts_qdSt = {opts_qdSt};

    // dS * K
    auto problem_ktq = std::move(problem);
    problem_ktq.Ta_ext
            = convert_dnnl_to_kernel_type(desc()->key_md()->data_type);

    problem_ktq.A.layout
            = transpose_layout(convert_dnnl_to_kernel_layout(desc()->key_md()));
    problem_ktq.B.layout = MatrixLayout::Pr;
    problem_ktq.C.layout = MatrixLayout::N;

    problem_ktq.A.setAlignment(micro::alignmentForLD(int(ldk)));
    problem_ktq.B.setAlignment(64); // S is packed in SLM
    if (use_systolic_ukernel()) { problem_ktq.B.crosspack = 16; }

    ukernel_params.problem_ktq = {problem_ktq};

    heuristic_sizes.m = d->head_size();
    heuristic_sizes.n = wg_tile_n_BcBr;
    heuristic_sizes.k = wg_tile_m_BcBr;

    ukernel_params.sizes_ktq = {heuristic_sizes};

    /* Set up microkernel options */
    micro::GEMMOptions opts_ktq;
    opts_ktq.localA = false;
    opts_ktq.localB = true;
    opts_ktq.slmPtr = true;
    ukernel_params.opts_ktq = {opts_ktq};

    conf.ukernel_config = ukernel_params;

    return status::success;
}

status_t micro_fwd_t::init(impl::engine_t *engine) {
    CHECK(create_kernel(
            engine, kernel_, pd()->conf.get_kernel_names()[0], pd()->conf));

    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t micro_bwd_t::init(impl::engine_t *engine) {
    std::vector<const char *> kernel_names = pd()->conf.get_kernel_names();

    std::vector<compute::kernel_t> kernels;
    CHECK(create_kernels(engine, kernels, kernel_names, pd()->conf));

    preprocess_ = kernels[0];
    kernel_ = kernels[1];
    postprocess_ = kernels[2];

    if (!preprocess_) return status::runtime_error;
    if (!kernel_) return status::runtime_error;
    if (!postprocess_) return status::runtime_error;
    return status::success;
}

template <typename conf_t, typename pd_type>
static void init_conf_common(conf_t &conf, pd_type *pd) {
    using pd_t = sdpa_pd_t;
    auto *d = pd->desc();

    data_type_t data_t = pd->dst_md()->data_type;
    conf.data_t = data_t;
    conf.ndims = pd_t::ndims;

    const memory_desc_wrapper qry_mdw(pd->desc()->qry_md());
    const memory_desc_wrapper key_mdw(pd->desc()->key_md());
    const memory_desc_wrapper val_mdw(pd->desc()->val_md());
    const memory_desc_wrapper dst_mdw(pd->dst_md());
    const memory_desc_wrapper msk_mdw(pd->desc()->attn_mask_md());

    conf.key_data_t = key_mdw.data_type();
    conf.qry_data_t = qry_mdw.data_type();
    conf.val_data_t = val_mdw.data_type();
    conf.dst_data_t = dst_mdw.data_type();

    conf.msk_data_t = data_type::undef;
    if (pd->with_attn_mask()) { conf.msk_data_t = msk_mdw.data_type(); }

    auto Q_num_heads_dim = qry_mdw.dims()[1];
    conf.kv_group_size = static_cast<int>(Q_num_heads_dim / d->num_kv_heads());

    auto ldq = gemm_desc_t::get_ld(*pd->desc()->qry_md())
            * qry_mdw.data_type_size();
    auto ldk = gemm_desc_t::get_ld(*pd->desc()->key_md())
            * key_mdw.data_type_size();
    auto ldv = gemm_desc_t::get_ld(*pd->desc()->val_md())
            * val_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*pd->dst_md()) * dst_mdw.data_type_size();

    conf.q_align = micro::alignmentForLD(int(ldq));
    conf.k_align = micro::alignmentForLD(int(ldk));
    conf.v_align = micro::alignmentForLD(int(ldv));
    conf.a_align = micro::alignmentForLD(int(lda));

    conf.transpose_k
            = gemm_desc_t::get_trans(*pd->desc()->key_md()) == dnnl_trans;

    conf.scale_data_t = pd->desc()->scale_md()->data_type;

    conf.attn_mask_undef = attn_mask_type::undef;
    conf.attn_mask_buffer = attn_mask_type::buffer;
    conf.attn_mask_top_left = attn_mask_type::top_left;
    conf.attn_mask_bottom_right = attn_mask_type::bottom_right;

    conf.invert_scale = d->invert_scale;
    conf.with_attn_scale = pd->with_attn_scale();
    conf.with_host_scale = pd->with_host_scale();
    conf.with_attn_mask = (pd->with_attn_mask() && !pd->with_causal_mask());
    conf.broadcast_mask_q = (msk_mdw.dims()[pd_t::mask_q_index] == 1);
    conf.with_causal_mask = pd->with_causal_mask();

    conf.subgroup_size = pd->sg_size();
    conf.d_max = pd->d_max();

    bool d_full = (d->head_size() == pd->d_max());
    conf.d_full = d_full;
    conf.arch_gte_hpc = (pd->arch() >= compute::gpu_arch_t::xe_hpc);
    conf.dropout = !pd->attr()->dropout_.has_default_values();
    conf.dropout_output_mask = pd->attr()->dropout_.has_output_mask();
    conf.dropout_offset = pd->attr()->dropout_.use_offset_;
    conf.dropout_host_scalars = pd->attr()->dropout_.use_host_scalars_;
    conf.use_systolic_ukernel = pd->use_systolic_ukernel();
}

status_t micro_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    using namespace micro;
    init_conf_common(conf, this);

    conf.require_stateless_addressing = has_large_buffers();

    const memory_desc_wrapper qry_mdw(desc()->qry_md());
    const memory_desc_wrapper key_mdw(desc()->key_md());
    const memory_desc_wrapper val_mdw(desc()->val_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    conf.key_scales_data_t = key_scales_dt();
    conf.value_scales_data_t = value_scales_dt();

    conf.key_zp_data_t = key_zp_dt();
    conf.value_zp_data_t = value_zp_dt();

    auto ldq
            = gemm_desc_t::get_ld(*desc()->qry_md()) * qry_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*dst_md()) * dst_mdw.data_type_size();

    int kq_scale_mask = (static_cast<int>(with_key_scales()) << 1)
            | static_cast<int>(with_quantize_common(desc()->kq_scales));
    conf.kq_scale_mask = kq_scale_mask;

    int vs_scale_mask = (static_cast<int>(with_value_scales()) << 1)
            | static_cast<int>(with_quantize_common(desc()->vs_scales));
    conf.vs_scale_mask = vs_scale_mask;

    int kq_zp_mask = (static_cast<int>(with_key_zp()) << 1)
            | static_cast<int>(with_quantize_common(desc()->kq_zero_points));
    conf.kq_zp_mask = kq_zp_mask;

    int vs_zp_mask = (static_cast<int>(with_value_zp()) << 1)
            | static_cast<int>(with_quantize_common(desc()->vs_zero_points));
    conf.vs_zp_mask = vs_zp_mask;

    using namespace data_type;
    auto elems_per_byte = [](data_type_t dt) {
        switch (dt) {
            case u4:
            case s4: return 2;
            default: return 1;
        }
    };

    conf.key_elements_per_byte = elems_per_byte(key_mdw.data_type());
    conf.key_zp_elements_per_byte = elems_per_byte(key_zp_dt());
    conf.val_elements_per_byte = elems_per_byte(val_mdw.data_type());
    conf.val_zp_elements_per_byte = elems_per_byte(value_zp_dt());

    conf.key_group_size = 1;
    conf.val_group_size = 1;
    if (with_key_scales() || with_key_zp())
        conf.key_group_size = key_group_size();
    if (with_value_scales() || with_value_zp())
        conf.val_group_size = value_group_size();

    /* Set up microkernel strategy */
    const fwd_config_t config = {conf.ukernel_config.unroll_m_kq,
            conf.ukernel_config.unroll_n_kq, conf.ukernel_config.unroll_m_vs,
            conf.ukernel_config.unroll_n_vs, conf.ukernel_config.wg_m_kq,
            conf.ukernel_config.wg_n_kq, conf.ukernel_config.wg_m_vs,
            conf.ukernel_config.wg_n_vs};

    const int kq_wg_tile_m = config.wg_m_kq * config.unroll_m_kq;
    const int kq_wg_tile_n = config.wg_n_kq * config.unroll_n_kq;
    const int vs_wg_tile_m = config.wg_m_vs * config.unroll_m_vs;
    int tile_k = kq_wg_tile_m;
    int tile_v = vs_wg_tile_m;

    bool d_full = conf.d_full;
    bool v_full = (desc()->head_size() == tile_v);

    auto Q = desc()->queries();
    const dim_t Q_per_kv_group = (Q == 1 ? Q * conf.kv_group_size : Q);
    bool q_full = ((Q_per_kv_group % kq_wg_tile_n) != 0);
    conf.remainder_q = d_full && q_full;

    conf.block_q = conf.block_a = conf.block_2d_a = false;
    if (d_full) {
        conf.block_q = (ldq % 4 == 0);
        conf.block_a = (lda % 4 == 0 && v_full);
    } else if (arch() >= compute::gpu_arch_t::xe_hpc
            && config.unroll_m_vs < 64) {
        auto vbytes = desc()->values() * val_mdw.data_type_size();
        if (lda % 16 == 0 && vbytes % 4 == 0) conf.block_2d_a = true;
    }

    if (arch() >= compute::gpu_arch_t::xe_hpc) {
        conf.prefetch_mask = true;
        conf.prefetch_k0 = true;
        conf.prefetch_k = true;
        conf.prefetch_v = true;
        conf.prefetch_d_max = nstl::min(d_max(), 64);
        bool no_rem = d_full && v_full && (desc()->keys() % tile_k == 0);
        conf.prefetch_remainder = !no_rem;
    } else {
        conf.prefetch_mask = conf.prefetch_k0 = conf.prefetch_k
                = conf.prefetch_v = conf.prefetch_remainder = false;
        conf.prefetch_d_max = 0;
    }

    conf.q_arrive_await_barrier = (Q > 1);
    conf.softmax_inf_as_zero
            = (desc()->softmax_alg == alg_kind::softmax_accurate_inf_as_zero);
    conf.kq_f16_accumulate = (kq_acc_dt() == data_type::f16);
    conf.vs_f16_accumulate = (vs_acc_dt() == data_type::f16);

    bool is_training = desc()->prop_kind == prop_kind::forward_training;
    conf.is_training = is_training;
    if (is_training) { init_default_ws(); }

    return status::success;
}

status_t micro_bwd_t::pd_t::init_conf(impl::engine_t *engine) {
    init_conf_common(conf, this);

    conf.require_stateless_addressing = has_large_buffers();
    conf.with_dS = with_dS();

    const memory_desc_wrapper key_mdw(desc()->key_md());
    const memory_desc_wrapper val_mdw(desc()->val_md());

    auto ldk
            = gemm_desc_t::get_ld(*desc()->key_md()) * key_mdw.data_type_size();
    auto ldv
            = gemm_desc_t::get_ld(*desc()->val_md()) * val_mdw.data_type_size();

    /* Set up microkernel strategy */
    const bwd_config_t config = {conf.ukernel_config.unroll_m_BcBr,
            conf.ukernel_config.unroll_n_BcBr, conf.ukernel_config.unroll_m_DBc,
            conf.ukernel_config.unroll_n_DBc, conf.ukernel_config.unroll_m_DBr,
            conf.ukernel_config.unroll_n_DBr, conf.ukernel_config.wg_m_BcBr,
            conf.ukernel_config.wg_n_BcBr, conf.ukernel_config.wg_m_DBc,
            conf.ukernel_config.wg_n_DBc, conf.ukernel_config.wg_m_DBr,
            conf.ukernel_config.wg_n_DBr};

    const int kq_wg_tile_m = config.wg_m_BcBr * config.unroll_m_BcBr;
    const int tile_k = kq_wg_tile_m;

    const int tile_dv = config.wg_n_DBc * config.unroll_n_DBc;

    bool d_full = conf.d_full;
    bool dv_full = (desc()->head_size() == tile_dv);

    conf.block_k = conf.block_dK = conf.block_dV = false;
    if (d_full) {
        bool can_block_load_k
                = (ldk % 4 == 0) && (desc()->keys() % tile_k == 0);
        conf.block_k = can_block_load_k;
        if (conf.transpose_k) {
            // tile_store_dK_t uses lddk = max(DK_S2, DK_S3)
            const memory_desc_wrapper dk_mdw(desc()->diff_key_md());
            auto lddk_bytes = std::max(dk_mdw.strides()[2], dk_mdw.strides()[3])
                    * dk_mdw.data_type_size();
            conf.block_dK = (lddk_bytes % 4 == 0);
        } else {
            conf.block_dK = can_block_load_k;
        }
        conf.block_dV = (ldv % 4 == 0) && (dv_full);
    }

    return status::success;
}

status_t micro_bwd_t::pd_t::init_scratchpad(impl::engine_t *engine) {
    auto scratchpad = scratchpad_registry().registrar();
    auto gpu_align
            = utils::downcast<gpu::engine_t *>(engine)->get_buffer_alignment();
    size_t wspace_size = memory_desc_wrapper(desc()->diff_qry_md()).nelems();
    // f32 can directly atomic add to output
    // others need intermediate scratchpad before conversion
    if (conf.data_t != data_type::f32) {
        scratchpad.book(memory_tracking::names::key_sdpa_dQ_reduction,
                wspace_size, sizeof(float), gpu_align);
    }

    // for GQA cases multiple Q heads atomic add into shared dK/dV
    const bool needs_intermediate_dKV
            = (conf.kv_group_size > 1 && conf.data_t != data_type::f32);
    if (needs_intermediate_dKV) {
        size_t dK_size = memory_desc_wrapper(desc()->diff_key_md()).nelems();
        scratchpad.book(memory_tracking::names::key_sdpa_dK_reduction, dK_size,
                sizeof(float), gpu_align);

        size_t dV_size = memory_desc_wrapper(desc()->diff_val_md()).nelems();
        scratchpad.book(memory_tracking::names::key_sdpa_dV_reduction, dV_size,
                sizeof(float), gpu_align);
    }

    // space for D_i preprocess result
    size_t Di_size
            = desc()->batch() * desc()->num_q_heads() * desc()->queries();
    scratchpad.book(memory_tracking::names::key_sdpa_Di, Di_size, sizeof(float),
            gpu_align);

    // buffer for stride/offset parameters passed to backwards kernel
    // through global pointer to reduce kernel arguments size
    scratchpad.book(memory_tracking::names::key_sdpa_bwd_strides, 32,
            sizeof(int64_t), gpu_align);

    return status::success;
}

status_t micro_fwd_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace micro;
    kernel_ctx.require_stateless_addressing(require_stateless_addressing);

    kernel_ctx.define_int("NDIMS", ndims);
    kernel_ctx.set_data_type(data_t);

    def_data_type(kernel_ctx, key_data_t, "KEY");
    def_data_type(kernel_ctx, qry_data_t, "QRY");
    def_data_type(kernel_ctx, val_data_t, "VAL");
    def_data_type(kernel_ctx, dst_data_t, "DST");

    if (with_attn_mask) { def_data_type(kernel_ctx, msk_data_t, "MSK"); }

    def_data_type(kernel_ctx, key_scales_data_t, "KEY_ATTR_SCALES");
    def_data_type(kernel_ctx, value_scales_data_t, "VAL_ATTR_SCALES");

    def_data_type(kernel_ctx, key_zp_data_t, "KEY_ATTR_ZP");
    def_data_type(kernel_ctx, value_zp_data_t, "VAL_ATTR_ZP");

    kernel_ctx.define_int("KV_GROUP_SIZE", kv_group_size);

    kernel_ctx.define_int("Q_ALIGN", q_align);
    kernel_ctx.define_int("K_ALIGN", k_align);
    kernel_ctx.define_int("V_ALIGN", v_align);
    kernel_ctx.define_int("A_ALIGN", a_align);

    kernel_ctx.define_int("TRANSPOSE_K", transpose_k);

    kernel_ctx.define_int("KEY_SCALES", kq_scale_mask);
    kernel_ctx.define_int("VAL_SCALES", vs_scale_mask);
    kernel_ctx.define_int("KEY_ZERO_POINTS", kq_zp_mask);
    kernel_ctx.define_int("VAL_ZERO_POINTS", vs_zp_mask);

    kernel_ctx.define_int("KEY_ELEMENTS_PER_BYTE", key_elements_per_byte);
    kernel_ctx.define_int("KEY_ZP_ELEMENTS_PER_BYTE", key_zp_elements_per_byte);
    kernel_ctx.define_int("VAL_ELEMENTS_PER_BYTE", val_elements_per_byte);
    kernel_ctx.define_int("VAL_ZP_ELEMENTS_PER_BYTE", val_zp_elements_per_byte);

    kernel_ctx.define_int("KEY_GROUP_SIZE", key_group_size);
    kernel_ctx.define_int("VAL_GROUP_SIZE", val_group_size);

    def_data_type(kernel_ctx, scale_data_t, "SCALE", !with_host_scale);
    kernel_ctx.define_int("INVERT_SCALE", invert_scale);
    kernel_ctx.define_int("WITH_ATTN_SCALE", with_attn_scale);
    kernel_ctx.define_int("WITH_HOST_SCALE", with_host_scale);
    kernel_ctx.define_int("ATTN_MASK_UNDEF", attn_mask_undef);
    kernel_ctx.define_int("ATTN_MASK_BUFFER", attn_mask_buffer);
    kernel_ctx.define_int("ATTN_MASK_TOP_LEFT", attn_mask_top_left);
    kernel_ctx.define_int("ATTN_MASK_BOTTOM_RIGHT", attn_mask_bottom_right);

    kernel_ctx.define_int("WITH_ATTN_MASK", with_attn_mask);
    kernel_ctx.define_int("BROADCAST_MASK_Q", broadcast_mask_q);
    kernel_ctx.define_int("WITH_CAUSAL_MASK", with_causal_mask);

    kernel_ctx.define_int("SUBGROUP_SIZE", subgroup_size);
    kernel_ctx.define_int("D_MAX", d_max);

    kernel_ctx.define_int("BLOCK_Q", block_q);
    kernel_ctx.define_int("BLOCK_A", block_a);
    kernel_ctx.define_int("BLOCK_2D_A", block_2d_a);

    kernel_ctx.define_int("PREFETCH_MASK", prefetch_mask);
    kernel_ctx.define_int("PREFETCH_K0", prefetch_k0);
    kernel_ctx.define_int("PREFETCH_K", prefetch_k);
    kernel_ctx.define_int("PREFETCH_V", prefetch_v);
    kernel_ctx.define_int("PREFETCH_REMAINDER", prefetch_remainder);
    kernel_ctx.define_int("PREFETCH_D_MAX", prefetch_d_max);
    kernel_ctx.define_int("REMAINDER_Q", remainder_q);

    kernel_ctx.define_int("Q_ARRIVE_AWAIT_BARRIER", q_arrive_await_barrier);
    kernel_ctx.define_int("SOFTMAX_INF_AS_ZERO", softmax_inf_as_zero);
    kernel_ctx.define_int("USE_SYSTOLIC_UKERNEL", use_systolic_ukernel);
    kernel_ctx.define_int("KQ_F16_ACC", kq_f16_accumulate);
    kernel_ctx.define_int("VS_F16_ACC", vs_f16_accumulate);
    kernel_ctx.define_int("IS_TRAINING", is_training);
    kernel_ctx.define_int("WITH_DROPOUT", dropout);
    kernel_ctx.define_int("DROPOUT_HOST_SCALARS", dropout_host_scalars);
    kernel_ctx.define_int("DROPOUT_OUTPUT_MASK", dropout_output_mask);

    micro::HWInformation hw_info;
    gemmstone::GEMMProblem problem_kq, problem_vs;
    micro::GEMMOptions opts_kq, opts_vs;
    gemmstone::SizeParams sizes_kq, sizes_vs;

    deserialize_config_to_gemmstone(hw_info, problem_kq, problem_vs, opts_kq,
            opts_vs, sizes_kq, sizes_vs, ukernel_config);

    micro::Package gemm_kq, gemm_vs;

    /* Set up microkernel strategy */
    const fwd_config_t config
            = {ukernel_config.unroll_m_kq, ukernel_config.unroll_n_kq,
                    ukernel_config.unroll_m_vs, ukernel_config.unroll_n_vs,
                    ukernel_config.wg_m_kq, ukernel_config.wg_n_kq,
                    ukernel_config.wg_m_vs, ukernel_config.wg_n_vs};

    std::vector<StrategyRequirement> reqs_kq;
    reqs_kq.push_back(StrategyRequirement::UnrollM == config.unroll_m_kq);
    reqs_kq.push_back(StrategyRequirement::UnrollN == config.unroll_n_kq);
    reqs_kq.push_back(StrategyRequirement::WGM == config.wg_m_kq);
    reqs_kq.push_back(StrategyRequirement::WGN == config.wg_n_kq);

    std::vector<StrategyRequirement> reqs_vs;
    reqs_vs.push_back(StrategyRequirement::UnrollM == config.unroll_m_vs);
    reqs_vs.push_back(StrategyRequirement::UnrollN == config.unroll_n_vs);
    reqs_vs.push_back(StrategyRequirement::WGM == config.wg_m_vs);
    reqs_vs.push_back(StrategyRequirement::WGN == config.wg_n_vs);

    /* Ask microkernel provider for microkernel */
    auto kq_strat_override = [&](gemmstone::GEMMStrategy &strat) {
        std::string newStrat;
        newStrat = gpu_utils::dev_getenv("SDPA_KQ_USTRATEGY", newStrat);
        if (!newStrat.empty()) {
            // Example: 16 16 aT32 aM32 aB wg 2x4 sys
            auto product = ngen::npack::decodeHWIPVersion(hw_info.gmdid);
            auto hw = getCore(product.family);
            auto stepping = hw_info.gmdid & 0xFF;
            strat = gemmstone::GEMMStrategy(hw, stepping);
            std::stringstream ss(newStrat);
            ss >> strat.unroll[0];
            ss >> strat.unroll[1];
            std::string strategyString;
            std::getline(ss >> std::ws, strategyString);
            parseStrategy(strategyString, hw, problem_kq, strat);
            adjustStrategy(hw, problem_kq, strat);
        }
    };
    try {
        gemm_kq = micro::selectGEMM(opts_kq, hw_info, sizes_kq, problem_kq,
                reqs_kq, kq_strat_override);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_kq microkernel generation failure with message: %s",
                ex.what());
    }

    /* Ask microkernel provider for microkernel */
    auto vs_strat_override = [&](gemmstone::GEMMStrategy &strat) {
        std::string newStrat;
        newStrat = gpu_utils::dev_getenv("SDPA_VS_USTRATEGY", newStrat);
        if (!newStrat.empty()) {
            // Example: 16 16 aT32 aM32 aB wg 2x4 sys
            auto product = ngen::npack::decodeHWIPVersion(hw_info.gmdid);
            auto hw = getCore(product.family);
            auto stepping = hw_info.gmdid & 0xFF;
            strat = gemmstone::GEMMStrategy(hw, stepping);
            std::stringstream ss(newStrat);
            ss >> strat.unroll[0];
            ss >> strat.unroll[1];
            std::string strategyString;
            std::getline(ss >> std::ws, strategyString);
            parseStrategy(strategyString, hw, problem_vs, strat);
            adjustStrategy(hw, problem_vs, strat);
        }
    };
    try {
        if (use_systolic_ukernel) {
            auto adjust_vs = [&](GEMMStrategy &strategy) {
                /* Enable dpasw */
                strategy.dpasw |= strategy.fused;
                vs_strat_override(strategy);
            };
            gemm_vs = micro::selectGEMM(
                    opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs, adjust_vs);
        } else {
            gemm_vs = micro::selectGEMM(opts_vs, hw_info, sizes_vs, problem_vs,
                    reqs_vs, vs_strat_override);
        }
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_vs microkernel generation failure with message: %s",
                ex.what());
    }

    VDEBUGINFO(4, primitive, sdpa, "kq_gemm: %s, vs_gemm: %s,",
            problem_kq.toString().c_str(), problem_vs.toString().c_str());

    /* Generate microkernel shims */
    micro::ShimOptions shimOptions;
    shimOptions.subgroupSize = subgroup_size;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "kq";

    kernel_ctx.add_custom_header("gemm_kq.h",
            micro::generateShim(gemm_kq, HostLanguage::OpenCL_C, shimOptions));

    shimOptions.microkernelID++;
    shimOptions.decorator = "vs";

    kernel_ctx.add_custom_header("gemm_vs.h",
            micro::generateShim(gemm_vs, HostLanguage::OpenCL_C, shimOptions));

    if (gemm_kq.grfMin > 128 || gemm_vs.grfMin > 128)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    return status::success;
}

status_t micro_bwd_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace micro;
    kernel_ctx.require_stateless_addressing(require_stateless_addressing);

    kernel_ctx.define_int("NDIMS", ndims);
    kernel_ctx.set_data_type(data_t);

    def_data_type(kernel_ctx, key_data_t, "KEY");
    def_data_type(kernel_ctx, qry_data_t, "QRY");
    def_data_type(kernel_ctx, val_data_t, "VAL");
    def_data_type(kernel_ctx, dst_data_t, "DST");

    if (with_attn_mask) { def_data_type(kernel_ctx, msk_data_t, "MSK"); }

    kernel_ctx.define_int("KV_GROUP_SIZE", kv_group_size);

    kernel_ctx.define_int("Q_ALIGN", q_align);
    kernel_ctx.define_int("K_ALIGN", k_align);
    kernel_ctx.define_int("V_ALIGN", v_align);
    kernel_ctx.define_int("A_ALIGN", a_align);

    kernel_ctx.define_int("TRANSPOSE_K", transpose_k);

    def_data_type(kernel_ctx, scale_data_t, "SCALE", !with_host_scale);
    kernel_ctx.define_int("INVERT_SCALE", invert_scale);
    kernel_ctx.define_int("WITH_ATTN_SCALE", with_attn_scale);
    kernel_ctx.define_int("WITH_HOST_SCALE", with_host_scale);
    kernel_ctx.define_int("ATTN_MASK_UNDEF", attn_mask_undef);
    kernel_ctx.define_int("ATTN_MASK_BUFFER", attn_mask_buffer);
    kernel_ctx.define_int("ATTN_MASK_TOP_LEFT", attn_mask_top_left);
    kernel_ctx.define_int("ATTN_MASK_BOTTOM_RIGHT", attn_mask_bottom_right);

    kernel_ctx.define_int("WITH_ATTN_MASK", with_attn_mask);
    kernel_ctx.define_int("BROADCAST_MASK_Q", broadcast_mask_q);
    kernel_ctx.define_int("WITH_CAUSAL_MASK", with_causal_mask);
    kernel_ctx.define_int("WITH_DS", with_dS);

    kernel_ctx.define_int("SUBGROUP_SIZE", subgroup_size);
    kernel_ctx.define_int("D_MAX", d_max);

    kernel_ctx.define_int("BLOCK_K", block_k);
    kernel_ctx.define_int("BLOCK_DK", block_dK);
    kernel_ctx.define_int("BLOCK_DV", block_dV);

    kernel_ctx.define_int("USE_SYSTOLIC_UKERNEL", use_systolic_ukernel);
    kernel_ctx.define_int("WITH_DROPOUT", dropout);
    kernel_ctx.define_int("DROPOUT_HOST_SCALARS", dropout_host_scalars);
    kernel_ctx.define_int("DROPOUT_OUTPUT_MASK", dropout_output_mask);

    micro::HWInformation hw_info;
    gemmstone::GEMMProblem problem_kq, problem_vs;
    micro::GEMMOptions opts_kq, opts_vs;
    gemmstone::SizeParams sizes_kq, sizes_vs;

    gemmstone::GEMMProblem problem_vtdA, problem_ktq, problem_qdSt;
    micro::GEMMOptions opts_vtdA, opts_ktq, opts_qdSt;
    gemmstone::SizeParams sizes_vtdA, sizes_ktq, sizes_qdSt;

    deserialize_config_to_gemmstone(hw_info, problem_kq, problem_vs,
            problem_vtdA, problem_ktq, problem_qdSt, opts_kq, opts_vs,
            opts_vtdA, opts_ktq, opts_qdSt, sizes_kq, sizes_vs, sizes_vtdA,
            sizes_ktq, sizes_qdSt, ukernel_config);

    micro::Package gemm_kq, gemm_vs, gemm_vtdA, gemm_ktq, gemm_qdSt;

    /* Set up microkernel strategy */
    const bwd_config_t config
            = {ukernel_config.unroll_m_BcBr, ukernel_config.unroll_n_BcBr,
                    ukernel_config.unroll_m_DBc, ukernel_config.unroll_n_DBc,
                    ukernel_config.unroll_m_DBr, ukernel_config.unroll_n_DBr,
                    ukernel_config.wg_m_BcBr, ukernel_config.wg_n_BcBr,
                    ukernel_config.wg_m_DBc, ukernel_config.wg_n_DBc,
                    ukernel_config.wg_m_DBr, ukernel_config.wg_n_DBr};

    std::vector<StrategyRequirement> reqs_kq;
    reqs_kq.push_back(StrategyRequirement::UnrollM == config.unroll_m_BcBr);
    reqs_kq.push_back(StrategyRequirement::UnrollN == config.unroll_n_BcBr);
    reqs_kq.push_back(StrategyRequirement::WGM == config.wg_m_BcBr);
    reqs_kq.push_back(StrategyRequirement::WGN == config.wg_n_BcBr);

    std::vector<StrategyRequirement> reqs_vs;
    reqs_vs.push_back(StrategyRequirement::UnrollM == config.unroll_m_DBc);
    reqs_vs.push_back(StrategyRequirement::UnrollN == config.unroll_n_DBc);
    reqs_vs.push_back(StrategyRequirement::WGM == config.wg_m_DBc);
    reqs_vs.push_back(StrategyRequirement::WGN == config.wg_n_DBc);

    std::vector<StrategyRequirement> reqs_vtdA;
    reqs_vtdA.push_back(StrategyRequirement::UnrollM == config.unroll_m_BcBr);
    reqs_vtdA.push_back(StrategyRequirement::UnrollN == config.unroll_n_BcBr);
    reqs_vtdA.push_back(StrategyRequirement::WGM == config.wg_m_BcBr);
    reqs_vtdA.push_back(StrategyRequirement::WGN == config.wg_n_BcBr);

    std::vector<StrategyRequirement> reqs_ktq;
    reqs_ktq.push_back(StrategyRequirement::UnrollM == config.unroll_m_DBr);
    reqs_ktq.push_back(StrategyRequirement::UnrollN == config.unroll_n_DBr);
    reqs_ktq.push_back(StrategyRequirement::WGM == config.wg_m_DBr);
    reqs_ktq.push_back(StrategyRequirement::WGN == config.wg_n_DBr);

    std::vector<StrategyRequirement> reqs_qdSt;
    reqs_qdSt.push_back(StrategyRequirement::UnrollM == config.unroll_n_DBc);
    reqs_qdSt.push_back(StrategyRequirement::UnrollN == config.unroll_m_DBc);
    reqs_qdSt.push_back(StrategyRequirement::WGM == config.wg_n_DBc);
    reqs_qdSt.push_back(StrategyRequirement::WGN == config.wg_m_DBc);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq = micro::selectGEMM(
                opts_kq, hw_info, sizes_kq, problem_kq, reqs_kq);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_kq microkernel generation failure with message: %s",
                ex.what());
    }

    try {
        if (use_systolic_ukernel) {
            auto adjust_vs = [](GEMMStrategy &strategy) {
                /* Enable dpasw */
                strategy.dpasw |= strategy.fused;
            };
            gemm_vs = micro::selectGEMM(
                    opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs, adjust_vs);
        } else {
            gemm_vs = micro::selectGEMM(
                    opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs);
        }
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_vs microkernel generation failure with message: %s",
                ex.what());
    }

    VDEBUGINFO(4, primitive, sdpa,
            "kq_gemm: %s, vs_gemm: %s, vtdA_gemm: %s, ktq_gemm: %s, qdSt: %s\n",
            problem_kq.toString().c_str(), problem_vs.toString().c_str(),
            problem_vtdA.toString().c_str(), problem_ktq.toString().c_str(),
            problem_qdSt.toString().c_str());

    /* Generate microkernel shims */
    micro::ShimOptions shimOptions;
    shimOptions.subgroupSize = subgroup_size;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "kq";

    std::string gemm_kq_header
            = micro::generateShim(gemm_kq, HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_kq.h", std::move(gemm_kq_header));

    shimOptions.microkernelID++;
    shimOptions.decorator = "vs";

    std::string gemm_vs_header
            = micro::generateShim(gemm_vs, HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_vs.h", std::move(gemm_vs_header));

    try {
        gemm_vtdA = micro::selectGEMM(
                opts_vtdA, hw_info, sizes_vtdA, problem_vtdA, reqs_vtdA);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_vtdA microkernel generation failure with message: %s",
                ex.what());
    }

    shimOptions.microkernelID++;
    shimOptions.decorator = "vtdA";

    std::string gemm_vtdA_header = micro::generateShim(
            gemm_vtdA, HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_vtdA.h", std::move(gemm_vtdA_header));

    try {
        gemm_ktq = micro::selectGEMM(
                opts_ktq, hw_info, sizes_ktq, problem_ktq, reqs_ktq);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_ktq microkernel generation failure with message: %s",
                ex.what());
    }

    shimOptions.microkernelID++;
    shimOptions.decorator = "ktq";

    std::string gemm_ktq_header = micro::generateShim(
            gemm_ktq, HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_ktq.h", std::move(gemm_ktq_header));

    try {
        gemm_qdSt = micro::selectGEMM(
                opts_qdSt, hw_info, sizes_qdSt, problem_qdSt, reqs_qdSt);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_qdSt microkernel generation failure with message: %s",
                ex.what());
    }

    shimOptions.microkernelID++;
    shimOptions.decorator = "qdSt";

    std::string gemm_qdSt_header = micro::generateShim(
            gemm_qdSt, HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_qdSt.h", std::move(gemm_qdSt_header));

    if (gemm_kq.grfMin > 128 || gemm_vs.grfMin > 128 || gemm_vtdA.grfMin > 128
            || gemm_ktq.grfMin > 128 || gemm_qdSt.grfMin > 128)
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    return status::success;
}

using fwd_offset_t = dim_t[4][MAX_NDIMS];

template <typename T>
static void append_key_offs(
        compute::kernel_arg_list_t &arg_list, const fwd_offset_t &offs) {
    arg_list.append(static_cast<T>(offs[1][0])); // KEY_S0
    arg_list.append(static_cast<T>(offs[1][1])); // KEY_S1
    arg_list.append(static_cast<T>(offs[1][2])); // KEY_S2
    arg_list.append(static_cast<T>(offs[1][3])); // KEY_S3
    arg_list.append(static_cast<T>(offs[3][3])); // KEY_D3
}

template <typename T>
static void append_qry_offs(
        compute::kernel_arg_list_t &arg_list, const fwd_offset_t &offs) {
    arg_list.append(static_cast<T>(offs[1][0])); // QRY_S0
    arg_list.append(static_cast<T>(offs[1][1])); // QRY_S1
    arg_list.append(static_cast<T>(offs[1][2])); // QRY_S2
}

template <typename T>
static void append_val_offs(
        compute::kernel_arg_list_t &arg_list, const fwd_offset_t &offs) {
    arg_list.append(static_cast<T>(offs[1][0])); // VAL_S0
    arg_list.append(static_cast<T>(offs[1][1])); // VAL_S1
    arg_list.append(static_cast<T>(offs[1][2])); // VAL_S2
}

template <typename T>
static void append_dst_offs(
        compute::kernel_arg_list_t &arg_list, const fwd_offset_t &offs) {
    arg_list.append(static_cast<T>(offs[1][0])); // DST_S0
    arg_list.append(static_cast<T>(offs[1][1])); // DST_S1
    arg_list.append(static_cast<T>(offs[1][2])); // DST_S2
    arg_list.append(static_cast<T>(offs[3][1])); // DST_D1
}

template <typename T>
static void append_msk_offs(
        compute::kernel_arg_list_t &arg_list, const fwd_offset_t &offs) {
    arg_list.append(static_cast<T>(offs[1][0])); // MSK_S0
    arg_list.append(static_cast<T>(offs[1][1])); // MSK_S1
    arg_list.append(static_cast<T>(offs[1][2])); // MSK_S2
    arg_list.append(static_cast<T>(offs[3][0])); // MSK_D0
    arg_list.append(static_cast<T>(offs[3][1])); // MSK_D1
}

template <typename conf_t>
static status_t append_dropout_args(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, const conf_t &conf) {
    if (!conf.dropout) return status::success;

    const auto &dropout_p = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
    const auto &dropout_seed = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_SEED);
    const auto &dropout_offset = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_OFFSET);
    arg_list.append(CTX_OUT_STORAGE(DNNL_ARG_ATTR_DROPOUT_MASK));
    arg_list.append(static_cast<int>(conf.dropout_offset));
    if (conf.dropout_host_scalars) {
        int64_t scalar_seed = 0;
        int64_t scalar_offset = 0;
        float scalar_prob = 0.f;
        const host_scalar_memory_storage_t *seed_storage
                = utils::downcast<const host_scalar_memory_storage_t *>(
                        &dropout_seed);
        CHECK(seed_storage->get_scalar_value(
                &scalar_seed, sizeof(scalar_seed)));
        if (conf.dropout_offset) {
            const host_scalar_memory_storage_t *offset_storage
                    = utils::downcast<const host_scalar_memory_storage_t *>(
                            &dropout_offset);
            CHECK(offset_storage->get_scalar_value(
                    &scalar_offset, sizeof(scalar_offset)));
        }
        const host_scalar_memory_storage_t *prob_storage
                = utils::downcast<const host_scalar_memory_storage_t *>(
                        &dropout_p);
        CHECK(prob_storage->get_scalar_value(
                &scalar_prob, sizeof(scalar_prob)));
        arg_list.append(scalar_seed);
        arg_list.append(scalar_offset);
        arg_list.append(scalar_prob);
    } else {
        arg_list.append(dropout_seed);
        arg_list.append(dropout_offset);
        arg_list.append(dropout_p);
    }

    return status::success;
}

status_t micro_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    const auto &conf = pd()->conf;

    const auto &qry = CTX_IN_STORAGE(DNNL_ARG_QUERIES);
    const auto &key = CTX_IN_STORAGE(DNNL_ARG_KEYS);
    const auto &val = CTX_IN_STORAGE(DNNL_ARG_VALUES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);
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

    const int kv_group_size = pd()->conf.kv_group_size;
    const dim_t Q = pd()->desc()->queries();
    const dim_t K = pd()->desc()->keys();
    const dim_t D = pd()->desc()->head_size();
    const dim_t Q_per_kv_group = (Q == 1 ? Q * kv_group_size : Q);

    const fwd_config_t config = {conf.ukernel_config.unroll_m_kq,
            conf.ukernel_config.unroll_n_kq, conf.ukernel_config.unroll_m_vs,
            conf.ukernel_config.unroll_n_vs, conf.ukernel_config.wg_m_kq,
            conf.ukernel_config.wg_n_kq, conf.ukernel_config.wg_m_vs,
            conf.ukernel_config.wg_n_vs};
    const int kq_wg_tile_m = config.wg_m_kq * config.unroll_m_kq;
    const int kq_wg_tile_n = config.wg_n_kq * config.unroll_n_kq;
    auto wg_tile_q = kq_wg_tile_n;
    auto sg_per_wg = config.wg_m_kq * config.wg_n_kq;

    const memory_desc_wrapper qry_mdw(pd()->desc()->qry_md());
    const memory_desc_wrapper key_mdw(pd()->desc()->key_md());
    const memory_desc_wrapper val_mdw(pd()->desc()->val_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper msk_mdw(pd()->desc()->attn_mask_md());

    fwd_offset_t key_off, qry_off, val_off, dst_off, msk_off;

    set_offsets(key_mdw, key_off);
    set_offsets(qry_mdw, qry_off);
    set_offsets(val_mdw, val_off);
    set_offsets(dst_mdw, dst_off);
    set_offsets(msk_mdw, msk_off);

    const memory_desc_wrapper scale_mdw(pd()->desc()->scale_md());
    float scalar_scale = 1.f;
    float inv_scalar_scale = 1.f;
    if (pd()->with_host_scale()) {
        auto scalar_storage = utils::downcast<
                const dnnl::impl::host_scalar_memory_storage_t *>(&scale);
        auto status = scalar_storage->get_scalar_value(
                &scalar_scale, scale_mdw.data_type_size());
        assert(status == status::success);
        if (status != status::success) return status;
        scalar_scale = dnnl::impl::cpu::io::load_float_value(
                pd()->desc()->scale_md()->data_type, &scalar_scale, 0);
        inv_scalar_scale = 1. / scalar_scale;
    }

    int mask_type = static_cast<int>(pd()->desc()->mask_type);
    compute::kernel_arg_list_t arg_list;
    arg_list.append(key);
    arg_list.append(qry);
    arg_list.append(val);
    arg_list.append(ws);
    arg_list.append(dst);
    if (pd()->with_host_scale()) {
        arg_list.append(scalar_scale);
        arg_list.append(inv_scalar_scale);
    } else {
        arg_list.append(scale);
    }
    arg_list.append((int)D);
    arg_list.append((int)K);
    arg_list.append((int)Q);
    arg_list.append(key_scales);
    arg_list.append(key_zp);
    arg_list.append(value_scales);
    arg_list.append(value_zp);
    arg_list.append(mask_type);
    if (pd()->with_attn_mask()) arg_list.append(attn_mask);

    append_key_offs<int64_t>(arg_list, key_off);
    append_qry_offs<int64_t>(arg_list, qry_off);
    append_val_offs<int64_t>(arg_list, val_off);
    append_dst_offs<int64_t>(arg_list, dst_off);
    if (pd()->with_attn_mask()) { append_msk_offs<int64_t>(arg_list, msk_off); }
    const int remainder_k = (K % kq_wg_tile_m) != 0;

    arg_list.append(remainder_k);

    CHECK(append_dropout_args(ctx, arg_list, pd()->conf));
    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    if (Q == 1) {
        // For second token Grouped Query Attention(GQA) cases, we batch the
        // kernel across the KV heads instead of the q heads. This allows us to
        // batch multiple queries into a single work group.
        gws[0] *= utils::div_up(Q_per_kv_group, wg_tile_q);
        gws[1] *= utils::div_up(pd()->dst_md()->dims[1], kv_group_size);
    } else {
        gws[0] *= utils::div_up(Q, wg_tile_q);
        gws[1] *= pd()->dst_md()->dims[1];
    }
    gws[2] *= pd()->desc()->batch();

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

status_t micro_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    const auto &qry = CTX_IN_STORAGE(DNNL_ARG_QUERIES);
    const auto &key = CTX_IN_STORAGE(DNNL_ARG_KEYS);
    const auto &val = CTX_IN_STORAGE(DNNL_ARG_VALUES);
    const auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);
    const auto &dst = CTX_IN_STORAGE(DNNL_ARG_DST);
    const auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_q = CTX_OUT_STORAGE(DNNL_ARG_DIFF_QUERIES);
    auto &diff_k = CTX_OUT_STORAGE(DNNL_ARG_DIFF_KEYS);
    auto &diff_v = CTX_OUT_STORAGE(DNNL_ARG_DIFF_VALUES);
    const auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    const auto &attn_mask = CTX_IN_STORAGE(DNNL_ARG_ATTN_MASK);
    auto Di_scratch = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_sdpa_Di);
    auto diff_q_scratch = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_sdpa_dQ_reduction);
    auto diff_k_scratch = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_sdpa_dK_reduction);
    auto diff_v_scratch = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_sdpa_dV_reduction);
    auto strides_scratch = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_sdpa_bwd_strides);

    const bool with_dS = pd()->with_dS();

    const int kv_group_size = pd()->conf.kv_group_size;
    const dim_t Q = pd()->desc()->queries();
    const dim_t K = pd()->desc()->keys();
    const dim_t D = pd()->desc()->head_size();

    const data_type_t data_t = pd()->dst_md()->data_type;
    const bool needs_intermediate_dQ = (data_t != data_type::f32);
    const bool needs_intermediate_dKV
            = (kv_group_size > 1 && data_t != data_type::f32);
    const bool needs_zero_dKV = (kv_group_size > 1);

    const auto &conf = pd()->conf;

    const bwd_config_t config = {conf.ukernel_config.unroll_m_BcBr,
            conf.ukernel_config.unroll_n_BcBr, conf.ukernel_config.unroll_m_DBc,
            conf.ukernel_config.unroll_n_DBc, conf.ukernel_config.unroll_m_DBr,
            conf.ukernel_config.unroll_n_DBr, conf.ukernel_config.wg_m_BcBr,
            conf.ukernel_config.wg_n_BcBr, conf.ukernel_config.wg_m_DBc,
            conf.ukernel_config.wg_n_DBc, conf.ukernel_config.wg_m_DBr,
            conf.ukernel_config.wg_n_DBr};

    auto wg_tile_k = config.unroll_m_BcBr * config.wg_m_BcBr;
    auto wg_tile_q = config.unroll_n_BcBr * config.wg_n_BcBr;

    auto sg_per_wg_BcBr = config.wg_m_BcBr * config.wg_n_BcBr;
    auto sg_per_wg_DBc = config.wg_m_DBc * config.wg_n_DBc;
    auto sg_per_wg_DBr = config.wg_m_DBr * config.wg_n_DBr;

    auto sg_per_wg
            = std::max(std::max(sg_per_wg_BcBr, sg_per_wg_DBc), sg_per_wg_DBr);

    const memory_desc_wrapper qry_mdw(pd()->desc()->qry_md());
    const memory_desc_wrapper key_mdw(pd()->desc()->key_md());
    const memory_desc_wrapper val_mdw(pd()->desc()->val_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper msk_mdw(pd()->desc()->attn_mask_md());
    const memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
    const memory_desc_wrapper diff_qry_mdw(pd()->desc()->diff_qry_md());
    const memory_desc_wrapper diff_key_mdw(pd()->desc()->diff_key_md());
    const memory_desc_wrapper diff_val_mdw(pd()->desc()->diff_val_md());
    using offset_t = decltype(offsets_t().src_off);

    offset_t qry_off, key_off, val_off, dst_off, msk_off, da_off;
    offset_t dq_off, dk_off, dv_off;

    set_offsets(qry_mdw, qry_off);
    set_offsets(key_mdw, key_off);
    set_offsets(val_mdw, val_off);
    set_offsets(dst_mdw, dst_off);
    set_offsets(msk_mdw, msk_off);
    set_offsets(diff_dst_mdw, da_off);

    // scratch buffers are allocated as flat B*H*S*D element buffers
    auto set_contiguous_offsets
            = [](offset_t &offs, dim_t B, dim_t H, dim_t S, dim_t D) {
        offs[3][0] = B;
        offs[3][1] = H;
        offs[3][2] = S;
        offs[3][3] = D;

        offs[1][0] = H * S * D;
        offs[1][1] = S * D;
        offs[1][2] = D;
        offs[1][3] = 1;
    };

    if (needs_intermediate_dQ) {
        set_contiguous_offsets(dq_off, pd()->desc()->batch(),
                pd()->desc()->num_q_heads(), Q, D);
    }
    if (needs_intermediate_dKV) {
        // match the tile orientation used by the kernel in dK write
        if (conf.transpose_k) {
            set_contiguous_offsets(dk_off, pd()->desc()->batch(),
                    pd()->desc()->num_kv_heads(), K, D);
        } else {
            set_contiguous_offsets(dk_off, pd()->desc()->batch(),
                    pd()->desc()->num_kv_heads(), D, K);
        }
        set_contiguous_offsets(dv_off, pd()->desc()->batch(),
                pd()->desc()->num_kv_heads(), K, D);
    }

    auto append_qry_offs
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
        arg_list.append((int64_t)offs[1][0]); // QRY_S0
        arg_list.append((int64_t)offs[1][1]); // QRY_S1
        arg_list.append((int64_t)offs[1][2]); // QRY_S2
    };
    auto append_dst_offs
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
        arg_list.append((int64_t)offs[1][0]); // DST_S0
        arg_list.append((int64_t)offs[1][1]); // DST_S1
        arg_list.append((int64_t)offs[1][2]); // DST_S2
        arg_list.append((int64_t)offs[3][1]); // DST_D1
    };
    auto append_da_offs
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
        arg_list.append((int64_t)offs[1][0]); // DA_S0
        arg_list.append((int64_t)offs[1][1]); // DA_S1
        arg_list.append((int64_t)offs[1][2]); // DA_S2
    };

    int mask_type = static_cast<int>(pd()->desc()->mask_type);

    const memory_desc_wrapper scale_mdw(pd()->desc()->scale_md());
    float scalar_scale = 1.f;
    float inv_scalar_scale = 1.f;
    if (pd()->with_host_scale()) {
        auto scalar_storage = utils::downcast<
                const dnnl::impl::host_scalar_memory_storage_t *>(&scale);
        auto status = scalar_storage->get_scalar_value(
                &scalar_scale, scale_mdw.data_type_size());
        assert(status == status::success);
        if (status != status::success) return status;
        scalar_scale = dnnl::impl::cpu::io::load_float_value(
                pd()->desc()->scale_md()->data_type, &scalar_scale, 0);
        inv_scalar_scale = 1. / scalar_scale;
    }

    /// preprocess kernel
    // will zero dQ, calculate Di
    compute::range_t lws = {(size_t)pd()->sg_size(), (size_t)sg_per_wg, 1};
    compute::range_t gws_preprocess = lws;

    gws_preprocess[0] *= utils::div_up(Q, wg_tile_q);
    gws_preprocess[1] *= pd()->dst_md()->dims[1];
    gws_preprocess[2] *= pd()->desc()->batch();

    auto nd_range_preprocess = compute::nd_range_t(gws_preprocess, lws);

    compute::kernel_arg_list_t preprocess_arg_list;
    preprocess_arg_list.append(*Di_scratch);
    preprocess_arg_list.append(dst);
    preprocess_arg_list.append(diff_dst);
    preprocess_arg_list.append((int)D);
    preprocess_arg_list.append((int)K);
    preprocess_arg_list.append((int)Q);

    append_qry_offs(preprocess_arg_list, qry_off);
    append_dst_offs(preprocess_arg_list, dst_off);
    append_da_offs(preprocess_arg_list, da_off);

    CHECK(parallel_for(
            ctx, nd_range_preprocess, preprocess_, preprocess_arg_list));

    auto *d = pd()->desc();
    // zero f32 intermediates before atomic adds in the main kernel
    // dQ always needs atomics, dK/dV only for GQA cases
    {
        auto compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());
        auto &fill_deps = compute_stream->ctx().get_deps();

        const dim_t batch = pd()->dst_md()->dims[0];
        const dim_t num_kv_heads = d->num_kv_heads();
        const dim_t num_q_heads = d->num_q_heads();

        auto zero_fill
                = [&](const memory_storage_t &buf, size_t bytes) -> status_t {
            return compute_stream->fill(buf, 0, bytes, fill_deps, fill_deps);
        };

        // always zero dQ
        auto &dQ_buf = needs_intermediate_dQ ? *diff_q_scratch : diff_q;
        const size_t dQ_bytes = needs_intermediate_dQ
                ? size_t(batch * num_q_heads * Q * D) * sizeof(float)
                : diff_qry_mdw.size();
        CHECK(zero_fill(dQ_buf, dQ_bytes));

        // zero dK/dV for GQA cases
        if (needs_zero_dKV) {
            auto &dK_buf = needs_intermediate_dKV ? *diff_k_scratch : diff_k;
            auto &dV_buf = needs_intermediate_dKV ? *diff_v_scratch : diff_v;
            const size_t scratch_kv_bytes
                    = size_t(batch * num_kv_heads * K * D) * sizeof(float);
            const size_t dK_bytes = needs_intermediate_dKV
                    ? scratch_kv_bytes
                    : diff_key_mdw.size();
            const size_t dV_bytes = needs_intermediate_dKV
                    ? scratch_kv_bytes
                    : diff_val_mdw.size();
            CHECK(zero_fill(dK_buf, dK_bytes));
            CHECK(zero_fill(dV_buf, dV_bytes));
        }
    }

    /// backwards pass kernel, calculates dK, dV, dQ(float)
    compute::kernel_arg_list_t arg_list;
    arg_list.append(key);
    arg_list.append(qry);
    arg_list.append(val);
    arg_list.append(ws);
    arg_list.append(*Di_scratch);
    arg_list.append(dst);
    arg_list.append(diff_dst);
    if (with_dS) arg_list.append(CTX_OUT_STORAGE(DNNL_ARG_DS));
    arg_list.append(needs_intermediate_dKV ? *diff_k_scratch : diff_k);
    arg_list.append(needs_intermediate_dQ ? *diff_q_scratch : diff_q);
    arg_list.append(needs_intermediate_dKV ? *diff_v_scratch : diff_v);
    if (pd()->with_host_scale()) {
        arg_list.append(scalar_scale);
        arg_list.append(inv_scalar_scale);
    } else {
        arg_list.append(scale);
    }
    CHECK(append_dropout_args(ctx, arg_list, pd()->conf));
    arg_list.append((int)D);
    arg_list.append((int)K);
    arg_list.append((int)Q);
    arg_list.append(mask_type);
    if (pd()->with_attn_mask()) arg_list.append(attn_mask);

    // pack all stride/offset parameters into a global buffer to reduce
    // kernel argument pressure (replaces 27-32 int64 args with 1 pointer).
    offset_t diff_key_off, diff_qry_off, diff_val_off;
    set_offsets(diff_key_mdw, diff_key_off);
    set_offsets(diff_qry_mdw, diff_qry_off);
    set_offsets(diff_val_mdw, diff_val_off);

    const auto &dk_off_ref = needs_intermediate_dKV ? dk_off : diff_key_off;
    const auto &dq_off_ref = needs_intermediate_dQ ? dq_off : diff_qry_off;
    const auto &dv_off_ref = needs_intermediate_dKV ? dv_off : diff_val_off;
    {
        void *mapped = nullptr;
        CHECK(strides_scratch->map_data(
                &mapped, ctx.stream(), 32 * sizeof(int64_t)));
        auto *buf = static_cast<int64_t *>(mapped);
        buf[0] = key_off[1][0]; // KEY_S0
        buf[1] = key_off[1][1]; // KEY_S1
        buf[2] = key_off[1][2]; // KEY_S2
        buf[3] = key_off[1][3]; // KEY_S3
        buf[4] = qry_off[1][0]; // QRY_S0
        buf[5] = qry_off[1][1]; // QRY_S1
        buf[6] = qry_off[1][2]; // QRY_S2
        buf[7] = val_off[1][0]; // VAL_S0
        buf[8] = val_off[1][1]; // VAL_S1
        buf[9] = val_off[1][2]; // VAL_S2
        buf[10] = dst_off[1][0]; // DST_S0
        buf[11] = dst_off[1][1]; // DST_S1
        buf[12] = dst_off[1][2]; // DST_S2
        buf[13] = dst_off[3][1]; // DST_D1
        buf[14] = da_off[1][0]; // DA_S0
        buf[15] = da_off[1][1]; // DA_S1
        buf[16] = da_off[1][2]; // DA_S2
        buf[17] = dk_off_ref[1][0]; // DK_S0
        buf[18] = dk_off_ref[1][1]; // DK_S1
        buf[19] = dk_off_ref[1][2]; // DK_S2
        buf[20] = dk_off_ref[1][3]; // DK_S3
        buf[21] = dq_off_ref[1][0]; // DQ_S0
        buf[22] = dq_off_ref[1][1]; // DQ_S1
        buf[23] = dq_off_ref[1][2]; // DQ_S2
        buf[24] = dv_off_ref[1][0]; // DV_S0
        buf[25] = dv_off_ref[1][1]; // DV_S1
        buf[26] = dv_off_ref[1][2]; // DV_S2
        if (pd()->with_attn_mask()) {
            buf[27] = msk_off[1][0]; // MSK_S0
            buf[28] = msk_off[1][1]; // MSK_S1
            buf[29] = msk_off[1][2]; // MSK_S2
            buf[30] = msk_off[3][0]; // MSK_D0
            buf[31] = msk_off[3][1]; // MSK_D1
        }
        CHECK(strides_scratch->unmap_data(mapped, ctx.stream()));
    }
    arg_list.append(*strides_scratch);

    const int remainder_k = (K % wg_tile_k) != 0;

    const bool d_full = (d->head_size() == pd()->d_max());
    const int remainder_q = d_full && ((Q % wg_tile_q) != 0);

    arg_list.append(remainder_k);
    arg_list.append(remainder_q);

    compute::range_t gws = lws;

    gws[0] *= utils::div_up(K, wg_tile_k);
    gws[1] *= pd()->dst_md()->dims[1];
    gws[2] *= pd()->desc()->batch();
    auto nd_range = compute::nd_range_t(gws, lws);

    CHECK(parallel_for(ctx, nd_range, kernel_, arg_list));

    auto append_offs
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
        // dims
        arg_list.append((int64_t)offs[3][0]);
        arg_list.append((int64_t)offs[3][1]);
        arg_list.append((int64_t)offs[3][2]);
        arg_list.append((int64_t)offs[3][3]);
        // strides
        arg_list.append((int64_t)offs[1][0]);
        arg_list.append((int64_t)offs[1][1]);
        arg_list.append((int64_t)offs[1][2]);
        arg_list.append((int64_t)offs[1][3]);
    };
    // postprocess kernels use int64 strides
    auto append_strides
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
        arg_list.append((int64_t)offs[1][0]);
        arg_list.append((int64_t)offs[1][1]);
        arg_list.append((int64_t)offs[1][2]);
        arg_list.append((int64_t)offs[1][3]);
    };
    /// postprocessing kernels
    // will cast dQ/dK/dV to lower precision outputs if needed
    if (needs_intermediate_dQ) {
        static constexpr size_t lws_pp = 256;
        compute::range_t lws_p = {(size_t)lws_pp, 1, 1};
        compute::range_t gws_p = lws_p;
        gws_p[0] *= utils::div_up(Q * D, lws_pp);
        gws_p[1] *= pd()->dst_md()->dims[1]; // Q heads
        gws_p[2] *= pd()->desc()->batch();

        compute::kernel_arg_list_t pp;
        pp.append(diff_q);
        pp.append(*diff_q_scratch);
        pp.append((int)(Q * D));
        append_strides(pp, dq_off);
        append_offs(pp, diff_qry_off);
        CHECK(parallel_for(
                ctx, compute::nd_range_t(gws_p, lws_p), postprocess_, pp));
    }

    if (needs_intermediate_dKV) {
        const dim_t num_kv_heads = d->num_kv_heads();
        static constexpr size_t lws_pp = 256;
        compute::range_t lws_p = {(size_t)lws_pp, 1, 1};

        // dK
        {
            compute::range_t gws_p = lws_p;
            gws_p[0] *= utils::div_up(K * D, lws_pp);
            gws_p[1] *= num_kv_heads; // KV heads
            gws_p[2] *= pd()->desc()->batch();

            compute::kernel_arg_list_t pp;
            pp.append(diff_k);
            pp.append(*diff_k_scratch);
            pp.append((int)(K * D));

            pp.append((int64_t)dk_off[1][0]);
            pp.append((int64_t)dk_off[1][1]);
            // match TRANSPOSE_K write order for dK
            if (conf.transpose_k) {
                pp.append((int64_t)dk_off[1][3]);
                pp.append((int64_t)dk_off[1][2]);
            } else {
                pp.append((int64_t)dk_off[1][2]);
                pp.append((int64_t)dk_off[1][3]);
            }
            append_offs(pp, diff_key_off);
            CHECK(parallel_for(
                    ctx, compute::nd_range_t(gws_p, lws_p), postprocess_, pp));
        }
        // dV
        {
            compute::range_t gws_p = lws_p;
            gws_p[0] *= utils::div_up(K * D, lws_pp);
            gws_p[1] *= num_kv_heads;
            gws_p[2] *= pd()->desc()->batch();

            compute::kernel_arg_list_t pp;
            pp.append(diff_v);
            pp.append(*diff_v_scratch);
            pp.append((int)(K * D));
            append_strides(pp, dv_off);
            append_offs(pp, diff_val_off);
            CHECK(parallel_for(
                    ctx, compute::nd_range_t(gws_p, lws_p), postprocess_, pp));
        }
    }

    return status::success;
}

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
