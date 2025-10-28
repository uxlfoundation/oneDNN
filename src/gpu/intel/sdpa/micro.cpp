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

#include "gpu/intel/sdpa/micro.hpp"
#include "gpu/intel/sdpa/configs.hpp"

#include "common/c_types_map.hpp"
#include "common/sdpa_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "gemmstone/microkernel_provider.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/microkernels/shim.hpp"

#include <cstdio>
#include <iostream>
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

status_t update_config_from_devenv_values(config_t *config, bool quantized) {
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

status_t micro_t::pd_t::init_conf_microkernels(impl::engine_t *engine) {
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
    config_t *config = nullptr;
    const dim_t thin_q_threshold = 16;
    auto queries = d->queries();
    if (queries == 1) { queries = (d->q_desc.dims[1] / d->kv_head_number); }

    bool thin_q = (queries <= thin_q_threshold);
    bool quantized = with_key_scales() || with_key_zp() || with_value_scales()
            || with_value_zp();
    bool is_integrated = intel_engine->device_info()->is_integrated();
    bool is_f32 = (qry_md()->data_type == data_type::f32);
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
    micro_ukernel_params_t ukernel_params;
    ukernel_params.unroll_m_kq = config->unroll_m_kq;
    ukernel_params.unroll_n_kq = config->unroll_n_kq;
    ukernel_params.unroll_m_vs = config->unroll_m_vs;
    ukernel_params.unroll_n_vs = config->unroll_n_vs;
    ukernel_params.wg_m_kq = config->wg_m_kq;
    ukernel_params.wg_n_kq = config->wg_n_kq;
    ukernel_params.wg_m_vs = config->wg_m_vs;
    ukernel_params.wg_n_vs = config->wg_n_vs;

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel_;

    if (hw_info.gmdid == 0) return status::unimplemented;

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
    problem.Ta_ext = convert_dnnl_to_kernel_type(key_md()->data_type);
    problem.Tb_ext = convert_dnnl_to_kernel_type(qry_md()->data_type);
    if (qry_md()->data_type == data_type::f16) {
        problem.Ta = problem.Tb = Type::f16;
    } else if (qry_md()->data_type == data_type::bf16) {
        problem.Ta = problem.Tb = Type::bf16;
    } else if (qry_md()->data_type == data_type::f32) {
        problem.Ta = problem.Tb = Type::f32;
    } else {
        VCHECK_SDPA_COND(utils::one_of(qry_md()->data_type, data_type::f16,
                                 data_type::bf16),
                "Q tensor's data type must be bf16 or f16");
    }
    problem.Tc = problem.Tc_ext = Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.Tc = problem_kq.Ts
            = (kq_acc_dt() == data_type::f16) ? Type::f16 : Type::f32;

    problem_kq.A.layout = convert_dnnl_to_kernel_layout(key_md());

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

    ukernel_params.problem_kq = {problem_kq};

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;
    opts_kq.scaleA = with_key_scales() && !kq_common_scales;
    opts_kq.offsetA = with_key_zp();

    ukernel_params.opts_kq = {opts_kq};

    /* Set up problem size information */
    SizeParams heuristic_sizes;
    // quanatizing sizes to large intervals allows kernel
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
    heuristic_sizes.batch = utils::rnd_up_pow2(d->batch_size());

    ukernel_params.sizes_kq = {heuristic_sizes};

    /* Set up GEMMProblem structure for second GEMM: V * S  */
    auto problem_vs = std::move(problem);
    problem_vs.Tc = problem_vs.Ts
            = (vs_acc_dt() == data_type::f16) ? Type::f16 : Type::f32;

    bool vs_common_scales = with_quantize_common(d->vs_scales);
    bool vs_common_zp = with_quantize_common(d->vs_zero_points);

    problem_vs.Ta_ext = convert_dnnl_to_kernel_type(val_md()->data_type);
    problem_vs.A.layout = convert_dnnl_to_kernel_layout(val_md());
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
    const memory_desc_wrapper val_mdw(val_md());
    auto ldv = static_cast<int>(
            gemm_desc_t::get_ld(*val_md()) * val_mdw.data_type_size());
    problem_vs.A.setAlignment(alignmentForLD(ldv));
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
    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;
    opts_vs.scaleA = with_value_scales() && !vs_common_scales;
    opts_vs.offsetA = with_value_zp();

    ukernel_params.opts_vs = {opts_vs};

    conf.ukernel_config = ukernel_params;

    return status::success;
}

status_t micro_t::init(impl::engine_t *engine) {
    CHECK(create_kernel(
            engine, kernel_, pd()->conf.get_kernel_names()[0], pd()->conf));
    if (!kernel_) return status::runtime_error;
    return status::success;
}

status_t micro_t::pd_t::init_conf(impl::engine_t *engine) {
    using namespace micro;

    auto *pd = this;
    auto *d = pd->desc();

    data_type_t data_t = pd->dst_md()->data_type;
    conf.data_t = data_t;
    conf.ndims = pd_t::ndims;

    const memory_desc_wrapper qry_mdw(pd->qry_md());
    const memory_desc_wrapper key_mdw(pd->key_md());
    const memory_desc_wrapper val_mdw(pd->val_md());
    const memory_desc_wrapper dst_mdw(pd->dst_md());
    const memory_desc_wrapper msk_mdw(pd->attn_mask_md());

    conf.key_data_t = key_mdw.data_type();
    conf.qry_data_t = qry_mdw.data_type();
    conf.val_data_t = val_mdw.data_type();
    conf.dst_data_t = dst_mdw.data_type();

    conf.msk_data_t = data_type::undef;
    if (pd->with_attn_mask()) { conf.msk_data_t = msk_mdw.data_type(); }

    conf.key_scales_data_t = pd->key_scales_dt();
    conf.value_scales_data_t = pd->value_scales_dt();

    conf.key_zp_data_t = pd->key_zp_dt();
    conf.value_zp_data_t = pd->value_zp_dt();

    auto Q_num_heads_dim = qry_mdw.dims()[1];
    conf.kv_group_size = static_cast<int>(Q_num_heads_dim / d->kv_head_number);

    auto ldq = gemm_desc_t::get_ld(*pd->qry_md()) * qry_mdw.data_type_size();
    auto ldk = gemm_desc_t::get_ld(*pd->key_md()) * key_mdw.data_type_size();
    auto ldv = gemm_desc_t::get_ld(*pd->val_md()) * val_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*pd->dst_md()) * dst_mdw.data_type_size();

    conf.q_align = alignmentForLD(int(ldq));
    conf.k_align = alignmentForLD(int(ldk));
    conf.v_align = alignmentForLD(int(ldv));
    conf.a_align = alignmentForLD(int(lda));

    conf.transpose_k = gemm_desc_t::get_trans(*pd->key_md()) == dnnl_trans;

    int kq_scale_mask = (static_cast<int>(pd->with_key_scales()) << 1)
            | static_cast<int>(with_quantize_common(d->kq_scales));
    conf.kq_scale_mask = kq_scale_mask;

    int vs_scale_mask = (static_cast<int>(pd->with_value_scales()) << 1)
            | static_cast<int>(with_quantize_common(d->vs_scales));
    conf.vs_scale_mask = vs_scale_mask;

    int kq_zp_mask = (static_cast<int>(pd->with_key_zp()) << 1)
            | static_cast<int>(with_quantize_common(d->kq_zero_points));
    conf.kq_zp_mask = kq_zp_mask;

    int vs_zp_mask = (static_cast<int>(pd->with_value_zp()) << 1)
            | static_cast<int>(with_quantize_common(d->vs_zero_points));
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
    conf.key_zp_elements_per_byte = elems_per_byte(pd->key_zp_dt());
    conf.val_elements_per_byte = elems_per_byte(val_mdw.data_type());
    conf.val_zp_elements_per_byte = elems_per_byte(pd->value_zp_dt());

    conf.key_group_size = 1;
    conf.val_group_size = 1;
    if (pd->with_key_scales() || pd->with_key_zp())
        conf.key_group_size = pd->key_group_size();
    if (pd->with_value_scales() || pd->with_value_zp())
        conf.val_group_size = pd->value_group_size();

    conf.scale_data_t = pd->scale_md()->data_type;

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

    /* Set up microkernel strategy */
    const config_t config = {conf.ukernel_config.unroll_m_kq,
            conf.ukernel_config.unroll_n_kq, conf.ukernel_config.unroll_m_vs,
            conf.ukernel_config.unroll_n_vs, conf.ukernel_config.wg_m_kq,
            conf.ukernel_config.wg_n_kq, conf.ukernel_config.wg_m_vs,
            conf.ukernel_config.wg_n_vs};

    const int kq_wg_tile_m = config.wg_m_kq * config.unroll_m_kq;
    const int kq_wg_tile_n = config.wg_n_kq * config.unroll_n_kq;
    const int vs_wg_tile_m = config.wg_m_vs * config.unroll_m_vs;
    int tile_k = kq_wg_tile_m;
    int tile_v = vs_wg_tile_m;

    bool d_full = (d->head_size() == pd->d_max());
    bool v_full = (d->head_size() == tile_v);

    auto Q = d->queries();
    const dim_t Q_per_kv_group = (Q == 1 ? Q * conf.kv_group_size : Q);
    bool q_full = ((Q_per_kv_group % kq_wg_tile_n) != 0);
    conf.remainder_q = d_full && q_full;

    conf.d_full = d_full;
    conf.arch_gte_hpc = (pd->arch() >= compute::gpu_arch_t::xe_hpc);

    conf.block_q = conf.block_a = conf.block_2d_a = false;
    if (d_full) {
        conf.block_q = (ldq % 4 == 0);
        conf.block_a = (lda % 4 == 0 && v_full);
    } else if (pd->arch() >= compute::gpu_arch_t::xe_hpc
            && config.unroll_m_vs < 64) {
        auto vbytes = d->values() * val_mdw.data_type_size();
        if (lda % 16 == 0 && vbytes % 4 == 0) conf.block_2d_a = true;
    }

    if (pd->arch() >= compute::gpu_arch_t::xe_hpc) {
        conf.prefetch_mask = true;
        conf.prefetch_k0 = true;
        conf.prefetch_k = true;
        conf.prefetch_v = true;
        bool no_rem = d_full && v_full && (d->keys() % tile_k == 0);
        conf.prefetch_remainder = !no_rem;
        conf.prefetch_d_max = nstl::min(pd->d_max(), 64);
    } else {
        conf.prefetch_mask = conf.prefetch_k0 = conf.prefetch_k
                = conf.prefetch_v = conf.prefetch_remainder = false;
        conf.prefetch_d_max = 0;
    }

    conf.q_arrive_await_barrier = (Q > 1);
    conf.softmax_inf_as_zero
            = (d->softmax_alg == alg_kind::softmax_accurate_inf_as_zero);
    conf.use_systolic_ukernel = pd->use_systolic_ukernel();
    conf.kq_f16_accumulate = (kq_acc_dt() == data_type::f16);
    conf.vs_f16_accumulate = (vs_acc_dt() == data_type::f16);
    return status::success;
}

status_t micro_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace micro;

    kernel_ctx.define_int("NDIMS", ndims);
    kernel_ctx.set_data_type(data_t);

    def_data_type(kernel_ctx, key_data_t, "KEY");
    def_data_type(kernel_ctx, qry_data_t, "QRY");
    def_data_type(kernel_ctx, val_data_t, "VAL");
    def_data_type(kernel_ctx, dst_data_t, "DST");
    def_data_type(kernel_ctx, scale_data_t, "SCALE", !with_host_scale);

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

    gemmstone::HWInformation hw_info;
    gemmstone::GEMMProblem problem_kq, problem_vs;
    micro::GEMMProtocol::Options opts_kq, opts_vs;
    gemmstone::SizeParams sizes_kq, sizes_vs;

    deserialize_config_to_gemmstone(hw_info, problem_kq, problem_vs, opts_kq,
            opts_vs, sizes_kq, sizes_vs, ukernel_config);

    micro::Package gemm_kq, gemm_vs;

    /* Set up microkernel strategy */
    const config_t config
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
    try {
        gemm_kq = selectGEMMMicrokernel(
                opts_kq, hw_info, sizes_kq, problem_kq, reqs_kq);
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_kq microkernel generation failure with message: %s",
                ex.what());
    }

    /* Ask microkernel provider for microkernel */
    try {
        if (use_systolic_ukernel) {
            auto adjust_vs = [](GEMMStrategy &strategy) {
                /* Enable dpasw */
                strategy.dpasw |= strategy.fused;
            };
            gemm_vs = selectGEMMMicrokernel(
                    opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs, adjust_vs);
        } else {
            gemm_vs = selectGEMMMicrokernel(
                    opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs);
        }
    } catch (const std::runtime_error &ex) {
        VCHECK_SDPA_COND(false,
                "gemm_vs microkernel generation failure with message: %s",
                ex.what());
    }
    VDEBUGINFO(4, primitive, sdpa, "kq_gemm: %s, vs_gemm: %s,",
            problem_kq.toString().c_str(), problem_vs.toString().c_str());

    /* Generate microkernel shims */
    ShimOptions shimOptions;
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

status_t micro_t::execute(const exec_ctx_t &ctx) const {
    const auto &conf = pd()->conf;

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

    const int kv_group_size = pd()->conf.kv_group_size;
    const dim_t Q = pd()->desc()->queries();
    const dim_t K = pd()->desc()->keys();
    const dim_t D = pd()->desc()->head_size();
    const dim_t Q_per_kv_group = (Q == 1 ? Q * kv_group_size : Q);

    const config_t config = {conf.ukernel_config.unroll_m_kq,
            conf.ukernel_config.unroll_n_kq, conf.ukernel_config.unroll_m_vs,
            conf.ukernel_config.unroll_n_vs, conf.ukernel_config.wg_m_kq,
            conf.ukernel_config.wg_n_kq, conf.ukernel_config.wg_m_vs,
            conf.ukernel_config.wg_n_vs};
    const int kq_wg_tile_m = config.wg_m_kq * config.unroll_m_kq;
    const int kq_wg_tile_n = config.wg_n_kq * config.unroll_n_kq;
    auto wg_tile_q = kq_wg_tile_n;
    auto sg_per_wg = config.wg_m_kq * config.wg_n_kq;

    const memory_desc_wrapper qry_mdw(pd()->qry_md());
    const memory_desc_wrapper key_mdw(pd()->key_md());
    const memory_desc_wrapper val_mdw(pd()->val_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper msk_mdw(pd()->attn_mask_md());
    using offset_t = decltype(offsets_t().src_off);

    offset_t key_off, qry_off, val_off, dst_off, msk_off;

    set_offsets(key_mdw, key_off);
    set_offsets(qry_mdw, qry_off);
    set_offsets(val_mdw, val_off);
    set_offsets(dst_mdw, dst_off);
    set_offsets(msk_mdw, msk_off);

    //TODO: change arg_list type based on large_idx
    //bool use_int32_offset = conf.use_int32_offset;

    auto append_offs
            = [](compute::kernel_arg_list_t &arg_list, const offset_t &offs) {
                  compute::int64x4_t dims4
                          = {offs[3][0], offs[3][1], offs[3][2], offs[3][3]};
                  compute::int64x4_t strides4
                          = {offs[1][0], offs[1][1], offs[1][2], offs[1][3]};
                  arg_list.append(dims4);
                  arg_list.append(strides4);
              };

    int mask_type = static_cast<int>(pd()->desc()->mask_type);
    compute::kernel_arg_list_t arg_list;

    const memory_desc_wrapper scale_mdw(pd()->scale_md());
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
                pd()->scale_md()->data_type, &scalar_scale, 0);
        inv_scalar_scale = 1. / scalar_scale;
    }

    arg_list.append(key);
    arg_list.append(qry);
    arg_list.append(val);
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

    append_offs(arg_list, key_off);
    append_offs(arg_list, qry_off);
    append_offs(arg_list, val_off);
    append_offs(arg_list, dst_off);

    if (pd()->with_attn_mask()) { append_offs(arg_list, msk_off); }
    const int remainder_k = (K % kq_wg_tile_m) != 0;

    arg_list.append(remainder_k);

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
    gws[2] *= pd()->dst_md()->dims[0];

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
