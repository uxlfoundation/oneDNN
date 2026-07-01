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

#include "gpu/intel/gated_mlp/micro_horz.hpp"

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/type_helpers.hpp"
#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gated_mlp {

//#define UGEMM_UP_ONLY

namespace {

struct gated_mlp_config_t {
    int unroll_m_gwu, unroll_n_gwu;
    int wg_m_gwu, wg_n_gwu;
};

// TODO: refine the config selection process
gated_mlp_config_t select_config(const compute::device_info_t &dev_info) {
    if (dev_info.gpu_arch() > compute::gpu_arch_t::xe3) {
        return {32, 32, 1, 1};
    } else if (dev_info.gpu_arch() > compute::gpu_arch_t::xe_hpg) {
        return {32, 16, 1, 1};
    } else {
        return {8, 8, 4, 4};
    }
}

#define CASE(gen, name, ...) \
    gen(name##_scales, scales_, __VA_ARGS__) \
            gen(name##_zp, zero_points_, __VA_ARGS__)

#define WITH_QUANT(name, quant, arg) \
    bool with_##name(const micro_horz_t::pd_t *pd) { \
        return !pd->attr()->quant.has_default_values(arg); \
    }
CASE(WITH_QUANT, src, DNNL_ARG_SRC)
CASE(WITH_QUANT, wts_gate, DNNL_ARG_WEIGHTS_GATE)
CASE(WITH_QUANT, wts_up, DNNL_ARG_WEIGHTS_UP)
CASE(WITH_QUANT, wts_down, DNNL_ARG_WEIGHTS_DOWN)
#undef WITH_QUANT

#define WITH_QUANT_COMMON(name, quant, arg) \
    bool with_common_##name(const micro_horz_t::pd_t *pd) { \
        return with_##name(pd) && (pd->attr()->quant.get_mask(arg) == 0); \
    }
CASE(WITH_QUANT_COMMON, src, DNNL_ARG_SRC)
CASE(WITH_QUANT_COMMON, wts_gate, DNNL_ARG_WEIGHTS_GATE)
CASE(WITH_QUANT_COMMON, wts_up, DNNL_ARG_WEIGHTS_UP)
CASE(WITH_QUANT_COMMON, wts_down, DNNL_ARG_WEIGHTS_DOWN)
#undef QUANT_COMMON

#define QUANT_DT(name, quant, arg) \
    data_type_t name##_dt(const micro_horz_t::pd_t *pd) { \
        return pd->attr()->quant.get_data_type(arg); \
    }
CASE(QUANT_DT, src, DNNL_ARG_SRC)
CASE(QUANT_DT, wts_gate, DNNL_ARG_WEIGHTS_GATE)
CASE(QUANT_DT, wts_up, DNNL_ARG_WEIGHTS_UP)
CASE(QUANT_DT, wts_down, DNNL_ARG_WEIGHTS_DOWN)
#undef QUANT_DT

#define QUANT_GS(name, quant, arg, idx) \
    dim_t name##_group_size(const micro_horz_t::pd_t *pd) { \
        return pd->attr()->quant.get_group(arg, idx); \
    }
CASE(QUANT_GS, src, DNNL_ARG_SRC, 1)
CASE(QUANT_GS, wts_gate, DNNL_ARG_WEIGHTS_GATE, 0)
CASE(QUANT_GS, wts_up, DNNL_ARG_WEIGHTS_UP, 0)
CASE(QUANT_GS, wts_down, DNNL_ARG_WEIGHTS_DOWN, 0)
#undef QUANT_GS

#undef CASE

int sg_size(impl::engine_t *engine) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    return intel_engine->device_info()->min_subgroup_size();
}

} // anonymous namespace

status_t micro_horz_t::pd_t::init(impl::engine_t *engine) {
    memory_desc_t inter_md;
    CHECK(get_gate_dst_md(inter_md));
    CHECK(set_default_formats());
    CHECK(init_microkernels(engine, &inter_md));

#ifndef UGEMM_UP_ONLY
    primitive_attr_t down_attr;
    CHECK(move_attr_down(down_attr, DNNL_ARG_WEIGHTS_DOWN, DNNL_ARG_WEIGHTS));
    auto down_desc = matmul_desc_t();
    CHECK(impl::matmul_desc_init(&down_desc, &inter_md,
            arg_md(DNNL_ARG_WEIGHTS_DOWN), nullptr, arg_md(DNNL_ARG_DST)));
    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&down_desc, &down_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;
    gemm_down_pd_ = *(++it);
    if (!gemm_down_pd_) return status::unimplemented;

    using namespace memory_tracking::names;
    auto scratchpad = scratchpad_registry().registrar();
    const memory_desc_wrapper inter_mdw(inter_md);
    scratchpad.book(
            key_matmul_src_trans, inter_mdw.size(), 1, OCL_BUFFER_ALIGNMENT);
    scratchpad.book(key_nested_multiple + DNNL_ARG_WEIGHTS_DOWN,
            gemm_down_pd_->scratchpad_registry());
#endif
    return status::success;
}

status_t micro_horz_t::pd_t::init_microkernels(
        impl::engine_t *engine, const memory_desc_t *inter_md) {
    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();

    VCONDCHECK(primitive, create, dispatch, gated_mlp,
            compute::mayiuse_microkernels(intel_engine), status::unimplemented,
            "Microkernels not supported by the OpenCL driver.");

    auto config = select_config(*dev_info);

#ifdef DNNL_DEV_MODE
    auto gmlp_conf = gpu_utils::dev_getenv("GMLP_CONF", std::string());
    if (!gmlp_conf.empty()) {
        std::vector<int> tokens;
        std::stringstream ss(gmlp_conf);
        std::string tmp;
        try {
            while (getline(ss, tmp, ' '))
                tokens.push_back(std::stoi(tmp));
            if (tokens.size() == 4) {
                config.unroll_m_gwu = tokens[0];
                config.unroll_n_gwu = tokens[1];
                config.wg_m_gwu = tokens[2];
                config.wg_n_gwu = tokens[3];
            }
        } catch (...) {}
        printf("GMLP_CONF: (%d %d %d %d)\n", config.unroll_m_gwu,
                config.unroll_n_gwu, config.wg_m_gwu, config.wg_n_gwu);
    }
#endif

    gemmstone::microkernel::HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = intel_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);
    hw_info.isEfficient64Bit = dev_info->is_efficient_64bit();

    if (hw_info.gmdid == 0) return status::unimplemented;

    gemmstone::GEMMProblem problem;
    auto a_dt = arg_md(DNNL_ARG_WEIGHTS_GATE)->data_type;
    auto b_dt = arg_md(DNNL_ARG_SRC)->data_type;
    problem.Ta = problem.Ta_ext = gemm::jit::convert_dnnl_to_kernel_type(a_dt);
    problem.Tb = problem.Tb_ext = gemm::jit::convert_dnnl_to_kernel_type(b_dt);
    problem.Tc = problem.Tc_ext = problem.Ts
            = gemm::jit::convert_dnnl_to_kernel_type(get_accum_type());

    // Mixed int8/int4 DPAS support:
    // - Xe3p: Not supported, requires int4->int8 upconversion
    if (dev_info->gpu_arch() == compute::gpu_arch_t::xe3p) {
        if ((problem.Ta_ext.isInt4()) && (problem.Tb_ext.isInt8()))
            problem.Ta = gemmstone::Type::s8;
        if ((problem.Ta_ext.isInt8()) && (problem.Tb_ext.isInt4()))
            problem.Tb = gemmstone::Type::s8;
    }
    VCONDCHECK(primitive, create, dispatch, gated_mlp,
            (problem.Tc != gemmstone::Type::invalid)
                    && (problem.Ta != gemmstone::Type::invalid)
                    && (problem.Tb != gemmstone::Type::invalid),
            status::unimplemented, "Incompatible A/B/C types in uGEMM.");

    auto problem_wgu = std::move(problem);
    problem_wgu.A.layout = gemmstone::MatrixLayout::T;
    problem_wgu.B.layout = gemmstone::MatrixLayout::N;
    problem_wgu.C.layout = gemmstone::MatrixLayout::N;

    const memory_desc_wrapper src_mdw(arg_md(DNNL_ARG_SRC));
    const memory_desc_wrapper W_gate_mdw(arg_md(DNNL_ARG_WEIGHTS_GATE));
    const memory_desc_wrapper W_up_mdw(arg_md(DNNL_ARG_WEIGHTS_UP));
    auto alignment = [](const memory_desc_wrapper &mdw) {
        return int(gemm_desc_t::get_ld(*mdw.md_) * mdw.data_type_size());
    };
    if (alignment(W_gate_mdw) != alignment(W_up_mdw))
        return status::unimplemented;
    problem_wgu.A.setAlignment(
            gemmstone::microkernel::alignmentForLD(alignment(W_gate_mdw)));
    problem_wgu.B.setAlignment(
            gemmstone::microkernel::alignmentForLD(alignment(src_mdw)));

    if (with_wts_gate_scales(this) != with_wts_up_scales(this))
        return status::unimplemented;
    if (wts_gate_scales_group_size(this) != wts_up_scales_group_size(this))
        return status::unimplemented;

    if (with_wts_gate_zp(this) != with_wts_up_zp(this))
        return status::unimplemented;
    if (wts_gate_zp_group_size(this) != wts_up_zp_group_size(this))
        return status::unimplemented;

    if (with_src_scales(this) && !with_common_src_scales(this)) {
        auto scale_dt = src_scales_dt(this);
        problem_wgu.Tb_scale = gemm::jit::convert_dnnl_to_kernel_type(scale_dt);
        problem_wgu.B_scale.alignment
                = uint8_t(types::data_type_size(scale_dt));
        problem_wgu.bsPtrDims = 2; // no 1D-scales for uGEMM at this point
        problem_wgu.B_scale.layout = (problem_wgu.bsPtrDims > 1)
                ? gemmstone::MatrixLayout::N
                : gemmstone::MatrixLayout::T;
    }
    if (with_wts_gate_scales(this) && !with_common_wts_gate_scales(this)) {
        auto scale_dt = wts_gate_scales_dt(this);
        problem_wgu.Ta_scale = gemm::jit::convert_dnnl_to_kernel_type(scale_dt);
        problem_wgu.A_scale.alignment
                = uint8_t(types::data_type_size(scale_dt));
        problem_wgu.asPtrDims = 2; // no 1D-scales for uGEMM at this point
        problem_wgu.A_scale.layout = gemmstone::MatrixLayout::N;
    }

    if (with_src_zp(this)) {
        if (problem_wgu.Tb.isInt4()) problem_wgu.Tb = gemmstone::Type::s8;
        auto zp_dt = src_zp_dt(this);
        problem_wgu.bOffset = gemmstone::ABOffset::Calc;
        problem_wgu.Tbo = gemm::jit::convert_dnnl_to_kernel_type(zp_dt);
        problem_wgu.BO.alignment = uint8_t(types::data_type_size(zp_dt));
        problem_wgu.boPtrDims = (!with_common_src_zp(this))
                ? (src_zp_group_size(this) > 1) ? 2 : 1
                : 0;
        problem_wgu.BO.layout = (problem_wgu.boPtrDims > 1)
                ? gemmstone::MatrixLayout::N
                : gemmstone::MatrixLayout::T;
    }
    if (with_wts_gate_zp(this)) {
        if (problem_wgu.Ta.isInt4()) problem_wgu.Ta = gemmstone::Type::s8;
        auto zp_dt = wts_gate_zp_dt(this);
        problem_wgu.aOffset = gemmstone::ABOffset::Calc;
        problem_wgu.Tao = gemm::jit::convert_dnnl_to_kernel_type(zp_dt);
        problem_wgu.AO.alignment = uint8_t(types::data_type_size(zp_dt));
        problem_wgu.aoPtrDims = (!with_common_wts_gate_zp(this))
                ? (wts_gate_zp_group_size(this) > 1) ? 2 : 1
                : 0;
        problem_wgu.AO.layout = gemmstone::MatrixLayout::N;
    }

    if (with_src_scales(this) || with_src_zp(this)) {
        problem_wgu.bqGroupN = 1;
        problem_wgu.bqGroupK = int(IC());
        // TODO
        //VCONDCHECK(primitive, create, dispatch, gated_mlp,
        //        src_scales_group_size(this) == src_zp_group_size(this),
        //        status::unimplemented, "Incompatible src scale/zp groups.");
        if (src_scales_group_size(this) > 1)
            problem_wgu.bqGroupK = int(src_scales_group_size(this));
        if (src_zp_group_size(this) > 1)
            problem_wgu.bqGroupK = int(src_zp_group_size(this));
    }
    if (with_wts_gate_scales(this) || with_wts_gate_zp(this)) {
        problem_wgu.aqGroupM = 1;
        problem_wgu.aqGroupK = int(IC());
        // TODO
        //VCONDCHECK(primitive, create, dispatch, gated_mlp,
        //        wts_gate_scales_group_size(this) == wts_gate_zp_group_size(this),
        //        status::unimplemented, "Incompatible gate scale/zp groups.");
        if (wts_gate_scales_group_size(this) > 1)
            problem_wgu.aqGroupK = int(wts_gate_scales_group_size(this));
        if (wts_gate_zp_group_size(this) > 1)
            problem_wgu.aqGroupK = int(wts_gate_zp_group_size(this));
    }

    // upconversions
    bool upconvert = false;

    // convert to F16/F16 if INT8/F16 or F16/INT8
    if ((problem_wgu.Ta_ext.isInt8() && problem_wgu.Tb_ext.isFP())
            || (problem_wgu.Ta_ext.isFP() && problem_wgu.Tb_ext.isInt8())) {
        problem_wgu.Ta = problem_wgu.Tb = (problem_wgu.Ta_ext.isFP())
                ? problem_wgu.Ta_ext
                : problem_wgu.Tb_ext;
        upconvert = true;
    }
    // convert to F16 if dual INT8 gets quantized (see gemm/jit/gen_kernel.cpp)
    if (((src_scales_group_size(this) > 1)
                || (wts_gate_scales_group_size(this) > 1) || with_src_zp(this))
            && problem_wgu.Ta_ext.isInt8() && problem_wgu.Tb_ext.isInt8()
            && with_wts_gate_zp(this)) {
        problem_wgu.Ta = problem_wgu.Tb = gemmstone::Type::f16;
        upconvert = true;
    }
    // bumping up the quant dims for upconverted cases
    if (upconvert) {
        if (problem_wgu.asPtrDims == 1) {
            problem_wgu.asPtrDims = 2;
            problem_wgu.A_scale.layout = gemmstone::MatrixLayout::N;
            VCONDCHECK(primitive, create, dispatch, gated_mlp,
                    problem_wgu.aqGroupK == IC(), status::unimplemented,
                    "Incompatible gate scale/zp groups.");
        }
        if (problem_wgu.bsPtrDims == 1) {
            problem_wgu.bsPtrDims = 2;
            problem_wgu.B_scale.layout = gemmstone::MatrixLayout::N;
            VCONDCHECK(primitive, create, dispatch, gated_mlp,
                    problem_wgu.bqGroupK == IC(), status::unimplemented,
                    "Incompatible src scale/zp groups.");
        }
    }

    /* Set up transposed problem size */
    gemmstone::SizeParams sizes;
    sizes.m = OC();
    sizes.n = MB();
    sizes.k = IC();
    sizes.batch = 1;

    std::vector<gemmstone::StrategyRequirement> reqs_wgu;

    reqs_wgu.push_back(
            gemmstone::StrategyRequirement::UnrollM == config.unroll_m_gwu);
    reqs_wgu.push_back(
            gemmstone::StrategyRequirement::UnrollN == config.unroll_n_gwu);

    reqs_wgu.push_back(gemmstone::StrategyRequirement::WGM == config.wg_m_gwu);
    reqs_wgu.push_back(gemmstone::StrategyRequirement::WGN == config.wg_n_gwu);

    gemmstone::microkernel::GEMMOptions opts_wgu;
    opts_wgu.scaleB = with_src_scales(this) && !with_common_src_scales(this);
    opts_wgu.offsetB = with_src_zp(this);
    opts_wgu.scaleA
            = with_wts_gate_scales(this) && !with_common_wts_gate_scales(this);
    opts_wgu.offsetA = with_wts_gate_zp(this);
    opts_wgu.slmPtr = true;

    try {
        gemm_gate_up_pkg_
                = selectGEMM(opts_wgu, hw_info, sizes, problem_wgu, reqs_wgu);
    } catch (std::exception &e) {
        VDISPATCH_GATED_MLP(false,
                "gemm_gateup microkernel generation failed with message: %s",
                e.what());
    }

    size_t kern_slm = gemm_gate_up_pkg().getSetting("slm_size")
            + types::data_type_size(get_accum_type()) * config.unroll_m_gwu
                    * config.wg_m_gwu * config.unroll_n_gwu * config.wg_n_gwu;
    size_t slm = compute::device_info_t::max_slm_size(dev_info->product());

    VCONDCHECK(primitive, create, dispatch, gated_mlp, kern_slm <= slm,
            status::unimplemented, "Insufficient SLM size for uGEMM.");

    return status::success;
}

status_t micro_horz_t::init(impl::engine_t *engine) {
    compute::kernel_ctx_t kernel_ctx;

    memory_desc_t inter_md;
    CHECK(pd()->get_gate_dst_md(inter_md));
    const memory_desc_wrapper inter_mdw(inter_md);
    const memory_desc_wrapper src_mdw(pd()->arg_md(DNNL_ARG_SRC));
    const memory_desc_wrapper W_gate_mdw(pd()->arg_md(DNNL_ARG_WEIGHTS_GATE));
    const memory_desc_wrapper W_up_mdw(pd()->arg_md(DNNL_ARG_WEIGHTS_UP));
    const memory_desc_wrapper W_down_mdw(pd()->arg_md(DNNL_ARG_WEIGHTS_DOWN));
    const memory_desc_wrapper dst_mdw(pd()->arg_md(DNNL_ARG_DST));

    kernel_ctx.set_data_type(dst_mdw.data_type());

    using offset_t = decltype(offsets_t().src_off);
    offset_t inter_off, src_off, W_gate_off, W_up_off, W_down_off, dst_off;
    set_offsets(inter_mdw, inter_off);
    set_offsets(src_mdw, src_off);
    set_offsets(W_gate_mdw, W_gate_off);
    set_offsets(W_up_mdw, W_up_off);
    set_offsets(W_down_mdw, W_down_off);
    set_offsets(dst_mdw, dst_off);

    kernel_ctx.define_int("WGU_QUANT_S0", pd()->OC());
    if (with_src_scales(pd()) && !with_common_src_scales(pd())) {
        kernel_ctx.define_int(
                "SRC_QUANT_S0", pd()->IC() / src_scales_group_size(pd()));
    } else if (with_src_zp(pd())) {
        kernel_ctx.define_int(
                "SRC_QUANT_S0", pd()->IC() / src_zp_group_size(pd()));
    }

    def_offsets(inter_off, kernel_ctx, "INTER", inter_mdw.ndims());
    def_offsets(src_off, kernel_ctx, "SRC", src_mdw.ndims());
    def_offsets(W_gate_off, kernel_ctx, "W_GATE", W_gate_mdw.ndims());
    def_offsets(W_up_off, kernel_ctx, "W_UP", W_up_mdw.ndims());
    def_offsets(W_down_off, kernel_ctx, "W_DOWN", W_down_mdw.ndims());
    def_offsets(dst_off, kernel_ctx, "DST", dst_mdw.ndims());
    kernel_ctx.define_int("NDIMS", inter_mdw.ndims());

    def_data_type(kernel_ctx, pd()->get_accum_type(), "ACCUM");
    def_data_type(kernel_ctx, inter_mdw.data_type(), "INTER");
    def_data_type(kernel_ctx, src_mdw.data_type(), "SRC");
    def_data_type(kernel_ctx, W_gate_mdw.data_type(), "WTS_GATE");
    def_data_type(kernel_ctx, W_up_mdw.data_type(), "WTS_UP");
    def_data_type(kernel_ctx, W_down_mdw.data_type(), "WTS_DOWN");
    def_data_type(kernel_ctx, dst_mdw.data_type(), "DST");

    def_data_type(kernel_ctx, src_scales_dt(pd()), "SRC_ATTR_SCALES");
    def_data_type(kernel_ctx, wts_gate_scales_dt(pd()), "WTS_GATE_ATTR_SCALES");
    def_data_type(kernel_ctx, wts_up_scales_dt(pd()), "WTS_UP_ATTR_SCALES");
    def_data_type(kernel_ctx, wts_down_scales_dt(pd()), "WTS_DOWN_ATTR_SCALES");

    def_data_type(kernel_ctx, src_zp_dt(pd()), "SRC_ATTR_ZP");
    def_data_type(kernel_ctx, wts_gate_zp_dt(pd()), "WTS_GATE_ATTR_ZP");
    def_data_type(kernel_ctx, wts_up_zp_dt(pd()), "WTS_UP_ATTR_ZP");
    def_data_type(kernel_ctx, wts_down_zp_dt(pd()), "WTS_DOWN_ATTR_ZP");

    auto ldi = gemm_desc_t::get_ld(*inter_mdw.md_) * inter_mdw.data_type_size();
    auto lds = gemm_desc_t::get_ld(*src_mdw.md_) * src_mdw.data_type_size();
    auto lda = gemm_desc_t::get_ld(*dst_mdw.md_) * dst_mdw.data_type_size();
    auto ldwgu = gemm_desc_t::get_ld(*W_gate_mdw.md_)
            * W_gate_mdw.data_type_size();

    kernel_ctx.define_int(
            "INTER_ALIGN", gemmstone::microkernel::alignmentForLD(int(ldi)));
    kernel_ctx.define_int(
            "SRC_ALIGN", gemmstone::microkernel::alignmentForLD(int(lds)));
    kernel_ctx.define_int(
            "DST_ALIGN", gemmstone::microkernel::alignmentForLD(int(lda)));
    kernel_ctx.define_int(
            "WGU_ALIGN", gemmstone::microkernel::alignmentForLD(int(ldwgu)));

    switch (pd()->activation()) {
        case (alg_kind::eltwise_gelu_erf):
            kernel_ctx.define_int("ACTIVATION_GELU_ERF", 1);
            break;
        case (alg_kind::eltwise_gelu_tanh):
            kernel_ctx.define_int("ACTIVATION_GELU_TANH", 1);
            break;
        case (alg_kind::eltwise_swish):
        default: kernel_ctx.define_int("ACTIVATION_SWISH", 1);
    }

    kernel_ctx.define_int("SRC_SCALES",
            (int(with_src_scales(pd())) << 1)
                    | int(with_common_src_scales(pd())));
    kernel_ctx.define_int("WTS_GATE_SCALES",
            (int(with_wts_gate_scales(pd())) << 1)
                    | int(with_common_wts_gate_scales(pd())));
    kernel_ctx.define_int("WTS_UP_SCALES",
            (int(with_wts_up_scales(pd())) << 1)
                    | int(with_common_wts_up_scales(pd())));
    kernel_ctx.define_int("WTS_DOWN_SCALES",
            (int(with_wts_down_scales(pd())) << 1)
                    | int(with_common_wts_down_scales(pd())));

    kernel_ctx.define_int("SRC_ZERO_POINTS",
            (int(with_src_zp(pd())) << 1) | int(with_common_src_zp(pd())));
    kernel_ctx.define_int("WTS_GATE_ZERO_POINTS",
            (int(with_wts_gate_zp(pd())) << 1)
                    | int(with_common_wts_gate_zp(pd())));
    kernel_ctx.define_int("WTS_UP_ZERO_POINTS",
            (int(with_wts_up_zp(pd())) << 1)
                    | int(with_common_wts_up_zp(pd())));
    kernel_ctx.define_int("WTS_DOWN_ZERO_POINTS",
            (int(with_wts_down_zp(pd())) << 1)
                    | int(with_common_wts_down_zp(pd())));

    using namespace data_type;
    auto elems_per_byte = [](data_type_t dt) {
        switch (dt) {
            case u4:
            case s4: return 2;
            default: return 1;
        }
    };
    kernel_ctx.define_int(
            "SRC_ELEMENTS_PER_BYTE", elems_per_byte(src_mdw.data_type()));
    kernel_ctx.define_int("WTS_GATE_ELEMENTS_PER_BYTE",
            elems_per_byte(W_gate_mdw.data_type()));
    kernel_ctx.define_int(
            "WTS_UP_ELEMENTS_PER_BYTE", elems_per_byte(W_up_mdw.data_type()));
    kernel_ctx.define_int("WTS_DOWN_ELEMENTS_PER_BYTE",
            elems_per_byte(W_down_mdw.data_type()));

    kernel_ctx.define_int(
            "SRC_ZP_ELEMENTS_PER_BYTE", elems_per_byte(src_zp_dt(pd())));
    kernel_ctx.define_int("WTS_GATE_ZP_ELEMENTS_PER_BYTE",
            elems_per_byte(wts_gate_zp_dt(pd())));
    kernel_ctx.define_int(
            "WTS_UP_ZP_ELEMENTS_PER_BYTE", elems_per_byte(wts_up_zp_dt(pd())));
    kernel_ctx.define_int("WTS_DOWN_ZP_ELEMENTS_PER_BYTE",
            elems_per_byte(wts_down_zp_dt(pd())));

    if (with_src_scales(pd()))
        kernel_ctx.define_int(
                "SRC_SCALES_GROUP_SIZE", src_scales_group_size(pd()));
    if (with_src_zp(pd()))
        kernel_ctx.define_int("SRC_ZP_GROUP_SIZE", src_zp_group_size(pd()));

    if (with_wts_gate_scales(pd()))
        kernel_ctx.define_int(
                "WTS_GATE_SCALES_GROUP_SIZE", wts_gate_scales_group_size(pd()));
    if (with_wts_gate_zp(pd()))
        kernel_ctx.define_int(
                "WTS_GATE_ZP_GROUP_SIZE", wts_gate_zp_group_size(pd()));

    if (with_wts_up_scales(pd()))
        kernel_ctx.define_int(
                "WTS_UP_SCALES_GROUP_SIZE", wts_up_scales_group_size(pd()));
    if (with_wts_up_zp(pd()))
        kernel_ctx.define_int(
                "WTS_UP_ZP_GROUP_SIZE", wts_up_zp_group_size(pd()));

    if (with_wts_down_scales(pd()))
        kernel_ctx.define_int(
                "WTS_DOWN_SCALES_GROUP_SIZE", wts_down_scales_group_size(pd()));
    if (with_wts_down_zp(pd()))
        kernel_ctx.define_int(
                "WTS_DOWN_ZP_GROUP_SIZE", wts_down_zp_group_size(pd()));

#ifdef UGEMM_UP_ONLY
    kernel_ctx.define_int("UGEMM_UP_ONLY", 1);
#endif

    kernel_ctx.define_int(
            "WITH_SLM", pd()->gemm_gate_up_pkg().getSetting("slm_size") > 0);
    kernel_ctx.define_int("SUBGROUP_SIZE", sg_size(engine));

    int tile_wgu_m = pd()->gemm_gate_up_pkg().getSetting("wg_tile_m");
    int tile_wgu_n = pd()->gemm_gate_up_pkg().getSetting("wg_tile_n");

    kernel_ctx.define_int("REMAINDER_SRC", pd()->MB() % tile_wgu_n);
    if (lds % 4 == 0) kernel_ctx.define_int("BLOCK_SRC", 1);
    if (lda % 4 == 0 && (pd()->OC() % tile_wgu_m) == 0)
        kernel_ctx.define_int("BLOCK_DST", 1);

    gemmstone::microkernel::ShimOptions shimOptions;
    shimOptions.subgroupSize = sg_size(engine);
    shimOptions.useTileOps = true;
    shimOptions.decorator = "wgu";

    auto header = generateShim(pd()->gemm_gate_up_pkg(),
            gemmstone::microkernel::HostLanguage::OpenCL_C, shimOptions);
    kernel_ctx.add_custom_header("gemm_gateup.h", std::move(header));

    if (pd()->gemm_gate_up_pkg().grfMin > 128) {
        kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");
    }

    CHECK(create_kernel(
            engine, &gemm_gate_up_, "micro_gated_mlp_horz", kernel_ctx));
    if (!gemm_gate_up_) return status::runtime_error;
#ifndef UGEMM_UP_ONLY
    CHECK(create_nested_primitive(gemm_down_, pd()->gemm_down_pd_, engine));
#endif
    return status::success;
}

status_t micro_horz_t::execute(const exec_ctx_t &ctx) const {
    auto *engine = ctx.stream()->engine();
    const auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &W_gate = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_GATE);
    const auto &W_up = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_UP);
    const auto &W_down = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_DOWN);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &src_scales
            = CTX_IN_STORAGE(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES);
    const auto &wts_gate_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_SCALES);
    const auto &wts_up_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_SCALES);
    const auto &wts_down_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_SCALES);

    const auto &src_zp
            = CTX_IN_STORAGE(DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wts_gate_zp
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wts_up_zp
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wts_down_zp
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_ZERO_POINTS);

    const dim_t MB = pd()->MB();
    const dim_t IC = pd()->IC();
    const dim_t OC = pd()->OC();

    auto &gemm_gate_up_pkg = pd()->gemm_gate_up_pkg();

    auto wg_tile_m = gemm_gate_up_pkg.getSetting("wg_tile_m");
    auto wg_tile_n = gemm_gate_up_pkg.getSetting("wg_tile_n");
    auto sg_per_wg = gemm_gate_up_pkg.getSetting("sg_per_wg_m")
            * gemm_gate_up_pkg.getSetting("sg_per_wg_n");

    auto inter_src_stor = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_src_trans);

    compute::kernel_arg_list_t arg_list;
    int iter = 0;
    arg_list.set(iter++, src);
    arg_list.set(iter++, W_gate);
    arg_list.set(iter++, W_up);
    arg_list.set(iter++, W_down);
    arg_list.set(iter++, dst);
    arg_list.set(iter++, MB);
    arg_list.set(iter++, IC);
    arg_list.set(iter++, OC);
#ifdef UGEMM_UP_ONLY
    arg_list.set(iter++, dst);
#else
    arg_list.set(iter++, *inter_src_stor);
#endif
    arg_list.set(iter++, src_scales);
    arg_list.set(iter++, src_zp);
    arg_list.set(iter++, wts_gate_scales);
    arg_list.set(iter++, wts_gate_zp);
    arg_list.set(iter++, wts_up_scales);
    arg_list.set(iter++, wts_up_zp);
    arg_list.set(iter++, wts_down_scales);
    arg_list.set(iter++, wts_down_zp);

    compute::range_t lws = {(size_t)sg_size(engine), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(MB, wg_tile_n);
    gws[2] *= utils::div_up(OC, wg_tile_m);

    auto nd_range = compute::nd_range_t(gws, lws);
    CHECK(parallel_for(ctx, nd_range, gemm_gate_up_, arg_list));

#ifndef UGEMM_UP_ONLY
    memory_desc_t inter_md;
    CHECK(pd()->get_gate_dst_md(inter_md));

    std::unique_ptr<memory_t, memory_deleter_t> inter_src_mem;
    CHECK(safe_ptr_assign(inter_src_mem,
            new memory_t(engine, &inter_md, std::move(inter_src_stor))));
    exec_args_t down_args;
    down_args[DNNL_ARG_SRC] = memory_arg_t {inter_src_mem.get(), true};
    down_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_WEIGHTS_DOWN);
    down_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS_DOWN))
        down_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS]
                = ctx.args().at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN);
    if (!pd()->attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS_DOWN))
        down_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = ctx.args().at(
                DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN);

    const auto &post_ops = pd()->attr()->post_ops_;
    for (int p = 0, pl = post_ops.len(); p < pl; p++) {
        if (!post_ops.entry_[p].is_like_binary()) continue;
        auto idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(p) | DNNL_ARG_SRC_1;
        down_args[idx] = ctx.args().at(idx);
    }
    exec_ctx_t down_ctx(ctx, std::move(down_args));
    auto *down_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested_multiple + DNNL_ARG_WEIGHTS_DOWN,
            gemm_down_->pd()->scratchpad_registry());
    down_ctx.set_scratchpad_grantor(down_grantor);
    CHECK(gemm_down_->execute(down_ctx));
#endif
    return status::success;
}

} // namespace gated_mlp
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
