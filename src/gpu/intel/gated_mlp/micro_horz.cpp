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

bool with_wts_quant(const quant_entries_t &q, int arg) {
    return !q.has_default_values(arg);
}
bool with_wts_gate_scales(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->scales_, DNNL_ARG_WEIGHTS_GATE);
}
bool with_wts_up_scales(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->scales_, DNNL_ARG_WEIGHTS_UP);
}
bool with_wts_down_scales(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->scales_, DNNL_ARG_WEIGHTS_DOWN);
}
bool with_wts_gate_zp(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_GATE);
}
bool with_wts_up_zp(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_UP);
}
bool with_wts_down_zp(const micro_horz_t::pd_t *pd) {
    return with_wts_quant(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_DOWN);
}

data_type_t wts_quant_dt(const quant_entries_t &q, int arg) {
    return q.get_data_type(arg);
}
data_type_t wts_gate_scales_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->scales_, DNNL_ARG_WEIGHTS_GATE);
}
data_type_t wts_up_scales_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->scales_, DNNL_ARG_WEIGHTS_UP);
}
data_type_t wts_down_scales_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->scales_, DNNL_ARG_WEIGHTS_DOWN);
}
data_type_t wts_gate_zp_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_GATE);
}
data_type_t wts_up_zp_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_UP);
}
data_type_t wts_down_zp_dt(const micro_horz_t::pd_t *pd) {
    return wts_quant_dt(pd->attr()->zero_points_, DNNL_ARG_WEIGHTS_DOWN);
}

dim_t wts_group_size(const micro_horz_t::pd_t *pd, int arg) {
    auto gs = pd->attr()->scales_.get_group(arg, 0);
    auto gz = pd->attr()->zero_points_.get_group(arg, 0);
    gpu_assert(IMPLICATION(gs && gz, gs == gz));
    return gs;
}
dim_t wts_gate_group_size(const micro_horz_t::pd_t *pd) {
    return wts_group_size(pd, DNNL_ARG_WEIGHTS_GATE);
}
dim_t wts_up_group_size(const micro_horz_t::pd_t *pd) {
    return wts_group_size(pd, DNNL_ARG_WEIGHTS_UP);
}
dim_t wts_down_group_size(const micro_horz_t::pd_t *pd) {
    return wts_group_size(pd, DNNL_ARG_WEIGHTS_DOWN);
}

bool with_quantize_common(const quant_entries_t &q, int arg) {
    return !q.has_default_values(arg) && (q.get_mask(arg) == 0);
}

int sg_size(impl::engine_t *engine) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    return intel_engine->device_info()->min_subgroup_size();
}

} // anonymous namespace

status_t micro_horz_t::pd_t::init(impl::engine_t *engine) {
    //VDISPATCH_GATED_MLP(gpu_utils::dev_getenv("gmlp_horz_ukern", false),
    //        VERBOSE_SKIP_PRIMITIVE_IMPL);
    memory_desc_t inter_md;
    CHECK(get_gate_dst_md(inter_md));

#ifdef UGEMM_UP_ONLY
    CHECK(init_microkernels(engine, &inter_md));
#else
    VDISPATCH_GATED_MLP(pd_ok(), VERBOSE_INCONSISTENT_PRB);
    VDISPATCH_GATED_MLP_SC(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    CHECK(init_microkernels(engine, &inter_md));

    primitive_attr_t down_attr;
    CHECK(move_attr(down_attr, DNNL_ARG_WEIGHTS_DOWN, DNNL_ARG_WEIGHTS));
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

gemmstone::Type get_ab_type(gemmstone::Type src, gemmstone::Type wei) {
    using ty = gemmstone::Type;
    if (src == ty::f32) return ty::f32;
    if (src == ty::bf16)
        return (utils::one_of(wei, ty::u4, ty::s4, ty::u8, ty::s8, ty::bf16))
                ? ty::bf16
                : ty::invalid;
    if (src == ty::f16)
        return (utils::one_of(wei, ty::u4, ty::s4, ty::u8, ty::s8, ty::f16))
                ? ty::f16
                : ty::invalid;
    return ty::invalid;
}

status_t micro_horz_t::pd_t::init_microkernels(
        impl::engine_t *engine, const memory_desc_t *inter_md) {
    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();

    VCONDCHECK(primitive, create, check, gated_mlp,
            compute::mayiuse_microkernels(intel_engine), status::unimplemented,
            "Microkernels not supported by the OpenCL driver.");

    gated_mlp_config_t config = {16, 16, 2, 2};

    gemmstone::microkernel::HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = intel_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    if (hw_info.gmdid == 0) return status::unimplemented;

    gemmstone::GEMMProblem problem;
    problem.Ta_ext = gemm::jit::convert_dnnl_to_kernel_type(
            arg_md(DNNL_ARG_WEIGHTS_GATE)->data_type);
    problem.Tb_ext = gemm::jit::convert_dnnl_to_kernel_type(
            arg_md(DNNL_ARG_SRC)->data_type);
    problem.Tc_ext
            = gemm::jit::convert_dnnl_to_kernel_type(inter_md->data_type);
    problem.Ta = problem.Tb = get_ab_type(problem.Tb_ext, problem.Ta_ext);
    problem.Tc = gemmstone::Type::f32;
    problem.Ts = problem.Tc;

    VCONDCHECK(primitive, create, check, gated_mlp,
            (problem.Ta != gemmstone::Type::invalid)
                    && (problem.Tb != gemmstone::Type::invalid),
            status::unimplemented, "Incompatible A/B types in uGEMM.");

    auto problem_wgu = problem;
    problem_wgu.A.layout = gemmstone::MatrixLayout::T;
    problem_wgu.B.layout = gemmstone::MatrixLayout::Pr;
    problem_wgu.C.layout = gemmstone::MatrixLayout::T;

    const memory_desc_wrapper W_gate_mdw(arg_md(DNNL_ARG_WEIGHTS_GATE));
    const memory_desc_wrapper W_up_mdw(arg_md(DNNL_ARG_WEIGHTS_UP));
    auto alignment = [](const memory_desc_wrapper &mdw) {
        return int(gemm_desc_t::get_ld(*mdw.md_) * mdw.data_type_size());
    };
    problem_wgu.A.setAlignment(gemmstone::microkernel::alignmentForLD(
            std::min(alignment(W_gate_mdw), alignment(W_up_mdw))));
    problem_wgu.B.setAlignment(64);
    problem_wgu.B.crosspack = 2;

    problem_wgu.B.tileR = uint16_t(config.unroll_m_gwu * config.wg_m_gwu);
    problem_wgu.B.tileC = uint16_t(sg_size(engine));

    bool wgu_common_scales
            = with_quantize_common(attr()->scales_, DNNL_ARG_WEIGHTS_GATE);
    bool wgu_common_zp
            = with_quantize_common(attr()->zero_points_, DNNL_ARG_WEIGHTS_GATE);

    if (with_wts_gate_scales(this) && !wgu_common_scales) {
        auto scale_dt = wts_gate_scales_dt(this);
        problem_wgu.Ta_scale = gemm::jit::convert_dnnl_to_kernel_type(scale_dt);
        problem_wgu.A_scale.alignment
                = uint8_t(types::data_type_size(scale_dt));
        problem_wgu.A_scale.layout = gemmstone::MatrixLayout::N;
        problem_wgu.asPtrDims = 2;
    }
    if (with_wts_gate_zp(this)) {
        auto zp_dt = wts_gate_zp_dt(this);
        problem_wgu.Tao = gemm::jit::convert_dnnl_to_kernel_type(zp_dt);
        problem_wgu.AO.alignment = uint8_t(types::data_type_size(zp_dt));
        problem_wgu.AO.layout = gemmstone::MatrixLayout::N;
        problem_wgu.aoPtrDims = (wgu_common_zp) ? 0 : 2;
        problem_wgu.aOffset = gemmstone::ABOffset::Calc;
    }

    if (with_wts_gate_scales(this) || with_wts_gate_zp(this)) {
        problem_wgu.aqGroupM = problem_wgu.aqGroupK = 1;
        if (!wgu_common_scales && !wgu_common_zp)
            problem_wgu.aqGroupK = int(wts_gate_group_size(this));
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
    opts_wgu.localB = true;
    opts_wgu.slmPtr = true;

    opts_wgu.scaleA = with_wts_gate_scales(this) && !wgu_common_scales;
    opts_wgu.offsetA = with_wts_gate_zp(this);

    try {
        gemm_gate_up_pkg_
                = selectGEMM(opts_wgu, hw_info, sizes, problem_wgu, reqs_wgu);
    } catch (std::exception &e) {
        VDISPATCH_GATED_MLP(false,
                "gemm_gateup microkernel generation failed with message: %s",
                e.what());
    }
    return status::success;
}

status_t micro_horz_t::init(impl::engine_t *engine) {
    compute::kernel_ctx_t kernel_ctx;

    int ndims = 2;
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

    def_offsets(inter_off, kernel_ctx, "INTER", ndims);
    def_offsets(src_off, kernel_ctx, "SRC", ndims);
    def_offsets(W_gate_off, kernel_ctx, "W_GATE", ndims);
    def_offsets(W_up_off, kernel_ctx, "W_UP", ndims);
    def_offsets(W_down_off, kernel_ctx, "W_DOWN", ndims);
    def_offsets(dst_off, kernel_ctx, "DST", ndims);
    kernel_ctx.define_int("NDIMS", ndims);

    def_data_type(kernel_ctx, inter_mdw.data_type(), "INTER");
    def_data_type(kernel_ctx, src_mdw.data_type(), "SRC");
    def_data_type(kernel_ctx, W_gate_mdw.data_type(), "WTS_GATE");
    def_data_type(kernel_ctx, W_up_mdw.data_type(), "WTS_UP");
    def_data_type(kernel_ctx, W_down_mdw.data_type(), "WTS_DOWN");
    def_data_type(kernel_ctx, dst_mdw.data_type(), "DST");

    def_data_type(kernel_ctx, wts_gate_scales_dt(pd()), "WTS_GATE_ATTR_SCALES");
    def_data_type(kernel_ctx, wts_up_scales_dt(pd()), "WTS_UP_ATTR_SCALES");
    def_data_type(kernel_ctx, wts_down_scales_dt(pd()), "WTS_DOWN_ATTR_SCALES");

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

    auto attr = pd()->attr();

    int wts_gate_scales_mask = (int(with_wts_gate_scales(pd())) << 1)
            | int(with_quantize_common(attr->scales_, DNNL_ARG_WEIGHTS_GATE));
    int wts_up_scales_mask = (int(with_wts_up_scales(pd())) << 1)
            | int(with_quantize_common(attr->scales_, DNNL_ARG_WEIGHTS_UP));
    int wts_down_scales_mask = (int(with_wts_down_scales(pd())) << 1)
            | int(with_quantize_common(attr->scales_, DNNL_ARG_WEIGHTS_DOWN));

    kernel_ctx.define_int("WTS_GATE_SCALES", wts_gate_scales_mask);
    kernel_ctx.define_int("WTS_UP_SCALES", wts_up_scales_mask);
    kernel_ctx.define_int("WTS_DOWN_SCALES", wts_down_scales_mask);

    int wts_gate_zp_mask = (int(with_wts_gate_zp(pd())) << 1)
            | int(with_quantize_common(
                    attr->zero_points_, DNNL_ARG_WEIGHTS_GATE));
    int wts_up_zp_mask = (int(with_wts_up_zp(pd())) << 1)
            | int(with_quantize_common(
                    attr->zero_points_, DNNL_ARG_WEIGHTS_UP));
    int wts_down_zp_mask = (int(with_wts_down_zp(pd())) << 1)
            | int(with_quantize_common(
                    attr->zero_points_, DNNL_ARG_WEIGHTS_DOWN));

    kernel_ctx.define_int("WTS_GATE_ZERO_POINTS", wts_gate_zp_mask);
    kernel_ctx.define_int("WTS_UP_ZERO_POINTS", wts_up_zp_mask);
    kernel_ctx.define_int("WTS_DOWN_ZERO_POINTS", wts_down_zp_mask);

    using namespace data_type;
    auto elems_per_byte = [](data_type_t dt) {
        switch (dt) {
            case u4:
            case s4: return 2;
            default: return 1;
        }
    };
    kernel_ctx.define_int("WTS_GATE_ELEMENTS_PER_BYTE",
            elems_per_byte(W_gate_mdw.data_type()));
    kernel_ctx.define_int(
            "WTS_UP_ELEMENTS_PER_BYTE", elems_per_byte(W_up_mdw.data_type()));
    kernel_ctx.define_int("WTS_DOWN_ELEMENTS_PER_BYTE",
            elems_per_byte(W_down_mdw.data_type()));

    kernel_ctx.define_int("WTS_GATE_ZP_ELEMENTS_PER_BYTE",
            elems_per_byte(wts_gate_zp_dt(pd())));
    kernel_ctx.define_int(
            "WTS_UP_ZP_ELEMENTS_PER_BYTE", elems_per_byte(wts_up_zp_dt(pd())));
    kernel_ctx.define_int("WTS_DOWN_ZP_ELEMENTS_PER_BYTE",
            elems_per_byte(wts_down_zp_dt(pd())));

    if (with_wts_gate_scales(pd()) || with_wts_gate_zp(pd()))
        kernel_ctx.define_int("WTS_GATE_GROUP_SIZE", wts_gate_group_size(pd()));
    if (with_wts_up_scales(pd()) || with_wts_up_zp(pd()))
        kernel_ctx.define_int("WTS_UP_GROUP_SIZE", wts_up_group_size(pd()));
    if (with_wts_down_scales(pd()) || with_wts_down_zp(pd()))
        kernel_ctx.define_int("WTS_DOWN_GROUP_SIZE", wts_down_group_size(pd()));

#ifdef UGEMM_UP_ONLY
    kernel_ctx.define_int("UGEMM_UP_ONLY", 1);
#endif

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

    const auto &wts_gate_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_SCALES);
    const auto &wts_up_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_SCALES);
    const auto &wts_down_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_SCALES);

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

    auto wg_tile_OC = gemm_gate_up_pkg.getSetting("wg_tile_m");
    auto wg_tile_MB = gemm_gate_up_pkg.getSetting("wg_tile_n");
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
    arg_list.set(iter++, wts_gate_scales);
    arg_list.set(iter++, wts_gate_zp);
    arg_list.set(iter++, wts_up_scales);
    arg_list.set(iter++, wts_up_zp);
    arg_list.set(iter++, wts_down_scales);
    arg_list.set(iter++, wts_down_zp);

    compute::range_t lws = {(size_t)sg_size(engine), (size_t)sg_per_wg, 1};
    compute::range_t gws = lws;

    gws[0] *= utils::div_up(OC, wg_tile_OC);
    gws[2] *= utils::div_up(MB, wg_tile_MB);

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
