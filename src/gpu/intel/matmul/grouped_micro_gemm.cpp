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

#include "gpu/intel/matmul/grouped_micro_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t grouped_micro_gemm_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using namespace gemmstone;
    using namespace gemmstone::microkernel;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    //arch_ = dev_info->gpu_arch();
    bool use_systolic_ukernel = intel_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = use_systolic_ukernel;

    if (hw_info.gmdid == 0) return status::unimplemented;

    auto src_mdw = memory_desc_wrapper(pd()->src_md(0));
    auto wei_mdw = memory_desc_wrapper(pd()->weights_md());

    int m = static_cast<int>(pd()->M());
    int n = static_cast<int>(pd()->N());
    int k = static_cast<int>(pd()->K());

    auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
        return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
                                                           : MatrixLayout::N;
    };

    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(wei_mdw.data_type());
    problem.Tb_ext = convert_dnnl_to_kernel_type(src_mdw.data_type());
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;

    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;

    problem.A.setAlignment(
            alignmentForLD(static_cast<int>(pd()->K()) * problem.Ta_ext));
    problem.B.setAlignment(
            alignmentForLD(static_cast<int>(pd()->N()) * problem.Tb_ext));
    problem.C.setAlignment(problem.Tc.size());

    problem.A.layout = convert_dnnl_to_kernel_layout(wei_mdw.md_);
    problem.B.layout = MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;

    GEMMOptions opts;
    opts.scaleA = !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
            && pd()->wei_group_sizes_[1] != pd()->K();
    opts.offsetA
            = !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS);
    opts.scaleB = !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)
            && pd()->src_group_sizes_[1] != pd()->K();
    opts.offsetB = !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_SRC);
    opts.slmPtr = true;
    opts.kParallelLocal = true;

    if (opts.scaleA) {
        auto wei_scales = pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS);
        data_type_t wei_scale_dt = wei_scales.get_data_type();
        problem.Ta_scale = convert_dnnl_to_kernel_type(wei_scale_dt);
        problem.A_scale.setAlignment(alignmentForLD(
                static_cast<int>(types::data_type_size(wei_scale_dt))));
        problem.A_scale.layout = MatrixLayout::N;
        problem.asPtrDims = 2;
    }

    if (opts.offsetA) {
        auto wei_zp = pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS);
        data_type_t wei_zp_dt = wei_zp.get_data_type();
        problem.Tao = convert_dnnl_to_kernel_type(wei_zp_dt);
        problem.AO.setAlignment(
                static_cast<int>(types::data_type_size(wei_zp_dt)));
        problem.AO.layout = MatrixLayout::N;
        problem.aoPtrDims = 2;
        problem.aOffset = ABOffset::Calc;
    }

    if (opts.scaleB) {
        auto src_scales = pd()->attr()->scales_.get(DNNL_ARG_SRC);
        data_type_t src_scale_dt = src_scales.get_data_type();
        problem.Tb_scale = convert_dnnl_to_kernel_type(src_scale_dt);
        problem.B_scale.setAlignment(
                static_cast<int>(types::data_type_size(src_scale_dt)));
        problem.B_scale.layout = MatrixLayout::N;
        problem.bsPtrDims = 2;
    }
    if (opts.offsetB) {
        auto src_zp = pd()->attr()->zero_points_.get(DNNL_ARG_SRC);
        data_type_t src_zp_dt = src_zp.get_data_type();
        problem.Tbo = convert_dnnl_to_kernel_type(src_zp_dt);
        problem.BO.setAlignment(
                static_cast<int>(types::data_type_size(src_zp_dt)));
        problem.BO.layout = MatrixLayout::N;
        problem.boPtrDims = 2;
        problem.bOffset = ABOffset::Calc;
    }

    if (opts.scaleA || opts.offsetA) {
        problem.aqGroupM = pd()->wei_group_sizes_[2];
        problem.aqGroupK = utils::rnd_up_pow2(pd()->wei_group_sizes_[1]);
    }

    if (opts.scaleB || opts.offsetB) {
        problem.bqGroupN = pd()->src_group_sizes_[0];
        problem.bqGroupK = static_cast<int>(
                utils::rnd_up_pow2(pd()->src_group_sizes_[1]));
    }

    SizeParams sizes;
    sizes.m = static_cast<uint16_t>(n);
    sizes.n = static_cast<uint16_t>(m);
    sizes.k = static_cast<uint16_t>(k);

    auto sg_size = dev_info->min_subgroup_size();
    try {
        gemm = selectGEMM(opts, hw_info, sizes, problem);
    } catch (const std::runtime_error &ex) {
        std::vector<StrategyRequirement> reqs;
        reqs.push_back(StrategyRequirement::UnrollM == sg_size);
        reqs.push_back(StrategyRequirement::UnrollN
                == utils::rnd_up_pow2(std::min<dim_t>(pd()->M(), 64)));
        reqs.push_back(StrategyRequirement::WGM == 2);
        reqs.push_back(StrategyRequirement::WGN
                == utils::rnd_up_pow2(std::max<dim_t>(
                        1, std::min<dim_t>(pd()->M() / reqs[1].value, 8))));
        try {
            gemm = selectGEMM(opts, hw_info, sizes, problem, reqs);
        } catch (const std::runtime_error &ex) {
            //CHECK_BOOL(false,
            //         "gemm microkernel generation failure with message: %s",
            //         ex.what());
            return status::unimplemented;
        }
    }

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = sg_size;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "grouped";

    kernel_ctx_.define_int("SUBGROUP_SIZE", sg_size);
    kernel_ctx_.add_custom_header("gemm_grouped.h",
            generateShim(gemm, HostLanguage::OpenCL_C, shimOptions));

    return status::success;
}

template <size_t N>
void calc_group_sizes(std::array<int, N> &dims, const quant_entry_t &entry,
        const memory_desc_wrapper &desc) {
    memory_desc_t md;
    entry.get_md(md, *desc.md_);
    std::transform(desc.dims(), desc.dims() + dims.size(), md.dims, begin(dims),
            [](dim_t d, dim_t d2) -> int {
        return static_cast<int>(d2 == 0 ? 1 : d / d2);
    });
}

status_t grouped_micro_gemm_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    data_type_t src_dt = src_d.data_type();
    data_type_t wei_dt = wei_d.data_type();
    data_type_t dst_dt = dst_d.data_type();

    // Check for grouped encoding on src and dst
    VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Weights should be dense
    VDISPATCH_MATMUL(!wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Extract grouped encoding
    const sparse_desc_t::grouped_desc_t &src_grouped
            = src_d.sparse_desc().grouped_desc;
    const sparse_desc_t::grouped_desc_t &dst_grouped
            = dst_d.sparse_desc().grouped_desc;

    VDISPATCH_MATMUL(wei_d.matches_one_of_tag(format_tag::ab, format_tag::ba,
                             format_tag::abc, format_tag::acb),
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    // Validate matching number of groups
    VDISPATCH_MATMUL(src_grouped.group_count == dst_grouped.group_count,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src ngroups", "dst ngroups",
            (int)src_grouped.group_count, (int)dst_grouped.group_count);

    ngroups_ = src_grouped.group_count;
    // only supported dt for now
    VDISPATCH_MATMUL(utils::one_of(src_dt, f32, f16, bf16, u8, s8, s4, u4),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(wei_dt, f32, f16, bf16, u8, s8, s4, u4),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    const bool src_subbyte = utils::one_of(src_dt, s4, u4);
    const bool wei_subbyte = utils::one_of(wei_dt, s4, u4);
    VDISPATCH_MATMUL(IMPLICATION(src_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "src", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (K() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 1);
    VDISPATCH_MATMUL(IMPLICATION(wei_subbyte, (N() % 2) == 0), VERBOSE_BAD_DIM,
            "weights", 2);

    // Check offsets are int32
    VDISPATCH_MATMUL(utils::everyone_is(s32, src_d.metadata_type(0),
                             dst_d.metadata_type(0)),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Check for limited Bias support
    if (with_bias()) {
        memory_desc_wrapper bia_d(weights_md(1));
        VDISPATCH_MATMUL(!bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
        VDISPATCH_MATMUL(bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
        // Bias shape should be [num_experts, N]
        VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.group_count,
                VERBOSE_INCONSISTENT_DIM, "bia_d", 0, "src_grouped.group_count",
                -1);
        VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                VERBOSE_INCONSISTENT_DIM, "bia_d", 1, "wei_d", 2);
    }

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");

    // Check for supported quantization schemes
    const scales_t &attr_scales = attr()->scales_;
    if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
        const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
        const int rowwise_mask = src_qmask_M();
        // Only row-wise f32 scales supported for src
        VDISPATCH_MATMUL(
                src_mask == rowwise_mask, VERBOSE_UNSUPPORTED_SCALES_CFG);
        // No groups for src scales
        VDISPATCH_MATMUL(attr_scales.get(DNNL_ARG_SRC).has_default_groups(),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }
    if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
        const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
        // Only column-wise f32 scales supported for weights
        VDISPATCH_MATMUL(
                utils::one_of(wei_mask, 7, 5), VERBOSE_UNSUPPORTED_SCALES_CFG);
    }
    VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // No post-ops for now
    VDISPATCH_MATMUL(
            attr()->post_ops_.has_default_values(), VERBOSE_UNSUPPORTED_POSTOP);

    if (!attr()->scales_.has_default_values(DNNL_ARG_SRC)) {
        calc_group_sizes(
                src_group_sizes_, attr()->scales_.get(DNNL_ARG_SRC), src_d);
    } else if (!attr()->zero_points_.has_default_values(DNNL_ARG_SRC)) {
        calc_group_sizes(src_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_SRC), src_d);
    }
    if (!attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        calc_group_sizes(
                wei_group_sizes_, attr()->scales_.get(DNNL_ARG_WEIGHTS), wei_d);
    } else if (!attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)) {
        calc_group_sizes(wei_group_sizes_,
                attr()->zero_points_.get(DNNL_ARG_WEIGHTS), wei_d);
    }
    sg_size_ = dev_info->min_subgroup_size();

    return status::success;
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {

    CHECK(init_microkernels(engine));
    auto src_dt = pd()->src_md(0)->data_type;
    auto wei_dt = pd()->weights_md(0)->data_type;
    auto dst_dt = pd()->dst_md(0)->data_type;

    kernel_ctx_.set_data_type(dst_dt);

    if (gemm.grfMin > 128)
        kernel_ctx_.add_option("-cl-intel-256-GRF-per-thread");

    def_data_type(kernel_ctx_, src_dt, "SRC");
    def_data_type(kernel_ctx_, wei_dt, "WEI");
    def_data_type(kernel_ctx_, dst_dt, "DST");

    kernel_ctx_.define_int("WITH_SRC_SCALES",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC));
    kernel_ctx_.define_int("WITH_WEI_SCALES",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS));
    kernel_ctx_.define_int("WITH_SRC_ZP",
            !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_SRC));
    kernel_ctx_.define_int("WITH_WEI_ZP",
            !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS));
    def_data_type(kernel_ctx_,
            pd()->attr()->scales_.get(DNNL_ARG_SRC).get_data_type(),
            "SRC_SCALES");

    def_data_type(kernel_ctx_,
            pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS).get_data_type(),
            "WEI_SCALES");

    def_data_type(kernel_ctx_,
            pd()->attr()->zero_points_.get(DNNL_ARG_SRC).get_data_type(),
            "SRC_ZP");

    def_data_type(kernel_ctx_,
            pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS).get_data_type(),
            "WEI_ZP");
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)
            || !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_SRC)) {
        kernel_ctx_.define_int("SRC_GROUP_SIZE", pd()->src_group_sizes_[1]);
    }
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
            || !pd()->attr()->zero_points_.has_default_values(
                    DNNL_ARG_WEIGHTS)) {
        kernel_ctx_.define_int("WEI_GROUP_SIZE", pd()->wei_group_sizes_[1]);
    }

    kernel_ctx_.define_int("SRC_SCALES_GROUPED",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)
                    && pd()->src_group_sizes_[1] != pd()->K());
    kernel_ctx_.define_int("WEI_SCALES_GROUPED",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
                    && pd()->wei_group_sizes_[1] != pd()->K());
    kernel_ctx_.define_int(
            "SRC_ELEMS_PER_BYTE", types::bytes_to_elements(src_dt, 1));
    kernel_ctx_.define_int(
            "WEI_ELEMS_PER_BYTE", types::bytes_to_elements(wei_dt, 1));

    auto bia_dt = pd()->weights_md(1)->data_type;
    def_data_type(kernel_ctx_, bia_dt, "BIA");
    kernel_ctx_.define_int("WITH_BIAS", pd()->with_bias());

    return create_kernel(engine, &kernel_, "grouped_micro_gemm", kernel_ctx_);
}

status_t grouped_micro_gemm_t::execute(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto &src_scales
            = CTX_IN_STORAGE(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES);
    const auto &src_zero_points
            = CTX_IN_STORAGE(DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS);
    const auto &wei_scales
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES);
    const auto &wei_zero_points
            = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS);

    const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const memory_desc_t *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const memory_desc_t *wei_md = pd()->weights_md();
    const memory_desc_t *dst_md = ctx.output(DNNL_ARG_DST)->md();

    const size_t num_groups = pd()->ngroups_;

    const quant_entries_t &attr_scales = pd()->attr()->scales_;
    const quant_entries_t &attr_zero_points = pd()->attr()->zero_points_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_src_zero_points
            = !attr_zero_points.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_wei_zero_points
            = !attr_zero_points.has_default_values(DNNL_ARG_WEIGHTS);

    int ldsrcq = 0;
    int ldweiq = 0;

    if (with_src_scales || with_src_zero_points) {
        // Only row-wise cales are supported for src
        ldsrcq = 1;
    }
    if (with_wei_scales || with_wei_zero_points) {
        const memory_desc_t *wei_quant_md = with_wei_scales
                ? ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES)->md()
                : ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_ZERO_POINTS)->md();
        ldweiq = static_cast<int>(wei_quant_md->format_desc.blocking
                                          .strides[wei_quant_md->ndims - 2]);
    }
    int m_all = static_cast<int>(dst_md->dims[dst_md->ndims - 2]);
    int n = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    int k = static_cast<int>(src_md->dims[src_md->ndims - 1]);

    int ldsrc = static_cast<int>(src_md->dims[src_md->ndims - 1]);
    int lddst = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    const dims_t &wei_strides_ = wei_md->format_desc.blocking.strides;
    compute::int64x4_t wei_strides
            = {static_cast<int64_t>(wei_strides_[wei_md->ndims - 3]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 2]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 1]),
                    static_cast<int64_t>(wei_strides_[wei_md->ndims - 0])};

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(wei_data);
    arg_list.append(wei_strides);
    arg_list.append(dst_data);
    arg_list.append(lddst);
    arg_list.append(src_offsets);
    arg_list.append(dst_offsets);
    arg_list.append(src_scales);
    arg_list.append(src_zero_points);
    arg_list.append(ldsrcq);
    arg_list.append(wei_scales);
    arg_list.append(wei_zero_points);
    arg_list.append(ldweiq);
    arg_list.append(n);
    arg_list.append(k);

    arg_list.append(bias_data);

    size_t sg_per_wg_m = gemm.getSetting("sg_per_wg_m");
    size_t sg_per_wg_n = gemm.getSetting("sg_per_wg_n");
    size_t sg_per_wg_k = gemm.getSetting("sg_per_wg_k");
    size_t wg_tile_m = gemm.getSetting("wg_tile_m");
    size_t wg_tile_n = gemm.getSetting("wg_tile_n");

    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = compute::range_t::one(3);
    lws[0] *= pd()->sg_size_;

    lws[0] *= sg_per_wg_m;
    lws[1] *= sg_per_wg_n;
    lws[2] *= sg_per_wg_k;

    compute::range_t gws = lws;
    // Swap wg_tile_[mn]_ for col-major vs row-major representations
    gws[0] *= utils::div_up(n, wg_tile_m);
    gws[1] *= utils::div_up(m_all, wg_tile_n);
    gws[2] *= num_groups;
    //std::cout << "LWS: " << lws.str() << std::endl;
    //std::cout << "GWS: " << gws.str() << std::endl;

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
