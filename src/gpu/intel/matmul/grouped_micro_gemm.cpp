/*******************************************************************************
* Copyright 2025 Intel Corporation
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

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = pd()->use_systolic_ukernel_;

    if (hw_info.gmdid == 0) return status::unimplemented;

    auto src_mdw = memory_desc_wrapper(pd()->src_md(0));
    auto wei_mdw = memory_desc_wrapper(pd()->weights_md());
    auto dst_mdw = memory_desc_wrapper(pd()->dst_md(0));

    int m = std::max(128, static_cast<int>(pd()->M()));
    int n = static_cast<int>(pd()->N());
    int k = static_cast<int>(pd()->K());

    //auto convert_dnnl_to_kernel_layout = [](const memory_desc_t *md) {
    //    return (gemm_desc_t::get_trans(*md) == dnnl_trans) ? MatrixLayout::T
    //                                                       : MatrixLayout::N;
    //};

    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(src_mdw.data_type());
    problem.Tb_ext = convert_dnnl_to_kernel_type(wei_mdw.data_type());
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;
    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;
    problem.A.layout = MatrixLayout::T;
    problem.B.layout = MatrixLayout::T;
    problem.C.layout = MatrixLayout::T;
    problem.A.setAlignment(alignmentForLD(k * problem.Ta_ext.bits() / 8));
    problem.B.setAlignment(alignmentForLD(n * problem.Tb_ext.bits() / 8));
    problem.C.setAlignment(problem.Tc.size());

    auto quantizedType = [](data_type_t dt, data_type_t ddt) {
        switch (dt) {
            case dnnl_bf16: return Type::bf16;
            case dnnl_f16: return Type::f16;
            case dnnl_f32: return Type::f32;
            case dnnl_u8:
            case dnnl_s8:
            case dnnl_u4:
            case dnnl_s4:
                if (ddt == dnnl_bf16)
                    return Type::bf16;
                else
                    return Type::f16;
            default: return Type::invalid;
        }
    };

    GEMMOptions opts;
    opts.slmPtr = true;
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)) {
        opts.scaleA = true;
        problem.Ta = quantizedType(src_mdw.data_type(), dst_mdw.data_type());
        auto src_scales = pd()->attr()->scales_.get(DNNL_ARG_SRC);
        data_type_t src_scale_dt = src_scales.get_data_type();
        problem.Ta_scale = convert_dnnl_to_kernel_type(src_scale_dt);
        problem.A_scale.setAlignment(
                int8_t(types::data_type_size(src_scale_dt)));
        problem.A_scale.layout = MatrixLayout::N;
        problem.asPtrDims = 2;
    }
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        opts.scaleB = true;
        problem.Tb = quantizedType(wei_mdw.data_type(), dst_mdw.data_type());
        auto wei_scales = pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS);
        data_type_t wei_scale_dt = wei_scales.get_data_type();
        problem.Tb_scale = convert_dnnl_to_kernel_type(wei_scale_dt);
        problem.B_scale.setAlignment(pd()->N()
                / pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS).get_group_size()
                * int8_t(types::data_type_size(wei_scale_dt)));
        problem.B_scale.layout = MatrixLayout::T;
        problem.bsPtrDims = 2;
    }

    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)) {
        auto &src_scales = pd()->attr()->scales_.get(DNNL_ARG_SRC);
        memory_desc_t md;
        const memory_desc_t &src_md = *pd()->src_md();
        src_scales.get_md(md, src_md);
        problem.aqGroupM = pd()->src_group_sizes_[0];
        problem.aqGroupK = pd()->src_group_sizes_[1];
    }
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        auto &wei_scales = pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS);
        memory_desc_t md;
        const memory_desc_t &wei_md = *pd()->weights_md();
        wei_scales.get_md(md, wei_md);
        problem.bqGroupK = pd()->wei_group_sizes_[1];
        problem.bqGroupN = pd()->wei_group_sizes_[2];
    }

    SizeParams sizes;
    sizes.m = static_cast<uint16_t>(m);
    sizes.n = static_cast<uint16_t>(n);
    sizes.k = static_cast<uint16_t>(k);

    Package gemm;

    //std::vector<StrategyRequirement> reqs;
    //reqs.push_back(StrategyRequirement::UnrollM == 32);
    //reqs.push_back(StrategyRequirement::UnrollN == 16);
    //reqs.push_back(StrategyRequirement::WGM == 2);
    //reqs.push_back(StrategyRequirement::WGN == 1);

    try {
        std::vector<StrategyRequirement> reqs;
        gemm = microkernel::selectGEMM(opts, hw_info, sizes, problem);
    } catch (const std::runtime_error &ex) {
        std::vector<StrategyRequirement> reqs;
        reqs.push_back(StrategyRequirement::UnrollM == 32);
        reqs.push_back(StrategyRequirement::UnrollN == 32);
        reqs.push_back(StrategyRequirement::WGM == 2);
        reqs.push_back(StrategyRequirement::WGN == 2);
        try {
            gemm = selectGEMM(opts, hw_info, sizes, problem, reqs);
        } catch (const std::runtime_error &ex) {
            //CHECK_BOOL(false,
            //         "gemm microkernel generation failure with message: %s",
            //         ex.what());
            return status::unimplemented;
        }
    }
    //printf("Gemm: sg_per_wg_m=%d, sg_per_wg_n=%d, unroll_m=%d, unroll_n=%d\n",
    //        gemm.getSetting("sg_per_wg_m"), gemm.getSetting("sg_per_wg_n"),
    //        gemm.getSetting("sg_tile_m"), gemm.getSetting("sg_tile_n"));

    //printf(__FILE__ "(%d)\n", __LINE__);
    auto sg_size = dev_info->min_subgroup_size();

    /* Generate microkernel shims */
    ShimOptions shimOptions;
    shimOptions.subgroupSize = sg_size;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "grouped";

    kernel_ctx_.define_int("SUBGROUP_SIZE", sg_size);
    kernel_ctx_.add_custom_header("gemm_grouped.h",
            generateShim(gemm, HostLanguage::OpenCL_C, shimOptions));

    auto pd_ = (pd_t *)primitive_t::pd().get();
    pd_->sg_per_wg_m_ = gemm.getSetting("sg_per_wg_m");
    pd_->sg_per_wg_n_ = gemm.getSetting("sg_per_wg_n");
    pd_->sg_tile_m_ = gemm.getSetting("sg_tile_m");
    pd_->sg_tile_n_ = gemm.getSetting("sg_tile_n");
    if (gemm.grfMin > 128 || gemm.grfMin > 128) pd_->use_256_grf_ = true;

    return status::success;
}

template <size_t N>
void calc_group_sizes(std::array<dim_t, N> &dims, const quant_entry_t &entry,
        const memory_desc_t &desc) {
    memory_desc_t md;
    entry.get_md(md, desc);
    std::transform(desc.dims, desc.dims + dims.size(), md.dims, begin(dims),
            [](dim_t d, dim_t d2) { return d2 == 0 ? 1 : d / d2; });
}

status_t grouped_micro_gemm_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    auto src_dt = src_md(0)->data_type;
    auto wei_dt = weights_md(0)->data_type;
    auto dst_dt = dst_md(0)->data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());

    calc_group_sizes(
            src_group_sizes_, attr()->scales_.get(DNNL_ARG_SRC), *src_md());
    calc_group_sizes(wei_group_sizes_, attr()->scales_.get(DNNL_ARG_WEIGHTS),
            *weights_md());

    // Check for grouped encoding on src and dst
    VDISPATCH_MATMUL(src_d.is_grouped_desc() && dst_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Weights should be dense
    VDISPATCH_MATMUL(!wei_d.is_sparse_desc() && !wei_d.is_grouped_desc(),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Extract grouped encoding
    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const auto &dst_grouped = dst_d.sparse_desc().grouped_desc;

    // Validate matching number of groups
    VDISPATCH_MATMUL(src_grouped.ngroups == dst_grouped.ngroups,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src ngroups", "dst ngroups",
            (int)src_grouped.ngroups, (int)dst_grouped.ngroups);

    ngroups_ = src_grouped.ngroups;
    // only supported dt for now
    VDISPATCH_MATMUL(utils::one_of(src_dt, f32, f16, bf16, u8, s8, s4),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(wei_dt, f32, f16, bf16, u8, s8, s4),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

    // Check offsets are int32
    VDISPATCH_MATMUL(
            src_d.metadata_type(0) == s32 && dst_d.metadata_type(0) == s32,
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Check for limited Bias support
    if (with_bias()) {
        memory_desc_wrapper bia_d(weights_md(1));
        VDISPATCH_MATMUL(!bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                VERBOSE_UNSUPPORTED_BIAS_CFG);
        VDISPATCH_MATMUL(bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
        // Bias shape should be [num_experts, N]
        VDISPATCH_MATMUL(bia_d.dims()[0] == src_grouped.ngroups,
                VERBOSE_INCONSISTENT_DIM, "bia_d", 0, "src_grouped.ngroups",
                -1);
        VDISPATCH_MATMUL(bia_d.dims()[1] == wei_d.dims()[2],
                VERBOSE_INCONSISTENT_DIM, "bia_d", 1, "wei_d", 2);
    }

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");

    use_systolic_ukernel_ = intel_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);
    sg_size_ = dev_info->min_subgroup_size();

    return status::success;
}

auto elems_per_byte(data_type_t dt) {
    switch (dt) {
        case data_type::u4:
        case data_type::s4: return 2;
        default: return 1;
    }
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {

    CHECK(init_microkernels(engine));
    auto src_dt = pd()->src_md(0)->data_type;
    auto wei_dt = pd()->weights_md(0)->data_type;
    auto dst_dt = pd()->dst_md(0)->data_type;

    kernel_ctx_.set_data_type(dst_dt);

    if (pd()->use_256_grf_)
        kernel_ctx_.add_option("-cl-intel-256-GRF-per-thread");

    def_data_type(kernel_ctx_, src_dt, "SRC");
    def_data_type(kernel_ctx_, wei_dt, "WEI");
    def_data_type(kernel_ctx_, dst_dt, "DST");

    kernel_ctx_.define_int("WITH_SRC_ATTR_SCALES",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC));
    kernel_ctx_.define_int("WITH_WEI_ATTR_SCALES",
            !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS));
    kernel_ctx_.define_int("WITH_SRC_ATTR_ZP",
            !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_SRC));
    kernel_ctx_.define_int("WITH_WEI_ATTR_ZP",
            !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS));
    def_data_type(kernel_ctx_,
            pd()->attr()->scales_.get(DNNL_ARG_SRC).get_data_type(),
            "SRC_ATTR_SCALES");

    def_data_type(kernel_ctx_,
            pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS).get_data_type(),
            "WEI_ATTR_SCALES");

    def_data_type(kernel_ctx_,
            pd()->attr()->zero_points_.get(DNNL_ARG_SRC).get_data_type(),
            "SRC_ATTR_ZP");

    def_data_type(kernel_ctx_,
            pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS).get_data_type(),
            "WEI_ATTR_ZP");

    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC)) {
        kernel_ctx_.define_int(
                "NUM_SRC_ATTR_SCALES", pd()->src_group_sizes_[0]);
    }
    if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        kernel_ctx_.define_int(
                "NUM_WEI_ATTR_SCALES", pd()->K() / pd()->wei_group_sizes_[1]);
    }
    kernel_ctx_.define_int("SRC_ELEMS_PER_BYTE", elems_per_byte(src_dt));
    kernel_ctx_.define_int("WEI_ELEMS_PER_BYTE", elems_per_byte(wei_dt));

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

    auto pd_ = pd();
    const auto *src_md = ctx.input(DNNL_ARG_SRC)->md();
    const auto *wei_md = pd()->weights_md();
    const auto *dst_md = ctx.output(DNNL_ARG_DST)->md();
    const memory_desc_t *src_scales_md = nullptr;
    const memory_desc_t *wei_scales_md = nullptr;

    const size_t num_groups = pd()->ngroups_;

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_bias = pd()->with_bias();

    if (with_src_scales) {
        src_scales_md = ctx.input(DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES)->md();
    }
    if (with_wei_scales) {
        wei_scales_md
                = ctx.input(DNNL_ARG_WEIGHTS | DNNL_ARG_ATTR_SCALES)->md();
    }

    int m_all = static_cast<int>(dst_md->dims[dst_md->ndims - 2]);
    int n = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    int k = static_cast<int>(src_md->dims[src_md->ndims - 1]);
    //printf("m_all=%d, n=%d, k=%d\n", m_all, n, k);

    int ldsrc = static_cast<int>(src_md->dims[src_md->ndims - 1]);
    int ldwei = static_cast<int>(wei_md->dims[wei_md->ndims - 1]);
    int lddst = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    int ldsrcq = static_cast<int>(
            src_scales_md ? src_scales_md->dims[src_scales_md->ndims - 1] : 0);
    int ldweiq = static_cast<int>(
            wei_scales_md ? wei_scales_md->dims[wei_scales_md->ndims - 1] : 0);

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(wei_data);
    arg_list.append(ldwei);
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

    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = {(size_t)pd_->sg_per_wg_m_ * pd_->sg_size_,
            (size_t)pd_->sg_per_wg_n_, 1};
    compute::range_t gws = {utils::div_up(m_all, lws[0]) * lws[0],
            utils::div_up(n, pd_->sg_per_wg_n_ * pd_->sg_tile_n_) * lws[1],
            num_groups};
    //std::cout << "LWS: " << lws.str() << std::endl;
    //std::cout << "GWS: " << gws.str() << std::endl;

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
