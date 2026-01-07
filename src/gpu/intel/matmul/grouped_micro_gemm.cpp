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

#include "gemmstone/microkernel_provider.hpp"
#include "gpu/intel/compute/ukernels.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/microkernels/shim.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t grouped_micro_gemm_t::init_microkernels(impl::engine_t *engine) {
    using namespace jit;
    using namespace gemmstone;
    using gemm::jit::convert_dnnl_to_kernel_type;

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    //arch_ = dev_info->gpu_arch();
    auto pd_ = (pd_t *)primitive_t::pd().get();

    /* Get device information */
    HWInformation hw_info;
    hw_info.euCount = dev_info->eu_count();
    hw_info.gmdid = dev_info->ip_version();
    hw_info.systolicAvailable = pd()->use_systolic_ukernel_;

    if (hw_info.gmdid == 0) return status::unimplemented;

    int m = static_cast<int>(pd_->dst_md(0)->dims[pd_->dst_md(0)->ndims - 2]);
    int n = static_cast<int>(pd_->dst_md(0)->dims[pd_->dst_md(0)->ndims - 1]);
    int k = static_cast<int>(pd_->src_md(0)->dims[pd_->src_md(0)->ndims - 1]);
    auto src_mdw = memory_desc_wrapper(pd()->src_md(0));
    auto wei_mdw = memory_desc_wrapper(pd()->weights_md());
    //auto dst_mdw = memory_desc_wrapper(pd()->dst_md(0));

    // User(rm)                         Gemmstone(cm)
    // MxN   MxK   KxN        NxM   MxK     KxN
    // Dst  =  Src  x  Weights  ->  Dst  =  Weights x Src

    //printf("User              D=SxW M=%d, N=%d, K=%d\n", m, n, k);
    //printf("Grouped MicroGEMM D=WxS M=%d, N=%d, K=%d\n", n, m, k);
    GEMMProblem problem;
    problem.Ta_ext = convert_dnnl_to_kernel_type(src_mdw.data_type());
    problem.Tb_ext = convert_dnnl_to_kernel_type(wei_mdw.data_type());
    problem.Tc_ext = problem.Ts = problem.Tc = Type::f32;
    problem.Ta = problem.Ta_ext;
    problem.Tb = problem.Tb_ext;
    problem.A.layout = MatrixLayout::T;
    problem.B.layout = MatrixLayout::T;
    problem.C.layout = MatrixLayout::T;
    problem.A.setAlignment(alignmentForLD(k * problem.Ta));
    problem.B.setAlignment(alignmentForLD(n * problem.Tb));
    problem.C.setAlignment(problem.Tc.size());
    //problem.A.setAlignment(alignmentForLD(ldk));

    SizeParams sizes;
    sizes.m = static_cast<uint16_t>(m);
    sizes.n = static_cast<uint16_t>(n);
    sizes.k = static_cast<uint16_t>(k);
    //std::cout << "Sizes: " << sizes.m << ", " << sizes.n << ", " << sizes.k << std::endl;

    micro::Package gemm;

    micro::GEMMProtocol::Options opts;
    //std::vector<StrategyRequirement> reqs;
    //reqs.push_back(StrategyRequirement::UnrollM == 32);
    //reqs.push_back(StrategyRequirement::UnrollN == 16);
    //reqs.push_back(StrategyRequirement::WGM == 2);
    //reqs.push_back(StrategyRequirement::WGN == 1);

    try {
        gemm = selectGEMMMicrokernel(opts, hw_info, sizes, problem);
    } catch (const std::runtime_error &ex) {
        return status::unimplemented;
        //CHECK_BOOL(false,
        //         "gemm microkernel generation failure with message: %s",
        //         ex.what());
    }
    //printf("Gemm: sg_per_wg_m=%d, sg_per_wg_n=%d, unroll_m=%d, unroll_n=%d\n",
    //        gemm.getSetting("sg_per_wg_m"), gemm.getSetting("sg_per_wg_n"),
    //        gemm.getSetting("sg_tile_m"), gemm.getSetting("sg_tile_n"));

    auto sg_size = dev_info->min_subgroup_size();

    /* Generate microkernel shims */
    micro::ShimOptions shimOptions;
    shimOptions.subgroupSize = sg_size;
    shimOptions.useTileOps = true;
    shimOptions.decorator = "grouped";

    kernel_ctx_.define_int("SUBGROUP_SIZE", sg_size);
    kernel_ctx_.add_custom_header("gemm_grouped.h",
            micro::generateShim(
                    gemm, micro::HostLanguage::OpenCL_C, shimOptions));

    pd_->sg_per_wg_m_ = gemm.getSetting("sg_per_wg_m");
    pd_->sg_per_wg_n_ = gemm.getSetting("sg_per_wg_n");
    pd_->sg_tile_m_ = gemm.getSetting("sg_tile_m");
    pd_->sg_tile_n_ = gemm.getSetting("sg_tile_n");
    if (gemm.grfMin > 128 || gemm.grfMin > 128) pd_->use_256_grf_ = true;

    return status::success;
}

status_t grouped_micro_gemm_t::pd_t::init(impl::engine_t *engine) {
    using namespace data_type;

    src_dt_ = src_md(0)->data_type;
    wei_dt_ = weights_md(0)->data_type;
    dst_dt_ = dst_md(0)->data_type;

    memory_desc_wrapper src_d(src_md());
    memory_desc_wrapper wei_d(weights_md(0));
    memory_desc_wrapper dst_d(dst_md());
    //std::cout << "src: " << src_d.dims()[0] << "," << src_d.dims()[1]
    //          << std::endl;
    //std::cout << "wei: " << wei_d.dims()[0] << "," << wei_d.dims()[1] << ","
    //          << wei_d.dims()[2] << std::endl;
    //std::cout << "dst: " << dst_d.dims()[0] << "," << dst_d.dims()[1]
    //          << std::endl;

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
    VDISPATCH_MATMUL(
            utils::one_of(src_dt_, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(wei_dt_, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(
            utils::one_of(dst_dt_, f32, f16, bf16), VERBOSE_UNSUPPORTED_DT_CFG);

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

    // No scales/post-ops for now
    VDISPATCH_MATMUL(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    assert(engine->kind() == engine_kind::gpu);
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    auto *dev_info = intel_engine->device_info();
    //arch_ = dev_info->gpu_arch();
    VDISPATCH_MATMUL(compute::mayiuse_microkernels(intel_engine),
            VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "microkernels");

    use_systolic_ukernel_ = intel_engine->mayiuse(
            compute::device_ext_t::intel_subgroup_matrix_multiply_accumulate);
    sg_size_ = dev_info->min_subgroup_size();

    return status::success;
}

status_t grouped_micro_gemm_t::init(impl::engine_t *engine) {

    CHECK(init_microkernels(engine));
    kernel_ctx_.set_data_type(pd()->dst_md()->data_type);

    if (pd()->use_256_grf_)
        kernel_ctx_.add_option("-cl-intel-256-GRF-per-thread");

    def_data_type(kernel_ctx_, pd()->src_dt_, "SRC");
    def_data_type(kernel_ctx_, pd()->wei_dt_, "WEI");
    def_data_type(kernel_ctx_, pd()->dst_dt_, "DST");
    return create_kernel(engine, &kernel_, "grouped_micro_gemm", kernel_ctx_);
}

status_t grouped_micro_gemm_t::execute(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    auto pd_ = pd();
    const auto *src_md = pd()->src_md(0);
    const auto *wei_md = pd()->weights_md();
    const auto *dst_md = pd()->dst_md(0);

    const size_t num_groups = pd()->ngroups_;

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_bias = pd()->with_bias();

    int m_all = static_cast<int>(dst_md->dims[dst_md->ndims - 2]);
    int n = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    int k = static_cast<int>(src_md->dims[src_md->ndims - 1]);
    //printf("m_all=%d, n=%d, k=%d\n", m_all, n, k);

    int ldsrc = static_cast<int>(src_md->dims[src_md->ndims - 1]);
    int ldwei = static_cast<int>(wei_md->dims[wei_md->ndims - 1]);
    int lddst = static_cast<int>(dst_md->dims[dst_md->ndims - 1]);
    //printf("lda=%d, ldb=%d, ldc=%d\n", ldsrc, ldwei, lddst);

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src_data);
    arg_list.append(ldsrc);
    arg_list.append(wei_data);
    arg_list.append(ldwei);
    arg_list.append(dst_data);
    arg_list.append(lddst);
    arg_list.append(src_offsets);
    arg_list.append(dst_offsets);
    arg_list.append(n);
    arg_list.append(k);
    //arg_list.append((int)num_groups);

    if (with_bias) {
        const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);
        arg_list.append(bias_data);
    }
    if (with_src_scales) {
        const auto &src_scales
                = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        arg_list.append(src_scales);
    }

    //printf("m_all=%d, n=%d, k=%d sg_per_wg_m: %ld sg_per_wg_n: %ld\n", m_all, n,
    //k, pd_->sg_per_wg_m_, pd_->sg_per_wg_n_);
    // Use total_tokens as upper bound for M dimension
    compute::range_t lws = {(size_t)pd_->sg_per_wg_m_ * pd_->sg_size_,
            (size_t)pd_->sg_per_wg_n_, 1};
    compute::range_t gws = {utils::div_up(m_all, pd_->sg_tile_m_) * lws[0],
            utils::div_up(n, pd_->sg_tile_n_) * lws[1], num_groups};
    //std::cout << "LWS: " << lws.str() << std::endl;
    //std::cout << "GWS: " << gws.str() << std::endl;

    return parallel_for(ctx, compute::nd_range_t(gws, lws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
