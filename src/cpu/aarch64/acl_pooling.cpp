/*******************************************************************************
* Copyright 2022-2023, 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_pooling.hpp"
#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_pooling_fwd_t::pd_t::init(engine_t *engine) {

    // Compute Library supports forward propagation only
    bool ok = set_default_params() == status::success && is_fwd()
            && utils::everyone_is(src_md()->data_type, dst_md()->data_type)
            && utils::one_of(
                    src_md()->data_type, data_type::f32, data_type::f16)
            && attr()->has_default_values()
            && attr_.set_default_formats(dst_md(0)) == status::success
            && !is_dilated() && !has_zero_dim_memory();

    ACL_CHECK_SUPPORT(!ok, "Unsupported primitive options");

    const pooling_desc_t *pod = desc();

    // Choose the pooling type
    const alg_kind_t alg = pod->alg_kind;
    const bool is_max_pool = (alg == alg_kind::pooling_max);
    asp_.pool_info.pool_type = is_max_pool ? arm_compute::PoolingType::MAX
                                           : arm_compute::PoolingType::AVG;

    // Check if workspace Tensor is needed
    const bool ws_init
            = (is_max_pool && pod->prop_kind == prop_kind::forward_training);
    asp_.use_ws = ws_init;

    ACL_CHECK_SUPPORT(ws_init && src_md()->data_type != data_type::f32,
            "ACL Max pooling forward training only supports f32");

    if (ws_init)
        // Compute Library only supports U32/S32 no U8
        init_default_ws(data_type::s32);
    auto src_tag = memory_desc_matches_one_of_tag(
            *src_md(), format_tag::nhwc, format_tag::nchw);
    auto dst_tag = memory_desc_matches_one_of_tag(
            *dst_md(), format_tag::nhwc, format_tag::nchw);

    ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag, dst_tag),
            "src or dst is not format nhwc or nchw");
    ACL_CHECK_SUPPORT(
            src_tag != dst_tag, "src and dst have different memory formats");

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper dst_d(dst_md());
    const int ndims = src_d.ndims();
    ACL_CHECK_SUPPORT(ndims != 4, "Tensor is not 4d");

    // Pooling window
    asp_.pool_info.pool_size = arm_compute::Size2D(KW(), KH());
    // Choose the data layout
    bool is_nhwc = src_tag == format_tag::nhwc;
    const auto acl_layout = is_nhwc ? arm_compute::DataLayout::NHWC
                                    : arm_compute::DataLayout::NCHW;
    asp_.pool_info.data_layout = acl_layout;
    const auto acl_data_t = acl_utils::get_acl_data_t(src_d.data_type());

    bool use_square_acl_kernel
            = !is_nhwc && KH() == KW() && (KH() == 2 || KH() == 3 || KH() == 7);
    if (is_max_pool) {
        ACL_CHECK_SUPPORT(
                !use_acl_max_pool_heuristic(
                        MB() * IC() * OH() * OW() * KH() * KW(),
                        dnnl_get_max_threads(), is_nhwc, use_square_acl_kernel,
                        pod->prop_kind == prop_kind::forward_training),
                "ACL not used as profiling suggests that native oneDNN "
                "kernels are faster for this problem");
    } else {
        ACL_CHECK_SUPPORT(
                !use_acl_avg_pool_heuristic(
                        MB() * IC() * OH() * OW() * KH() * KW(),
                        dnnl_get_max_threads(), is_nhwc, use_square_acl_kernel),
                "ACL not used as profiling suggests that native oneDNN "
                "kernels are faster for this problem");
    }

    asp_.pool_info.exclude_padding
            = (alg == alg_kind::pooling_avg_exclude_padding);

    asp_.pool_info.pad_stride_info
            = arm_compute::PadStrideInfo(KSW(), KSH(), padL(), padR(), padT(),
                    padB(), arm_compute::DimensionRoundingType::FLOOR);

    asp_.src_info = arm_compute::TensorInfo(is_nhwc
                    ? arm_compute::TensorShape(IC(), IW(), IH(), MB())
                    : arm_compute::TensorShape(IW(), IH(), IC(), MB()),
            1, acl_data_t, acl_layout);
    asp_.dst_info = arm_compute::TensorInfo(is_nhwc
                    ? arm_compute::TensorShape(OC(), OW(), OH(), MB())
                    : arm_compute::TensorShape(OW(), OH(), OC(), MB()),
            1, acl_data_t, acl_layout);

    // Use datatype lowest property instead of using -INF
    asp_.pool_info.use_inf_as_limit = false;

    if (ws_init) {
        asp_.ws_info = arm_compute::TensorInfo(is_nhwc
                        ? arm_compute::TensorShape(OC(), OW(), OH(), MB())
                        : arm_compute::TensorShape(OW(), OH(), OC(), MB()),
                1, arm_compute::DataType::U32, acl_layout);

        // Return kernel indices instead of source indices.
        asp_.pool_info.use_kernel_indices = true;
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuPool2d::validate(
                &asp_.src_info, &asp_.dst_info, asp_.pool_info, &asp_.ws_info));
    } else {
        asp_.pool_info.use_kernel_indices = false;
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuPool2d::validate(
                &asp_.src_info, &asp_.dst_info, asp_.pool_info));
    }

    arm_compute::experimental::op::CpuPool2d pool;
    if (ws_init)
        pool.configure(
                &asp_.src_info, &asp_.dst_info, asp_.pool_info, &asp_.ws_info);
    else
        pool.configure(&asp_.src_info, &asp_.dst_info, asp_.pool_info);

    aux_mem_req_ = pool.workspace();
    auto scratchpad = scratchpad_registry().registrar();
    CHECK(init_scratchpad(scratchpad, aux_mem_req_));

    return status::success;
}

// Generally, Compute Library is faster above a per thread problem size 'cutoff'.
// The value of this cutoff has been found empirically,
// through profiling on a Neoverse-N1 cpu.
// Note: This rule is approximate, Not all problems follow this rule.
//
// Parameters used in the heuristics:
// - problem_size: defined as mb * ic * oh * ow * kh * kw
// - thread_count
// - is_nhwc (as opposed to nchw)
// - use_square_acl_kernels: For nchw pooling, Compute Library has faster kernels
//   for pooling window (kernel) sizes of 2x2, 3x3, or 7x7.
// - is_training: Max pooling training cases require a workspace tensor,
//   so these cases have been implemented in a seperate kernel in Compute Library.
bool acl_pooling_fwd_t::pd_t::use_acl_avg_pool_heuristic(int problem_size,
        int thread_count, bool is_nhwc, bool use_square_acl_kernel) {
    int cutoff;
    if (is_nhwc) {
        if (thread_count == 1)
            cutoff = 200;
        else
            cutoff = 4096;
    } else {
        if (use_square_acl_kernel) {
            if (thread_count == 1)
                cutoff = 100;
            else if (thread_count > 32)
                return false;
            else
                cutoff = 2048;
        } else
            return false;
    }
    return problem_size > cutoff * thread_count;
}

bool acl_pooling_fwd_t::pd_t::use_acl_max_pool_heuristic(int problem_size,
        int thread_count, bool is_nhwc, bool use_square_acl_kernel,
        bool is_training) {
    int cutoff;
    if (is_nhwc) {
        if (thread_count == 1)
            cutoff = 200;
        else {
            if (is_training)
                cutoff = 2048;
            else
                cutoff = 4096;
        }
    } else {
        if (use_square_acl_kernel) {
            if (thread_count == 1)
                cutoff = 100;
            else
                cutoff = 1024;
        } else {
            if (thread_count == 1)
                return true;
            else if (thread_count > 16)
                return false;
            else
                cutoff = 25000;
        }
    }
    return problem_size > cutoff * thread_count;
}

status_t acl_pooling_fwd_t::pd_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const arm_compute::experimental::MemoryRequirements &aux_mem_req) {

    if (!aux_mem_req.empty()) {
        const auto &req = aux_mem_req[0];
        if (req.size > 0) {
            size_t align = req.alignment ? req.alignment : 64;
            if (align & (align - 1)) align = utils::rnd_up_pow2(align);
            scratchpad.book(memory_tracking::names::key_pool_wsp_buffer,
                    req.size, 1, align, align);
        }
    }

    return status::success;
}

acl_pooling_fwd_t::acl_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , pooling_op_(
              std::make_unique<arm_compute::experimental::op::CpuPool2d>()) {}

status_t acl_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

status_t acl_pooling_fwd_t::init(engine_t *engine) {
    auto asp = pd()->asp_;
    auto ws_info = asp.use_ws ? &asp.ws_info : nullptr;
    // Configure pooling operation
    pooling_op_->configure(
            &asp.src_info, &asp.dst_info, asp.pool_info, ws_info);

    return status::success;
}

status_t acl_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    std::lock_guard<std::mutex> _lock {this->mtx};
    status_t status = status::success;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    void *ws_base;

    auto asp = pd()->asp_;

    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(asp.src_info);
    src_tensor.allocator()->import_memory(const_cast<void *>(src));
    dst_tensor.allocator()->init(asp.dst_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::Tensor scratch_tensor;

    void *scratchpad_base = ctx.get_scratchpad_grantor().get<void>(
            memory_tracking::names::key_pool_wsp_buffer);
    const auto &aux_mem_req = pd()->aux_mem_req_;

    scratch_tensor.allocator()->init(arm_compute::TensorInfo(
            arm_compute::TensorShape(aux_mem_req[0].size), 1,
            arm_compute::DataType::U8));
    scratch_tensor.allocator()->import_memory(scratchpad_base);

    arm_compute::Tensor ws_tensor;

    arm_compute::ITensorPack run_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src_tensor},
            {arm_compute::TensorType::ACL_DST_0, &dst_tensor},
            {arm_compute::TensorType::ACL_INT_0, &scratch_tensor}};

    if (asp.use_ws) {
        ws_base = CTX_OUT_MEM(void *, DNNL_ARG_WORKSPACE);
        ws_tensor.allocator()->init(asp.ws_info);
        ws_tensor.allocator()->import_memory(ws_base);
        run_pack.add_tensor(arm_compute::TensorType::ACL_DST_1, &ws_tensor);
    }
    pooling_op_->run(run_pack);

    return status;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
