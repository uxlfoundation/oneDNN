/*******************************************************************************
* Copyright 2021-2022,2024-2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_inner_product_fwd_t::init(engine_t *engine) {
    auto aip = pd()->aip_;
    inner_product_op_ = std::make_unique<
            arm_compute::experimental::op::CpuFullyConnected>();

    // Configure inner product operation. Performs memory allocation.
    inner_product_op_->configure(&aip.src_tensor_info, &aip.wei_tensor_info,
            aip.with_bias ? &aip.bia_tensor_info : nullptr,
            &aip.dst_tensor_info, aip.fc_info, aip.weights_info);

    return status::success;
}

status_t acl_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

status_t acl_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    // Lock here is needed because we can't guarantee
    // concurrent multithreaded access to the scratchpad.
    std::lock_guard<std::mutex> _lock {this->mtx};

    bool with_bias = pd()->aip_.with_bias;
    bool use_dst_acc_for_sum = pd()->aip_.use_dst_acc_for_sum;

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);

    const auto scratchpad = ctx.get_scratchpad_grantor();

    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = use_dst_acc_for_sum
            ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
            : CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto aip = pd()->aip_;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor = nullptr;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(aip.src_tensor_info);
    src_tensor.allocator()->import_memory(const_cast<void *>(src_base));
    wei_tensor.allocator()->init(aip.wei_tensor_info);
    wei_tensor.allocator()->import_memory(const_cast<void *>(wei_base));
    dst_tensor.allocator()->init(aip.dst_tensor_info);
    dst_tensor.allocator()->import_memory((dst_base));

    if (with_bias) {
        auto bia_base = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
        bia_tensor.allocator()->init(aip.bia_tensor_info);
        bia_tensor.allocator()->import_memory(const_cast<void *>(bia_base));
    }

    arm_compute::ITensorPack run_pack {
            {arm_compute::TensorType::ACL_SRC_0, &src_tensor},
            {arm_compute::TensorType::ACL_SRC_1, &wei_tensor},
            {arm_compute::TensorType::ACL_BIAS, &bia_tensor},
            {arm_compute::TensorType::ACL_DST, &dst_tensor}};

    inner_product_op_->run(run_pack);

    void *dst = dst_tensor.buffer();
    pd()->post_ops.execute(ctx, dst);

    return status::success;
}

status_t acl_inner_product_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using smask_t = primitive_attr_t::skip_mask_t;
    const format_kind_t weights_format_kind_received = weights_md_.format_kind;
    const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
            && attr()->has_default_values(smask_t::post_ops, f16);
    const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
            && attr()->has_default_values(
                    smask_t::post_ops | smask_t::fpmath_mode, f32);
    const bool is_weights_md_format_ok
            = utils::one_of(weights_format_kind_received, format_kind::any,
                    format_kind::blocked);
    const bool ok = is_fwd() && !has_zero_dim_memory()
            && utils::one_of(true, is_fp16_ok, is_fp32_ok)
            && is_weights_md_format_ok
            && set_default_params(true) == status::success;

    VDISPATCH_INNER_PRODUCT(ok, VERBOSE_SKIP_PRIMITIVE_IMPL);

    CHECK(init_conf_ip(engine, weights_format_kind_received));

    if (aip_.use_dst_acc_for_sum) {
        const memory_desc_wrapper dst_d(&dst_md_);
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.nelems(),
                dst_d.data_type_size());
    }

    return status::success;
}

status_t acl_inner_product_fwd_t::pd_t::init_conf_ip(
        engine_t *engine, format_kind_t weights_format_kind_received) {

    const int ndims = src_md()->ndims;

    const bool is_2d = (ndims == 2);
    const bool is_4d = (ndims == 4);

    ACL_CHECK_SUPPORT(!(is_2d || is_4d), "ACL supports only 2d or 4d cases");

    using namespace format_tag;
    auto src_tag = memory_desc_matches_one_of_tag(src_md_, nhwc, nchw, nc);
    auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, nc);

    ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag, dst_tag),
            "unsupported memory layout");

    ACL_CHECK_SUPPORT(
            is_2d && src_tag != dst_tag, "for src and dst layouts must match");

    const dim_t ic_total = IC_total();
    const dim_t n = MB();
    const dim_t oc = OC();

    aip_.src_tensor_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(ic_total, n), 1,
                    acl_utils::get_acl_data_t(src_md()->data_type));

    // ACL requires the weights to be in 2D flattened shape
    aip_.wei_tensor_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(oc, ic_total), 1,
                    acl_utils::get_acl_data_t(weights_md(0)->data_type));

    auto acl_dst_data_t = acl_utils::get_acl_data_t(dst_md()->data_type);
    aip_.dst_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(oc, n), 1, acl_dst_data_t);

    aip_.with_bias = desc()->bias_desc.format_kind != format_kind::undef;
    auto acl_bia_data_t = aip_.with_bias
            ? acl_utils::get_acl_data_t(weights_md(1)->data_type)
            : acl_dst_data_t;
    aip_.bia_tensor_info = arm_compute::TensorInfo(aip_.with_bias
                    ? arm_compute::TensorShape(oc)
                    : arm_compute::TensorShape(),
            1, acl_bia_data_t);

    aip_.fc_info.transpose_weights = false;

    aip_.fc_info.enable_fast_math = utils::one_of(
            attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);

    CHECK(post_ops.init(
            engine, attr_.post_ops_, dst_md_, aip_.fc_info.activation_info));
    aip_.use_dst_acc_for_sum = post_ops.has_sum();

    // WeightFormat::ANY tells ACL we can handle any format
    aip_.weights_info = arm_compute::WeightsInfo(
            false, 1, 1, ic_total, false, arm_compute::WeightFormat::ANY);

    // Get the format that the ACL kernel will expect the weights to be
    // in (if a kernel exists) Note that these are referred to as fixed
    // format kernels, because they require one specific weights format
    arm_compute::WeightFormat expected_weight_format;
    ACL_CHECK_VALID(
            arm_compute::experimental::op::CpuFullyConnected::has_opt_impl(
                    expected_weight_format, &aip_.src_tensor_info,
                    &aip_.wei_tensor_info,
                    aip_.with_bias ? &aip_.bia_tensor_info : nullptr,
                    &aip_.dst_tensor_info, aip_.fc_info, aip_.weights_info));

    // Set weights info to the one returned by has_opt_impl
    aip_.weights_info.set_weight_format(expected_weight_format);

    // has_opt_impl may return a non fast math kernel, even if requested
    aip_.fc_info.enable_fast_math
            = arm_compute::is_fixed_format_fast_math(expected_weight_format);

    // Inner product is the same as the matmul n x (chw) * (ihw) x o
    // (note that the src c and weights i both correspond to the input
    // channel). ACL FullyConnectedLayer assumes the chw dimensions of
    // src and ihw dimensions of weights are collapsed, so we need to
    // make sure that they have the same layout. Given that weights are
    // more often fixed, (so reorders can be hoisted) it makes sense to
    // reorder the weights to fit the src.

    // For 4D tensors we need to:
    // - reorder the ihw of the weights to match the src chw
    // - collapse ihw
    // - pad the collapsed ihw
    // But there is not yet a way to express this collapse+pad as a
    // reorder. So we try to reorder the weights to match the src,
    // implicitly collapse ihw in our definition of the weights
    // TensorInfo and hope that the inner_dim has zero padding
    // (weights_md_.dims[inner_dim] % block_by == 0). If it does, we
    // fall back to a kernel without blocking (currently this is
    // equivalent to non-fastmath).

    // 2D just works because we just pad the only dimension.

    // o_dim is always the first logical dimension (oihw, ohwi, oi)
    dim_t o_dim = 0;
    dim_t inner_dim;
    // Rest of logical dimensions in order of innermost to outermost
    std::vector<dim_t> remaining_dims = {};

    if (src_tag == nchw) {
        inner_dim = 3; // w
        remaining_dims = {2, 1}; // h, i
    } else if (src_tag == nhwc) {
        inner_dim = 1; // i
        remaining_dims = {3, 2}; // w, h
    } else if (src_tag == nc) { // Only remaining case is 2D (nc)
        inner_dim = 1; // i
        remaining_dims = {}; // No other dimensions for 2D
    } else {
        assert(false);
    }

    // Fallback
    int block_by = arm_compute::block_by(expected_weight_format);
    bool is_bf16 = src_md()->data_type == data_type::bf16
            && weights_md()->data_type == data_type::bf16
            && dst_md()->data_type == data_type::bf16;
    if (is_4d && weights_md_.dims[inner_dim] % block_by != 0
            && (aip_.fc_info.enable_fast_math || is_bf16)) {
        aip_.fc_info.enable_fast_math = false;
        aip_.weights_info.set_weight_format(arm_compute::WeightFormat::ANY);
        ACL_CHECK_VALID(
                arm_compute::experimental::op::CpuFullyConnected::has_opt_impl(
                        expected_weight_format, &aip_.src_tensor_info,
                        &aip_.wei_tensor_info,
                        aip_.with_bias ? &aip_.bia_tensor_info : nullptr,
                        &aip_.dst_tensor_info, aip_.fc_info,
                        aip_.weights_info));
        aip_.weights_info.set_weight_format(expected_weight_format);
        block_by = arm_compute::block_by(expected_weight_format);

        VDISPATCH_INNER_PRODUCT(weights_md_.dims[inner_dim] % block_by == 0,
                VERBOSE_SKIP_PRIMITIVE_IMPL);
    }

    const memory_desc_t weights_md_received = weights_md_;
    acl_utils::reorder_to_weight_format(aip_.wei_tensor_info, weights_md_,
            expected_weight_format, inner_dim, o_dim, remaining_dims, {});

    ACL_CHECK_SUPPORT((weights_format_kind_received == format_kind::blocked)
                    && !(dnnl_memory_desc_equal(
                            &weights_md_received, &weights_md_)),
            "specific blocked format not supported by ACL, use "
            "format_kind_t::any to find a supported blocked format for "
            "your platform");

    // Validate fully connected layer manually to check for return status
    ACL_CHECK_VALID(arm_compute::experimental::op::CpuFullyConnected::validate(
            &aip_.src_tensor_info, &aip_.wei_tensor_info,
            aip_.with_bias ? &aip_.bia_tensor_info : nullptr,
            &aip_.dst_tensor_info, aip_.fc_info, aip_.weights_info));

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
