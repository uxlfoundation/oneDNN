/*******************************************************************************
* Copyright 2022,2024 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_deconvolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};

    const auto scratchpad = ctx.get_scratchpad_grantor();

    auto src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);

    bool use_dst_acc_for_sum = pd()->acl_pd_conf.use_dst_acc_for_sum;
    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = use_dst_acc_for_sum
            ? scratchpad.get<void>(memory_tracking::names::key_none)
            : CTX_OUT_MEM(void *, DNNL_ARG_DST);

    // Retrieve primitive resource and configured Compute Library objects
    auto *acl_resource
            = ctx.get_resource_mapper()->get<acl_deconv_resource_t>(this);
    acl_deconv_obj_t &acl_obj = acl_resource->get_acl_obj();

    acl_obj.src_tensor.allocator()->import_memory(const_cast<void *>(src_base));
    acl_obj.wei_tensor.allocator()->import_memory(const_cast<void *>(wei_base));
    acl_obj.bia_tensor.allocator()->import_memory(const_cast<void *>(bia_base));

    acl_obj.dst_tensor.allocator()->import_memory(dst_base);

    acl_obj.deconv.run();

    void *dst = acl_obj.dst_tensor.buffer();
    pd()->post_ops.execute(ctx, dst);

    acl_obj.src_tensor.allocator()->free();
    acl_obj.dst_tensor.allocator()->free();
    acl_obj.bia_tensor.allocator()->free();
    acl_obj.wei_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
