/*******************************************************************************
* Copyright 2023 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_reorder.hpp"

namespace {
/*
* Find the index of the dense dimension.
* The stride of the inner most dense block will be
* multiplied by the blocking of all prior blocks.
*/
int find_innermost_dense_idx(const dnnl::impl::memory_desc_t *md) {
    uint32_t dense_blk = 1;
    for (int i = 0; i < md->format_desc.blocking.inner_nblks; i++) {
        dense_blk *= md->format_desc.blocking.inner_blks[i];
    }

    int dense_idx = -1;
    for (int i = 0; i < md->ndims; i++) {
        if (md->format_desc.blocking.strides[i] == dense_blk) dense_idx = i;
    }
    return dense_idx;
}
} // namespace

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_reorder_resource_t::configure(const acl_reorder_conf_t &app) {
    if (!acl_obj_) return status::out_of_memory;

    // Init Compute Library tensors based on info from descriptor
    acl_obj_->src_tensor.allocator()->init(app.src_info);
    acl_obj_->dst_tensor.allocator()->init(app.dst_info);

    // clang-format off
    acl_obj_->reorder.configure(
        &acl_obj_->src_tensor,
        &acl_obj_->dst_tensor,
        app.src_wf,
        app.dst_wf,
        app.transpose
        );
    // clang-format on

    return status::success;
}

status_t acl_reorder_fwd_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    using namespace acl_utils;

    // ComputeLibrary reorders support f32->f32 and f32->bf16
    bool ok = src_md->data_type == data_type::f32
            && utils::one_of(dst_md->data_type, data_type::f32, data_type::bf16)
            && attr->has_default_values();

    VDISPATCH_REORDER_IC(ok, "unsupported datatype");

    // Create and check primitive descriptor
    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    VDISPATCH_REORDER_IC(
            _pd->init(engine, src_engine, dst_engine) == status::success,
            "pd initialization failed");

    // In case we have two or four dimensions, we can't have one of the
    // two first dimensions as 1. This is valid for f32->f32 and f32->bf16.
    VDISPATCH_REORDER_IC(dst_md->dims[0] != 1 && dst_md->dims[1] != 1,
            "first two dimensions of the reorder being 1 is not supported");

    auto src_tag = memory_desc_matches_one_of_tag(
            *src_md, format_tag::ab, format_tag::ba, format_tag::cdba);
    VDISPATCH_REORDER_IC(format_tag::undef != src_tag,
            "Only ab, ba or cdba source formats supported");

    auto &transpose = _pd->app_.transpose;
    auto &dst_blocking = dst_md->format_desc.blocking;

    VDISPATCH_REORDER_IC(src_md->ndims == dst_md->ndims,
            "Number of dimensions in src and dst do not match");
    VDISPATCH_REORDER_IC((dst_md->ndims == 2 || dst_md->ndims == 4),
            "ACL only supports 2D and 4D reorders");
    // Check if a transpose is needed during the reorder
    if (src_md->ndims == 4) {
        VDISPATCH_REORDER_IC(
                memory_desc_matches_tag(*src_md, dnnl::impl::format_tag::cdba)
                        && (memory_desc_matches_one_of_tag(*dst_md,
                                    dnnl::impl::format_tag::Acdb4a,
                                    dnnl::impl::format_tag::Acdb8a)
                                != format_tag::undef),
                VERBOSE_UNSUPPORTED_TAG);
        transpose = true;
    } else {
        int src_dense_idx = find_innermost_dense_idx(src_md);
        int dst_dense_idx = find_innermost_dense_idx(dst_md);

        transpose = src_dense_idx != dst_dense_idx;
    }

    // Return unimplemented for non-transposed reorders for now
    // as they are faster in JIT for most cases.
    VDISPATCH_REORDER_IC(
            transpose, "non-transposed reorders are not supported");

    auto &dst_wf = _pd->app_.dst_wf;

    VDISPATCH_REORDER_IC(
            dst_blocking.inner_nblks <= 2, VERBOSE_UNSUPPORTED_TAG);
    // Offsets to calculate the enum for ComputeLibrary weight formats
    // defined in arm_compute/core/CoreTypes.h
    const auto interleave_offset = 0x000100;
    const auto block_by_offset = 0x100000;
    for (int i = 0; i < dst_blocking.inner_nblks; i++) {
        auto blk = dst_blocking.inner_blks[i];
        if (i == 0) {
            auto offset = interleave_offset;
            dst_wf = (arm_compute::WeightFormat)(
                    static_cast<long int>(dst_wf) + offset * (blk - 1));
        } else if (i == 1) {
            auto offset = block_by_offset;
            // Set block_by
            dst_wf = (arm_compute::WeightFormat)(
                    static_cast<long int>(dst_wf) + offset * (blk - 1));
        }
    }

    arm_compute::TensorShape acl_tensor_shape_in;
    arm_compute::TensorShape acl_tensor_shape_out;

    // Switch for 2 or 4 dim tensors
    switch (src_md->ndims) {
        case 2: {
            if ((src_tag == format_tag::ab && transpose)
                    || (src_tag == format_tag::ba && !transpose)) {
                acl_tensor_shape_in = arm_compute::TensorShape(
                        src_md->dims[0], src_md->dims[1]);
                acl_tensor_shape_out = arm_compute::TensorShape(
                        dst_md->padded_dims[0], dst_md->padded_dims[1]);
            } else if ((src_tag == format_tag::ba && transpose)
                    || (src_tag == format_tag::ab && !transpose)) {
                acl_tensor_shape_in = arm_compute::TensorShape(
                        src_md->dims[1], src_md->dims[0]);
                acl_tensor_shape_out = arm_compute::TensorShape(
                        dst_md->padded_dims[1], dst_md->padded_dims[0]);
            } else {
                VINFO(primitive, create, dispatch, reorder,
                        "Unsupported source tag for 2D reorder");
                return status::unimplemented;
            }
        } break;
        case 4: {
            // Currently only supporting AxBx1x1 cases
            VDISPATCH_REORDER_IC(dst_md->dims[2] == 1 && dst_md->dims[3] == 1,
                    "currently only AxBx1x1 4d reorders are supported");

            acl_tensor_shape_in = arm_compute::TensorShape(src_md->dims[3],
                    src_md->dims[2], src_md->dims[1], src_md->dims[0]);
            acl_tensor_shape_out = arm_compute::TensorShape(
                    dst_md->padded_dims[3], dst_md->padded_dims[2],
                    dst_md->padded_dims[1], dst_md->padded_dims[0]);
            break;
        }
        default: {
            VINFO(primitive, create, dispatch, reorder,
                    VERBOSE_UNSUPPORTED_TAG);
            return status::unimplemented;
        }
    }

    // Choose the data layout
    const auto acl_layout = arm_compute::DataLayout::NCHW;

    // Set Source WeightFormat
    _pd->app_.src_wf = arm_compute::WeightFormat::OHWI;

    // Create ACL tensor infos
    const arm_compute::DataType src_acl_data_t
            = acl_utils::get_acl_data_t(src_md->data_type);
    _pd->app_.src_info = arm_compute::TensorInfo(
            acl_tensor_shape_in, 1, src_acl_data_t, acl_layout);

    const arm_compute::DataType dst_acl_data_t
            = acl_utils::get_acl_data_t(dst_md->data_type);
    _pd->app_.dst_info = arm_compute::TensorInfo(
            acl_tensor_shape_out, 1, dst_acl_data_t, acl_layout);

    ACL_CHECK_VALID(arm_compute::NEReorderLayer::validate(&_pd->app_.src_info,
            &_pd->app_.dst_info, _pd->app_.src_wf, dst_wf,
            _pd->app_.transpose));
    // Init scratch memory, not used so 0 in this implementation
    _pd->init_scratchpad_md();

    return safe_ptr_assign(*reorder_pd, _pd.release());
}

status_t acl_reorder_fwd_t::create_resource(
        engine_t *engine, resource_mapper_t &mapper) const {
    if (mapper.has_resource(this)) return status::success;

    auto r = utils::make_unique<acl_reorder_resource_t>();
    if (!r) return status::out_of_memory;

    // Configure the resource based on information from primitive descriptor
    CHECK(r->configure(pd()->app_));

    mapper.add(this, std::move(r));
    return status::success;
}

status_t acl_reorder_fwd_t::execute(const exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

status_t acl_reorder_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    // Lock here is needed because resource_mapper does not support
    // concurrent multithreaded access.
    std::lock_guard<std::mutex> _lock {this->mtx};

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_FROM);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_TO);

    // Retrieve primitive resource and configured Compute Library objects
    auto *acl_resource
            = ctx.get_resource_mapper()->get<acl_reorder_resource_t>(this);

    acl_reorder_obj_t &acl_obj = acl_resource->get_acl_obj();

    acl_obj.src_tensor.allocator()->import_memory(const_cast<void *>(src));
    acl_obj.dst_tensor.allocator()->import_memory(dst);

    acl_obj.reorder.run();

    acl_obj.src_tensor.allocator()->free();
    acl_obj.dst_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
