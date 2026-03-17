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

#ifndef DNNL_TEST_INTERNAL_SDPA_INTERNAL_HPP
#define DNNL_TEST_INTERNAL_SDPA_INTERNAL_HPP

#include "dnnl.hpp"

#include "common/sdpa_test_iface.hpp"

// NOLINTBEGIN(readability-identifier-naming)

namespace dnnl {
namespace impl {

/// Scaled Dot Product Attention (sdpa) internal primitive.
/// Implementing internally for more flexible validation
struct sdpa : public dnnl::primitive {
    /// Primitive descriptor for a sdpa primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc *attn_mask_desc,
                const memory::desc &scale_desc, const memory::desc &output_desc,
                bool invert_scale, memory::dim kv_head_number,
                int attn_mask_type, int softmax_alg,
                prop_kind_t prop_kind = prop_kind::forward_inference,
                const primitive_attr &attr = default_attr(),
                const primitive_attr &kq_attr = default_attr(),
                const primitive_attr &vs_attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = sdpa_primitive_desc_create(&pd,
                    aengine.get(), query_desc.get(), key_desc.get(),
                    value_desc.get(), output_desc.get(),
                    optional_arg(attn_mask_desc), scale_desc.get(),
                    invert_scale, kv_head_number, attn_mask_type,
                    (dnnl_alg_kind_t)softmax_alg, (prop_kind_t)prop_kind,
                    attr.get(), kq_attr.get(), vs_attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a sdpa "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    sdpa() = default;

    /// Constructs a sdpa primitive.
    /// @param pd Primitive descriptor for a sdpa primitive.
    sdpa(const primitive_desc &pd) : primitive(pd) {}
};

/// Scaled Dot Product Attention (sdpa) backward propagation internal primitive.
/// Implementing internally for more flexible validation
struct sdpa_backward : public dnnl::primitive {
    /// Primitive descriptor for a sdpa_backward primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        primitive_desc(const engine &aengine, const memory::desc &query_desc,
                const memory::desc &key_desc, const memory::desc &value_desc,
                const memory::desc *attn_mask_desc,
                const memory::desc &scale_desc, const memory::desc &output_desc,
                const memory::desc &diff_query_desc,
                const memory::desc &diff_key_desc,
                const memory::desc &diff_value_desc,
                const memory::desc &diff_output_desc,
                const memory::desc *dS_desc, bool invert_scale,
                memory::dim kv_head_number, int attn_mask_type, int softmax_alg,
                const sdpa::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = default_attr()) {

            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = sdpa_primitive_desc_create(&pd,
                    aengine.get(), query_desc.get(), key_desc.get(),
                    value_desc.get(), output_desc.get(),
                    optional_arg(attn_mask_desc), scale_desc.get(),
                    diff_query_desc.get(), diff_key_desc.get(),
                    diff_value_desc.get(), diff_output_desc.get(),
                    dS_desc ? dS_desc->get() : nullptr, invert_scale,
                    kv_head_number, attn_mask_type,
                    (dnnl_alg_kind_t)softmax_alg, attr.get(),
                    hint_fwd_pd.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a sdpa "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    sdpa_backward() = default;

    /// Constructs a sdpa primitive.
    /// @param pd Primitive descriptor for a sdpa primitive.
    sdpa_backward(const primitive_desc &pd) : primitive(pd) {}
};

} // namespace impl
} // namespace dnnl

// NOLINTEND(readability-identifier-naming)
#endif
