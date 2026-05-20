/*******************************************************************************
* Copyright 2022-2024, 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_POST_OPS_FALLBACK_HPP
#define CPU_AARCH64_POST_OPS_FALLBACK_HPP

#include "common/eltwise_pd.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct post_ops_fallback_t {

    post_ops_fallback_t() = default;

    // init the post_ops_fallback_t. Note that this function modifies the passed in
    // post ops by setting the preferred memory formats
    status_t init(engine_t *engine, post_ops_t &post_ops,
            const memory_desc_t &dst_md, int post_op_start_index = 0) {

        post_op_start_index_ = post_op_start_index;

        CHECK(post_ops.set_default_formats(&dst_md));
        dst_data_type = dst_md.data_type;

        // Reset properties derived from post_ops
        sum_index = -1;
        post_op_primitives = {};

        for (int i = post_op_start_index; i < post_ops.len(); i++) {
            auto &po = post_ops.entry_[i];

            if (po.is_sum()) {
                ACL_CHECK_SUPPORT(po.sum.scale != 1.0f,
                        "sum post op scale must be 1 (no scale)");

                ACL_CHECK_SUPPORT(po.sum.zero_point != 0,
                        "sum post op zero point must be 0 (no shift)");

                // >= 0 means we had one already
                ACL_CHECK_SUPPORT(sum_index >= 0,
                        "there must not be more than 1 sum post op");

                sum_index = i;

                // Sum is an add primitive where dst = temp_dst + dst
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = alg_kind::binary_add;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = dst_md;
                po_desc.dst_desc = dst_md;

                std::shared_ptr<primitive_t> binary_prim;
                CHECK(create_binary_primitive(engine, po_desc, binary_prim));
                post_op_primitives.push_back(std::move(binary_prim));

            } else if (po.is_binary()) {
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = po.binary.alg;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = po.binary.src1_desc;
                if (po.binary.alg == alg_kind::binary_select) {
                    po_desc.src_desc[2] = po.binary.src2_desc;
                }
                po_desc.dst_desc = dst_md;

                std::shared_ptr<primitive_t> binary_prim;
                CHECK(create_binary_primitive(engine, po_desc, binary_prim));
                post_op_primitives.push_back(std::move(binary_prim));

            } else if (po.is_eltwise()) {
                ACL_CHECK_SUPPORT(po.eltwise.scale != 1.0f,
                        "eltwise post op scale must be 1 (no scale)");

                // Use the helper function to validate the descriptor arguments and
                // assign them to our eltwise_desc_t
                eltwise_desc_t ed;
                CHECK(eltwise_desc_init(&ed, prop_kind_t::dnnl_forward,
                        po.eltwise.alg, &dst_md, &dst_md, nullptr, nullptr,
                        po.eltwise.alpha, po.eltwise.beta));

                std::shared_ptr<primitive_t> eltwise_prim;
                CHECK(create_eltwise_primitive(engine, ed, eltwise_prim));
                post_op_primitives.push_back(std::move(eltwise_prim));

            } else {
                // Unsupported catchall
                return status::unimplemented;
            }
        }

        return status::success;
    }

    bool has_sum() const { return sum_index >= 0; }

    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

    status_t execute(
            const exec_ctx_t &ctx, void *src, void *dst = nullptr) const;

private:
    status_t create_binary_primitive(engine_t *engine,
            const binary_desc_t &binary_desc,
            std::shared_ptr<primitive_t> &primitive) const {
        auto empty_attr = dnnl_primitive_attr();

        primitive_desc_iterator_t it(engine,
                reinterpret_cast<const op_desc_t *>(&binary_desc), &empty_attr,
                nullptr);

        std::shared_ptr<primitive_desc_t> binary_pd;
        while (++it != it.end()) {
            binary_pd = *it;
            if (binary_pd) break;
        }
        if (!binary_pd) return status::unimplemented;

        return binary_pd->create_primitive(primitive, engine);
    }

    status_t create_eltwise_primitive(engine_t *engine,
            const eltwise_desc_t &eltwise_desc,
            std::shared_ptr<primitive_t> &primitive) const {
        auto empty_attr = dnnl_primitive_attr();

        primitive_desc_iterator_t it(engine,
                reinterpret_cast<const op_desc_t *>(&eltwise_desc), &empty_attr,
                nullptr);

        std::shared_ptr<primitive_desc_t> eltwise_pd;
        while (++it != it.end()) {
            eltwise_pd = *it;
            if (eltwise_pd) break;
        }
        if (!eltwise_pd) return status::unimplemented;

        return eltwise_pd->create_primitive(primitive, engine);
    }

    status_t execute_binary(const exec_ctx_t &ctx, const primitive_t *post_op,
            const void *src0, const void *src1, const void *src2, void *dst,
            int primitive_index) const;

    status_t execute_eltwise(const exec_ctx_t &ctx, const primitive_t *post_op,
            void *src, int primitive_index) const;

    // Index of the sum post op if there is one, < 0 means no sum
    int sum_index = -1;
    // Index of the first post op this primitive executes. This is typically the
    // number of post ops which were fused.
    int post_op_start_index_ = 0;
    data_type_t dst_data_type;
    // Vector of primitives used to execute the post ops.
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
