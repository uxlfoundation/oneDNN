/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#ifndef CPU_RV64_RVV_POSTOPS_HPP
#define CPU_RV64_RVV_POSTOPS_HPP

#include <memory>
#include <vector>
#include <riscv_vector.h>

#include "common/primitive_desc_iterator.hpp"
#include "cpu/rv64/rvv_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_postops_t {
    rvv_postops_t(const post_ops_t &po) : po_(po) {
        assert(po.len() <= 1 && "rvv_postops_t supports at most one post-op");
        if (po.len() > 0) {
            if (po.entry_[0].is_eltwise()) {
                alg_ = po.entry_[0].eltwise.alg;
            } else if (po.entry_[0].is_binary()) {
                alg_ = po.entry_[0].binary.alg;
            }
        }
    }

    rvv_postops_t() = default;

    status_t init(engine_t *engine, const post_ops_t &post_ops,
            const memory_desc_t &dst_md, int post_op_start_index = 0) {
        post_op_start_index_ = post_op_start_index;

        post_ops_t local_post_ops = post_ops;
        CHECK(local_post_ops.set_default_formats(&dst_md));
        dst_data_type_ = dst_md.data_type;

        if (dst_data_type_ != data_type::f32) return status::unimplemented;

        post_op_primitives_.clear();
        po_ = local_post_ops;

        for (int i = post_op_start_index_; i < local_post_ops.len(); i++) {
            auto &po = local_post_ops.entry_[i];

            if (po.is_binary()) {
                binary_desc_t po_desc;
                po_desc.primitive_kind = primitive_kind::binary;
                po_desc.alg_kind = po.binary.alg;
                po_desc.src_desc[0] = dst_md;
                po_desc.src_desc[1] = po.binary.src1_desc;
                po_desc.src_desc[2] = po.binary.src2_desc;
                po_desc.dst_desc = dst_md;

                auto empty_attr = dnnl_primitive_attr();
                primitive_desc_iterator_t it(engine,
                        reinterpret_cast<const op_desc_t *>(&po_desc),
                        &empty_attr, nullptr);
                if (++it == it.end()) return status::unimplemented;

                std::shared_ptr<primitive_desc_t> bin_pd = *it;
                std::shared_ptr<primitive_t> bin_prim;
                CHECK(bin_pd->create_primitive(bin_prim, engine));
                post_op_primitives_.push_back(bin_prim);

            } else {
                return status::unimplemented;
            }
        }

        return status::success;
    }

    static bool post_ops_ok(const post_ops_t &po) {
        if (po.len() == 0) return true;
        if (po.len() > 1) return false;

        const auto &e = po.entry_[0];
        if (!e.is_eltwise()) return false;

        switch (e.eltwise.alg) {
            case alg_kind::eltwise_relu: return true;
            default: return false;
        }
    }

    inline vfloat32m1_t apply(vfloat32m1_t v, size_t vl) const {
        switch (alg_) {
            case alg_kind::eltwise_relu: {
                vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
                return __riscv_vfmax_vv_f32m1(v, zero, vl);
            }
            default: return v;
        }
    }

    status_t execute(
            const exec_ctx_t &ctx, void *src, void *dst = nullptr) const;

private:
    alg_kind_t alg_ = alg_kind::undef;
    post_ops_t po_;
    int post_op_start_index_ = 0;
    data_type_t dst_data_type_ = data_type::undef;
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_POSTOPS_HPP
