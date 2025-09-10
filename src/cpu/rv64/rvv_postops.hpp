/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_postops_t {
    rvv_postops_t(const post_ops_t &po)
        : alg_(po.len() > 0 ? po.entry_[0].eltwise.alg : alg_kind::undef) {
        assert(po.len() <= 1 && "rvv_postops_t supports at most one post-op");
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

private:
    alg_kind_t alg_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_POSTOPS_HPP