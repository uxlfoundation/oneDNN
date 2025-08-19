#ifndef CPU_RV64_RVV_POSTOPS_HPP
#define CPU_RV64_RVV_POSTOPS_HPP

#include "common/primitive_attr.hpp"
#include "common/utils.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_postops_t {
    rvv_postops_t(const post_ops_t &po) {
        is_supported_ = false;

        if (po.len() == 1 && po.entry_[0].is_eltwise()) {
            const auto &e = po.entry_[0];
            if (e.eltwise.alg == alg_kind::eltwise_relu) {
                is_supported_ = true;
            }
        }
    }

    inline bool has_postops() const { return is_supported_; }

    inline vfloat32m1_t apply(vfloat32m1_t v, size_t vl) const {
        if (is_supported_) {
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
            return __riscv_vfmax_vv_f32m1(v, zero, vl);
        }
        return v;
    }

private:
    bool is_supported_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_POSTOPS_HPP
