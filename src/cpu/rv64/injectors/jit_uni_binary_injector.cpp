/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace binary_injector {

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        case binary_add:
        case binary_sub:
        case binary_mul:
        case binary_div:
        case binary_max:
        case binary_min: return true;
        default: return false;
    }
}

} // namespace binary_injector

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; i++)
        compute_body(Vmm(i));
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_body(const Vmm &dst) {
    using namespace alg_kind;
    const bool scalar = bcast_ == binary_injector::broadcast_t::scalar;

    // Load the rhs operand: a single broadcast value, or one value per active
    // lane. In indirect mode this binary's base lives in the pointer array at
    // rhs_addr_[arg_idx_] and per-element lanes are contiguous at base + off_;
    // in direct mode the host has positioned rhs_addr_ at the first active lane.
    if (indirect_) {
        h_->ld(gpr_, rhs_addr_, arg_idx_ * (int)sizeof(void *));
        if (scalar)
            h_->flw(f_rhs_, gpr_, 0);
        else {
            h_->add(gpr_, gpr_, off_);
            h_->vle32_v(v_rhs_, gpr_);
        }
    } else if (scalar)
        h_->flw(f_rhs_, rhs_addr_, 0);
    else if (strided_)
        h_->vlse32_v(v_rhs_, rhs_addr_, rhs_stride_);
    else
        h_->vle32_v(v_rhs_, rhs_addr_);

    // dst = dst OP rhs (src0 is the accumulator, src1 the rhs).
    switch (alg_) {
        case binary_add:
            if (scalar)
                h_->vfadd_vf(dst, dst, f_rhs_);
            else
                h_->vfadd_vv(dst, dst, v_rhs_);
            break;
        case binary_sub:
            if (scalar)
                h_->vfsub_vf(dst, dst, f_rhs_);
            else
                h_->vfsub_vv(dst, dst, v_rhs_);
            break;
        case binary_mul:
            if (scalar)
                h_->vfmul_vf(dst, dst, f_rhs_);
            else
                h_->vfmul_vv(dst, dst, v_rhs_);
            break;
        case binary_div:
            if (scalar)
                h_->vfdiv_vf(dst, dst, f_rhs_);
            else
                h_->vfdiv_vv(dst, dst, v_rhs_);
            break;
        case binary_max:
            if (scalar) h_->vfmv_v_f(v_rhs_, f_rhs_);
            h_->vmflt_vv(VReg(0), dst, v_rhs_);
            h_->vmerge_vvm(dst, dst, v_rhs_);
            h_->vmfne_vv(VReg(0), v_rhs_, v_rhs_);
            h_->vmerge_vvm(dst, dst, v_rhs_);
            break;
        case binary_min:
            if (scalar) h_->vfmv_v_f(v_rhs_, f_rhs_);
            h_->vmflt_vv(VReg(0), v_rhs_, dst);
            h_->vmerge_vvm(dst, dst, v_rhs_);
            h_->vmfne_vv(VReg(0), v_rhs_, v_rhs_);
            h_->vmerge_vvm(dst, dst, v_rhs_);
            break;
        default: assert(!"unsupported binary alg"); break;
    }
}

template struct jit_uni_binary_injector_t<v>;
template struct jit_uni_binary_injector_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
