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

#ifndef CPU_X64_MATMUL_POSTOPS_ESTIMATOR_HPP
#define CPU_X64_MATMUL_POSTOPS_ESTIMATOR_HPP

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

class postops_estimator_t : public jit_generator_t {
public:
    static size_t estimate_insts_per_cacheline(
            memory_desc_t &dst_md, primitive_attr_t &attr) {
        postops_estimator_t post_ops_gen(dst_md, attr);
        post_ops_gen.generate();
        size_t code_size = post_ops_gen.getSize();
        size_t estimated_vec_code_size = (size_t)nstl::max(
                (int)code_size - (int)mean_none_vec_code_bytes, 0);
        size_t estimated_vec_insts
                = estimated_vec_code_size / mean_vec_inst_bytes;
        return estimated_vec_insts;
    }

private:
    using po_injector_t = injector::jit_uni_postops_injector_base_t<Xbyak::Zmm>;
    std::unique_ptr<po_injector_t> postops_injector_;
    static constexpr size_t mean_none_vec_code_bytes = 8;
    static constexpr size_t mean_vec_inst_bytes = 7;

    postops_estimator_t(memory_desc_t &dst_md, primitive_attr_t &attr)
        : jit_generator_t("dummy_generator") {
        auto dsc = memory_desc_wrapper(dst_md);
        const dnnl::impl::cpu::x64::binary_injector::rhs_arg_static_params_t
                rhs_sp(static_cast<size_t>(Xbyak::Zmm(1).getIdx()), this->r14,
                        this->r15, this->r13, false, false,
                        static_cast<size_t>(0), static_cast<size_t>(0), dsc,
                        static_cast<size_t>(0), Xbyak::Opmask(0), false);

        const dnnl::impl::cpu::x64::binary_injector::static_params_t bsp(
                this->param1,
                dnnl::impl::cpu::x64::binary_injector::
                        get_all_strategies_supported_by_injector(),
                rhs_sp, nullptr, nullptr);

        dnnl::impl::cpu::x64::eltwise_injector::static_params_t esp;
        esp.preserve_vmm = false;
        esp.preserve_p_table = false;

        auto st = safe_ptr_assign(postops_injector_,
                po_injector_t::create(this,
                        impl::cpu::x64::cpu_isa_t::avx512_core_amx,
                        attr.post_ops_, bsp, esp));
        if (st != status::success) {
            assert(!"postops_injector creation failed");
        }
    }

    const char *name() const final { return nullptr; }
    const char *source_file() const final { return nullptr; }

    void generate() final { postops_injector_->compute_vector(0); }
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif