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

#include <memory>

#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

// The postops estimator JIT-compiles post-operation code for a single
// cache line result produced by the GEMM. The size of the generated code
// serves as a metric to help blocking heuristics account for the impact
// of post-operations on execution performance. Note: the generated code
// is not executed.
//
// The estimate covers the post-op chain only, which includes the up-convert
// of fp8 binary post-op data. It deliberately does not model the conversion
// of the accumulators to the destination data type: the caller accounts for
// that separately with a fixed per-cache-line cost, see
// matmul_amx_blocking_params_macro_t::calculate_avx_insts_per_cache_line().

class postops_estimator_t : public jit_generator_t {
public:
    static status_t estimate_insts_per_cacheline(memory_desc_t &dst_md,
            primitive_attr_t &attr, cpu_isa_t isa, int &estimated_vec_insts) {
        postops_estimator_t post_ops_gen(dst_md, attr, isa);
        status_t res = post_ops_gen.check_status();
        if (res != status::success) return res;
        post_ops_gen.generate();
        res = post_ops_gen.check_status();
        if (res != status::success) return res;

        size_t code_size = post_ops_gen.getSize();
        const size_t estimated_vec_code_size
                = code_size > mean_none_vec_code_bytes
                ? code_size - mean_none_vec_code_bytes
                : 0;
        estimated_vec_insts = static_cast<int>(
                estimated_vec_code_size / mean_vec_inst_bytes);
        return status::success;
    }

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<Xbyak::Zmm>;

    static constexpr size_t mean_none_vec_code_bytes = 8;
    static constexpr size_t mean_vec_inst_bytes = 7;

    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;
    std::unique_ptr<po_injector_t> postops_injector_;

    postops_estimator_t(
            memory_desc_t &dst_md, primitive_attr_t &attr, cpu_isa_t isa)
        : jit_generator_t("dummy_generator", isa)
        , f8_e5m2_cvt_(utils::make_unique<fp8_conversion_e5m2_t>(
                  this, xmm1, xmm2, xmm3, k1, r8))
        , f8_e4m3_cvt_(utils::make_unique<fp8_conversion_e4m3_t>(
                  this, xmm1, xmm2, xmm3, xmm4, xmm5, r8)) {
        auto dsc = memory_desc_wrapper(dst_md);
        const dnnl::impl::cpu::x64::binary_injector::rhs_arg_static_params_t
                rhs_sp(Xbyak::Zmm(1).getIdx(), this->r14, this->r15, this->r13,
                        false, false, static_cast<size_t>(0),
                        static_cast<size_t>(0), dsc, static_cast<size_t>(0),
                        Xbyak::Opmask(0), false);

        const dnnl::impl::cpu::x64::binary_injector::static_params_t bsp(
                this->param1,
                dnnl::impl::cpu::x64::binary_injector::
                        get_all_strategies_supported_by_injector(),
                rhs_sp, f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get());

        dnnl::impl::cpu::x64::eltwise_injector::static_params_t esp;
        esp.preserve_vmm = false;
        esp.preserve_p_table = false;

        postops_injector_ = utils::make_unique<po_injector_t>(this,
                attr.post_ops_, bsp, esp,
                // The estimate leaves sum out, as it always has.
                /* inject_sum = */ false);
    }

    const char *name() const final {
        assert(!"postops_estimator_t name function is not expected to be "
                "called. The jitted code is not to be registered or used");
        return nullptr;
    }
    const char *source_file() const final {
        assert(!"postops_estimator_t source_file function is not expected to be"
                " called. The jitted code is not to be registered or used");
        return nullptr;
    }

    status_t check_status() {
        if (!f8_e5m2_cvt_ || !f8_e4m3_cvt_ || !postops_injector_)
            return status::out_of_memory;
        int err_code = Xbyak::GetError();
        if (err_code == Xbyak::ERR_CANT_ALLOC) return status::out_of_memory;
        if (err_code != Xbyak::ERR_NONE) return status::runtime_error;
        return status::success;
    }

    // The lookup tables of the fp8 converters are deliberately not emitted:
    // only the size of the code is measured, so the table bytes would be
    // counted as post-op instructions. The instructions referencing them are
    // emitted and measured, which is what the estimate needs.
    void generate() final { postops_injector_->compute_vector(0); }
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
