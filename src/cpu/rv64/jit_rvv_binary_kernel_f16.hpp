#ifndef CPU_RV64_JIT_RVV_BINARY_KERNEL_F16_HPP
#define CPU_RV64_JIT_RVV_BINARY_KERNEL_F16_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_binary_kernel_f16_t : public jit_generator_t {
    struct call_params_t {
        const void *src0;
        const void *src1;
        const int8_t *src2;
        void *dst;
        dim_t len;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_binary_kernel_f16_t)

    explicit jit_rvv_binary_kernel_f16_t(alg_kind_t alg);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void compute_vector(const Xbyak_riscv::VReg &v_dst,
            const Xbyak_riscv::VReg &v_src0, const Xbyak_riscv::VReg &v_src1,
            const Xbyak_riscv::FReg &f_zero, const Xbyak_riscv::FReg &f_one);

    alg_kind_t alg_;
};

bool jit_rvv_binary_f16_supported(alg_kind_t alg);

void jit_rvv_binary_apply_f16(alg_kind_t alg, const void *src0,
        const void *src1, const int8_t *src2, void *dst, dim_t len);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
