/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP
#define GPU_INTEL_JIT_CODEGEN_CODEGEN_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <typename ngen_generator_t>
void convert_ir_to_ngen(const stmt_t &body, ngen_generator_t *host,
        const walk_order_t *kernel_grid_walk_order = nullptr);

REG_XELP_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeLP> *host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHP_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHP> *host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHPG_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHPG> *host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHPC_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::XeHPC> *host,
        const walk_order_t *kernel_grid_walk_order));
REG_XE2_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Xe2> *host,
        const walk_order_t *kernel_grid_walk_order));
REG_XE3_ISA(extern template void convert_ir_to_ngen(const stmt_t &body,
        ir_kernel_t<ngen::HW::Xe3> *host,
        const walk_order_t *kernel_grid_walk_order));

struct elf_binary_t {
    std::vector<uint8_t> data;
};

struct compile_ctx_t {
    compile_ctx_t(std::string kernel_name, kernel_iface_t kernel_iface,
            exec_config_t exec_config, int thread_group_size)
        : kernel_name_(kernel_name)
        , kernel_iface_(kernel_iface)
        , exec_config_(exec_config)
        , thread_group_size_(thread_group_size) {}

    const std::string &kernel_name() const { return kernel_name_; }
    const kernel_iface_t &kernel_iface() const { return kernel_iface_; }
    const exec_config_t &exec_config() const { return exec_config_; }
    ngen::HW hw() const { return exec_config_.hw().ngen_hw(); }
    const ngen::Product &product() const { return exec_config_.hw().product(); }

    template <ngen::HW hw>
    void setup_interface(
            const stmt_t &kernel_body, ngen::ELFCodeGenerator<hw> &host) const {
        host.externalName(kernel_name_);
        host.requireLocalID(3);
        host.requireLocalSize();
        host.requireGRF(exec_config_.regs());
        host.requireSIMD(exec_config_.simd());
        host.requireBarrier();
        auto setup_flags = get_setup_flags(kernel_body);
        if (setup_flags.has_dpas) host.requireDPAS();
        if (setup_flags.has_send_atomics) host.requireGlobalAtomics();

        for (int i = 0; i < kernel_iface_.nargs(); i++) {
            auto &name = kernel_iface_.arg_name(i);
            auto &type = kernel_iface_.arg_type(i);
            if (type.is_ptr()) {
                host.newArgument(name, ngen::ExternalArgumentType::GlobalPtr,
                        ngen::GlobalAccessType::Stateless);
            } else {
                host.newArgument(name, to_ngen(type));
            }
        }

        if (!kernel_body.is_empty() && thread_group_size_) {
            int slm_size = alloc_manager_t(kernel_body)
                                   .total_size(alloc_kind_t::slm);
            int max_slm_size = compute::device_info_t::max_slm_size_per_tg(
                    convert_ngen_arch_to_dnnl(hw), thread_group_size_,
                    exec_config_.regs() > 128);
            if (slm_size > max_slm_size) {
                // TODO: Use status code for this check.
                gpu_except_not_implemented("SLM size limit is exceeded.");
            }
            host.requireSLM(slm_size);
        }

        host.finalizeInterface();
    }

private:
    std::string kernel_name_;
    kernel_iface_t kernel_iface_;
    exec_config_t exec_config_;
    int thread_group_size_;
};

elf_binary_t lower_ir(const stmt_t &body, const compile_ctx_t &elf_iface);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
