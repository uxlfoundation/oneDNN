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

#include "gpu/intel/jit/codegen/kernel_ext.hpp"
#include "gpu/intel/logging.hpp"

namespace gemmstone {
namespace dsl {
namespace ir {
extern ngen::NEOInterfaceHandler generate_ngen_interface(
        const dsl::kernel::iface_t &kernel_iface,
        const dsl::kernel::options_t &options, const stmt_t &kernel_body);

template <typename GeneratorT>
void convert_ir_to_ngen(const stmt_t &body, GeneratorT &host,
        const walk_order_t *kernel_grid_walk_order = nullptr);

template <ngen::HW hw>
using gen_t = ir_to_ngen_generator_t<
        dnnl::impl::gpu::intel::jit::ngen_code_generator_t<hw>>;

REG_XELP_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::XeLP>>(
        const stmt_t &body, gen_t<ngen::HW::XeLP> &host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHP_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::XeHP>>(
        const stmt_t &body, gen_t<ngen::HW::XeHP> &host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHPG_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::XeHPG>>(
        const stmt_t &body, gen_t<ngen::HW::XeHPG> &host,
        const walk_order_t *kernel_grid_walk_order));
REG_XEHPC_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::XeHPC>>(
        const stmt_t &body, gen_t<ngen::HW::XeHPC> &host,
        const walk_order_t *kernel_grid_walk_order));
REG_XE2_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::Xe2>>(
        const stmt_t &body, gen_t<ngen::HW::Xe2> &host,
        const walk_order_t *kernel_grid_walk_order));
REG_XE3_ISA(extern template void convert_ir_to_ngen<gen_t<ngen::HW::Xe3>>(
        const stmt_t &body, gen_t<ngen::HW::Xe3> &host,
        const walk_order_t *kernel_grid_walk_order));

} // namespace ir
} // namespace dsl
} // namespace gemmstone

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <ngen::HW hw>
using gen_t = ir::ir_to_ngen_generator_t<ngen_code_generator_t<hw>>;

template <typename GeneratorT>
std::string get_ngen_str(const stmt_t &body, GeneratorT *host,
        const walk_order_t *kernel_grid_walk_order) {
#ifdef NGEN_ASM
    ir::ir_to_ngen_generator_t<ir::ngen_asm_code_generator_with_interface_t>
            host_asm(host->kernel_iface(), host->options(), {});
    host_asm.set_interface(host->getInterface());

    try {
        ir::convert_ir_to_ngen(body, host_asm, kernel_grid_walk_order);
        return host_asm.str();
    } catch (std::runtime_error &e) {
        return "IR to nGEN Exception: " + std::string(e.what());
    }
#else
    return "";
#endif
}

template <typename GeneratorT>
void generate_from_ir(const stmt_t &kernel_body, GeneratorT &host,
        const walk_order_t *kernel_grid_walk_order, int &peak_regs) {
    gpu_trace() << get_ngen_str(kernel_body, &host, kernel_grid_walk_order);
    ir::convert_ir_to_ngen(kernel_body, host, kernel_grid_walk_order);
#ifdef DNNL_DEV_MODE
    peak_regs = host.ra().get_peak_regs();
#endif
}

void ir_kernel_t::generate_from_ir(
        const stmt_t &kernel_body, const walk_order_t *kernel_grid_walk_order) {
    gpu_assert(!generator_)
            << "ir_kernel_t::generate_from_ir() was called already.";

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            kernel_iface_, options_, kernel_body);

    if (local_range_) {
        size_t max_slm_size = compute::device_info_t::max_slm_size_per_tg(
                convert_ngen_arch_to_dnnl(options_.hw()), thread_group_size(),
                options_.regs() > 128);
        if (interface.getSLMSize() > max_slm_size) {
            gpu_trace() << "SLM size limit exceeded: " << interface.getSLMSize()
                        << " > " << max_slm_size;
            gpu_except_not_implemented("SLM size limit is exceeded.");
        }
    }

#define GPU_HW_CASE(hw) \
    auto gen = gen_t<(hw)>(kernel_iface_, options_, debug_config_); \
    gen.setInterface(ir::generate_ngen_interface( \
            kernel_iface_, options_, kernel_body)); \
    if (force_emulate64_) gen.force_emulate64(); \
    jit::generate_from_ir( \
            kernel_body, gen, kernel_grid_walk_order, peak_regs_); \
    generator_.reset(new generator_t<(hw)>(static_cast<gen_t<(hw)> &&>(gen)));

    GPU_HW_SWITCH(options_.hw().ngen_hw());

#undef GPU_HW_CASE
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
