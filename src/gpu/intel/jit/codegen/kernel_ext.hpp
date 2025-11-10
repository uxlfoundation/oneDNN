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

#ifndef GPU_INTEL_JIT_CODEGEN_KERNEL_EXT_HPP
#define GPU_INTEL_JIT_CODEGEN_KERNEL_EXT_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/generator.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <typename KernelT>
struct ir_generator_t : public generator_base_t {
    ir_generator_t(const kernel_desc_base_t &kernel_desc)
        : kernel_name_(kernel_desc.kernel_name()), kernel_desc_(kernel_desc) {}

    const char *kernel_name() const override { return kernel_name_.c_str(); }

    status_t get_kernel(
            compute::kernel_t &kernel, const intel::engine_t *engine) override {
        try {
            KernelT _kernel(kernel_desc_, engine);
            return _kernel.get_kernel(kernel, engine);
        } catch (ngen::out_of_registers_exception &) {
            return status::runtime_error;
        }
    }

private:
    std::string kernel_name_;
    const kernel_desc_base_t &kernel_desc_;
};

#define IR_TO_NGEN_GENERATOR_EMULATION_FORWARD(BaseGeneratorT) \
    using ir_to_ngen_generator_t<BaseGeneratorT>::emov; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::eadd; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::emul; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::eshl; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::eshr;

#define IR_TO_NGEN_GENERATOR_FORWARD(BaseGeneratorT) \
    NGEN_FORWARD_ELF(BaseGeneratorT::hardware) \
    IR_TO_NGEN_GENERATOR_EMULATION_FORWARD(BaseGeneratorT) \
    using ir_to_ngen_generator_t<BaseGeneratorT>::options; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::kernel_iface; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::generate_prologue; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::generate_epilogue; \
    using ir_to_ngen_generator_t<BaseGeneratorT>::ra;

class ir_kernel_t : public generator_base_t {
public:
    ir_kernel_t(const kernel_desc_base_t &desc, const impl::engine_t *engine,
            const debug_config_t &debug_config)
        : kernel_iface_(desc.kernel_name())
        , options_(desc.options(engine))
        , local_range_(desc.local_range())
        , debug_config_(debug_config) {
        desc.init_kernel_iface(kernel_iface_);
    }

    ir_kernel_t(const kernel::iface_t &kernel_iface,
            const kernel::options_t &options,
            const compute::range_t &local_range,
            const debug_config_t &debug_config)
        : kernel_iface_(kernel_iface)
        , options_(options)
        , local_range_(local_range)
        , debug_config_(debug_config) {}

    const kernel::options_t &options() const { return options_; }
    const kernel::iface_t &kernel_iface() const { return kernel_iface_; }
    void force_emulate64() { force_emulate64_ = true; }

    int peak_regs() const { return peak_regs_; }

    void generate_from_ir(const stmt_t &kernel_body,
            const walk_order_t *kernel_grid_walk_order = nullptr);

    const char *kernel_name() const override {
        return kernel_iface().kernel_name().c_str();
    }

    status_t get_kernel(
            compute::kernel_t &kernel, const intel::engine_t *engine) override {
        return generator_->get_kernel(kernel, engine);
    }

private:
    int thread_group_size() const {
        gpu_assert(local_range_);
        int local_size = 1;
        for (int i = 0; i < (int)local_range_.ndims(); i++) {
            local_size *= (int)local_range_[i];
        }
        return ir_utils::safe_divide(local_size, options_.simd());
    }

    kernel::iface_t kernel_iface_;
    kernel::options_t options_;
    compute::range_t local_range_;

    debug_config_t debug_config_;

    bool force_emulate64_ = false;
    int peak_regs_ = 0;

    std::unique_ptr<generator_base_t> generator_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
