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

#include "gpu/intel/jit/ir/include/kernel.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "ngen_interface.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

kernel::iface_t::iface_t(const ngen::InterfaceHandler &iface)
    : kernel_name_(iface.getExternalName()) {
    for (unsigned int i = 0; i < iface.numAssignments(); i++) {
        auto &a = iface.getAssignment(i);
        if (a.exttype == ngen::ExternalArgumentType::Scalar) {
            register_arg(a.name, type_t(a.type));
        } else if (a.exttype == ngen::ExternalArgumentType::GlobalPtr) {
            register_arg(a.name, type_t::byte(type::attr_t::ptr));
        } else if (a.exttype == ngen::ExternalArgumentType::LocalPtr) {
            register_arg(a.name,
                    type_t::byte(type::attr_t::ptr | type::attr_t::slm));
        } else {
            gpu_assert(false) << "Unimplemented";
        }
    }
}

void kernel::iface_t::register_arg(
        const std::string &name, const type_t &type) {
    register_arg(var_t::make(type, name));
}

const std::string &kernel::iface_t::arg_t::name() const {
    return var.as<var_t>().name;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
