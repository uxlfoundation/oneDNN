/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/conv/jit/v2/primitive_plan.hpp"

#include "gpu/intel/conv/jit/zero_out.hpp"
#include "gpu/intel/reorder/jit/kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {
namespace v2 {

status_t primitive_init_plan_t::create_exec_plan(
        primitive_exec_plan_t &exec_plan, primitive_t *primitive,
        impl::engine_t *engine) const {
    // Zero-out required buffers.
    for (auto &b : buf_entries_) {
        if (!b.zero_out) continue;
        CHECK(add_zero_out_kernel(exec_plan, b, primitive, engine));
    }
    // Pre-reorder.
    for (auto &a : buf_entries_) {
        if (a.is_user()) continue;
        for (auto &b : buf_entries_) {
            if (a.user_name == b.name) {
                if (b.is_user_input)
                    CHECK(add_reorder_kernel(
                            exec_plan, b, a, primitive, engine));
                break;
            }
        }
    }
    for (auto &e : kernel_entries_) {
        CHECK(add_kernel(exec_plan, *e.desc, *e.params, primitive, engine));
    }
    // Post-reorder.
    for (auto &a : buf_entries_) {
        if (a.is_user()) continue;
        for (auto &b : buf_entries_) {
            if (a.user_name == b.name) {
                if (b.is_user_output)
                    CHECK(add_reorder_kernel(
                            exec_plan, a, b, primitive, engine));
                break;
            }
        }
    }
    return status::success;
}

primitive_init_plan_t::buffer_entry_t primitive_init_plan_t::find_buf(
        const std::string &name) const {
    for (auto &b : buf_entries_) {
        if (b.name == name) return b;
    }
    return buffer_entry_t();
}

kernel_info_t primitive_init_plan_t::create_kernel_info(
        const kernel_desc_base_t &desc,
        const std::unordered_map<std::string, std::string> &buf_map) const {
    dsl::kernel::iface_t iface(desc.kernel_name());
    desc.init_kernel_iface(iface);
    kernel_info_t info;
    for (size_t i = 0; i < iface.nargs(); i++) {
        auto &var = iface[i];
        auto &name = var.as<var_t>().name;
        auto buf = find_buf(buf_map.count(name) == 0 ? name : buf_map.at(name));
        if (!buf) {
            info.register_immediate_arg(var);
        } else if (buf.is_user()) {
            info.register_user_arg(var, buf.arg_key, buf.is_user_input);
        } else {
            info.register_scratchpad_arg(var, buf.arg_key, /*is_input=*/false,
                    size_bytes(buf.layout));
        }
    }
    return info;
}

status_t primitive_init_plan_t::add_kernel(primitive_exec_plan_t &exec_plan,
        const kernel_desc_base_t &desc, const kernel_params_base_t &params,
        primitive_t *primitive, impl::engine_t *engine,
        const std::unordered_map<std::string, std::string> &buf_map) const {
    compute::kernel_t kernel;
    CHECK(desc.create_kernel(kernel, primitive, engine));
    auto kernel_info = create_kernel_info(desc, buf_map);
    desc.init_kernel_info(kernel_info, params, engine);
    exec_plan.add_kernel(kernel, kernel_info);
    return status::success;
}

status_t primitive_init_plan_t::add_zero_out_kernel(
        primitive_exec_plan_t &exec_plan, const buffer_entry_t &buf,
        primitive_t *primitive, impl::engine_t *engine) const {
    auto desc = std::make_shared<conv::jit::zero_out_kernel_desc_t>(
            regs_, simd_, dpas_);
    auto params = std::make_shared<conv::jit::zero_out_kernel_params_t>(
            size_bytes(buf.layout));
    std::unordered_map<std::string, std::string> buf_map;
    buf_map["ptr"] = buf.name;
    return add_kernel(exec_plan, *desc, *params, primitive, engine, buf_map);
}

status_t primitive_init_plan_t::add_reorder_kernel(
        primitive_exec_plan_t &exec_plan, const buffer_entry_t &src,
        const buffer_entry_t &dst, primitive_t *primitive,
        impl::engine_t *engine) const {
    kernel_info_t kernel_info;
    for (auto *e : {&src, &dst}) {
        auto buf_var = var_t::make(
                dsl::type_t::byte(dsl::type::attr_t::ptr), e->name);
        if (e->is_user()) {
            kernel_info.register_user_arg(
                    buf_var, e->arg_key, /*is_input=*/e == &src);
        } else {
            kernel_info.register_scratchpad_arg(buf_var, e->arg_key,
                    /*is_input=*/e == &src, size_bytes(e->layout));
        }
    }
    dsl::kernel::options_t options(make_ir_hw(engine), regs_, simd_);
    options.set_require_dpas(dpas_);
    reorder::jit::config_t cfg(options, src.layout, dst.layout);
    kernel_info.set_nd_range(cfg.nd_range());
    auto kernel = make_kernel<reorder::jit::kernel_t>(primitive,
            /*register_kernel=*/true, engine, cfg, "reorder", kernel_info);
    exec_plan.add_kernel(kernel, kernel_info);
    return status::success;
}

} // namespace v2
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
