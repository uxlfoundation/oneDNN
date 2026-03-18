/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/ze/kernel.hpp"
#include "gpu/intel/ze/engine.hpp"
#include "gpu/intel/ze/stream.hpp"
#include "gpu/intel/ze/utils.hpp"

#include "xpu/ze/context.hpp"
#include "xpu/ze/memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

// This class is to get around std::make_shared requirement to have a public
// constructor. We keep the original constructor as private but expose it here
// to use with std::make_shared.
class kernel_compat_t : public kernel_t {
public:
    template <typename... Args>
    kernel_compat_t(Args &&...args) : kernel_t(std::forward<Args>(args)...) {}
};

status_t kernel_t::make(compute::kernel_t &compute_kernel,
        const std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>> &amodule,
        ze_kernel_handle_t akernel, const std::string &kernel_name) {
    compute_kernel = compute::kernel_t(
            std::make_shared<kernel_compat_t>(amodule, akernel, kernel_name));
    return status::success;
}

kernel_t::kernel_t(
        const std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>> &amodule,
        ze_kernel_handle_t akernel, const std::string &kernel_name)
    : module_(amodule), kernel_(akernel), kernel_name_(kernel_name) {}

status_t kernel_t::check_alignment(
        const compute::kernel_arg_list_t &arg_list) const {
    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        if (!arg.is_global()) continue;

        auto *mem_storage = static_cast<const memory_storage_t *>(arg.value());
        if (!*mem_storage) continue;

        CHECK(compute::kernel_impl_t::check_alignment(
                mem_storage->data_handle(), i));
    }

    return status::success;
}

status_t kernel_t::set_arg(
        int arg_index, size_t arg_size, const void *arg_value) const {
    return xpu::ze::zeKernelSetArgumentValue(
            kernel_, arg_index, arg_size, arg_value);
}

status_t kernel_t::parallel_for(impl::stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list, const xpu::event_t &deps,
        xpu::event_t &out_dep) {
    CHECK(check_scalar_arguments(arg_list));
    CHECK(check_alignment(arg_list));

    auto *ze_stream = utils::downcast<stream_t *>(&stream);
    auto *ze_engine = utils::downcast<engine_t *>(stream.engine());
    auto *ze_device_info = ze_engine->device_info();

    const size_t pointer_size = ze_device_info->device_address_bits() / 8;

    size_t param_bytes = 0;
    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            if (!mem_storage->is_null()) {
                auto *ze_mem_storage
                        = utils::downcast<const xpu::ze::memory_storage_t *>(
                                mem_storage);
                auto ze_ctx
                        = utils::downcast<engine_t *>(ze_mem_storage->engine())
                                  ->context();
                if (ze_engine->context() != ze_ctx) {
                    VERROR(primitive, gpu,
                            "mismatched Level Zero context for "
                            "primitive/memory");
                    return status::invalid_arguments;
                }

                void *ptr = ze_mem_storage->ptr();
                CHECK(set_arg(i, pointer_size, &ptr));
                param_bytes += pointer_size;
            } else {
                CHECK(set_arg(i, pointer_size, nullptr));
                param_bytes += pointer_size;
            }
        } else if (arg.is_local()) {
            CHECK(set_arg(i, arg.size(), arg.value()));
            param_bytes += pointer_size;
        } else {
            CHECK(set_arg(i, arg.size(), arg.value()));
            param_bytes += arg.size();
        }
    }
    if (param_bytes > ze_device_info->max_kernel_param_size()) {
        VERROR(primitive, gpu,
                "parameter bytes requirements greater than device supports");
        return status::invalid_arguments;
    }

    if (range.is_zero()) { return status::success; }

    std::vector<uint32_t> global_size(3, 1);
    switch (range.global_range().ndims()) {
        case 3: global_size[2] = static_cast<uint32_t>(range.global_range()[2]);
        case 2: global_size[1] = static_cast<uint32_t>(range.global_range()[1]);
        case 1:
            global_size[0] = static_cast<uint32_t>(range.global_range()[0]);
            break;
        default:
            VERROR(primitive, gpu,
                    "incorrect number of global range dimensions");
            return status::invalid_arguments;
    }

    std::vector<uint32_t> group_size(3, 1);
    if (range.local_range()) {
        switch (range.local_range().ndims()) {
            case 3:
                group_size[2] = static_cast<uint32_t>(range.local_range()[2]);
            case 2:
                group_size[1] = static_cast<uint32_t>(range.local_range()[1]);
            case 1:
                group_size[0] = static_cast<uint32_t>(range.local_range()[0]);
                break;
            default:
                VERROR(primitive, gpu,
                        "incorrect number of local range dimensions");
                return status::invalid_arguments;
        }
    } else {
        CHECK(xpu::ze::zeKernelSuggestGroupSize(kernel_, global_size[0],
                global_size[1], global_size[2], &group_size[0], &group_size[1],
                &group_size[2]));
    }

    for (size_t i = 0; i < global_size.size(); i++) {
        if (global_size[i] % group_size[i] != 0) {
            VERROR(primitive, gpu, "only uniform work-groups are supported");
            return status::invalid_arguments;
        }
    }

    CHECK(xpu::ze::zeKernelSetGroupSize(
            kernel_, group_size[0], group_size[1], group_size[2]));
    ze_group_count_t group_count = {global_size[0] / group_size[0],
            global_size[1] / group_size[1], global_size[2] / group_size[2]};

    const auto &ze_deps = xpu::ze::event_t::from(deps);
    ze_event_handle_t out_event = ze_stream->create_event();

    CHECK(xpu::ze::zeCommandListAppendLaunchKernel(ze_stream->list(), kernel_,
            &group_count, out_event, ze_deps.size(), ze_deps.data()));

    if (out_event) xpu::ze::event_t::from(out_dep).append(out_event);
    if (stream.is_profiling_enabled()) {
        ze_stream->profiler().register_event(
                utils::make_unique<xpu::ze::event_t>(out_event));
    }

    return status::success;
}

status_t kernel_t::get_kernel_binary(xpu::binary_t &binary) const {
    return ze::get_kernel_binary(kernel_, binary);
}

status_t kernel_t::dump() const {
    xpu::binary_t binary;
    CHECK(get_kernel_binary(binary));

    return gpu_utils::dump_kernel_binary(binary, kernel_name_);
}

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
