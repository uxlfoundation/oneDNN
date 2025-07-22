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

#include "gpu/intel/l0/kernel.hpp"
#include "gpu/intel/l0/context.hpp"
#include "gpu/intel/l0/engine.hpp"
#include "gpu/intel/l0/memory_storage.hpp"
#include "gpu/intel/l0/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

module_wrapper_t::module_wrapper_t(ze_module_handle_t module)
    : module_(module) {};

module_wrapper_t::~module_wrapper_t() {
    func_zeModuleDestroy(module_);
};

module_wrapper_t::operator ze_module_handle_t() const {
    return module_;
}

kernel_wrapper_t::kernel_wrapper_t(
        const char *kernel_name, const ze_module_handle_t module_ptr)
    : event_pool_(nullptr), event_(nullptr) {
    ze_kernel_desc_t kernel_desc = {};
    kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernel_desc.pNext = nullptr;
    kernel_desc.flags = 0;
    kernel_desc.pKernelName = kernel_name;

    func_zeKernelCreate(module_ptr, &kernel_desc, &kernel_);
}

kernel_wrapper_t::~kernel_wrapper_t() {
    if (event_) {
        func_zeEventHostSynchronize(event_, UINT64_MAX);
        func_zeEventDestroy(event_);
    }
    if (event_pool_) func_zeEventPoolDestroy(event_pool_);
    func_zeKernelDestroy(kernel_);
}

kernel_wrapper_t::operator ze_kernel_handle_t() const {
    return kernel_;
}

status_t kernel_wrapper_t::set_arg(
        int arg_index, size_t arg_size, const void *arg_value) {
    return func_zeKernelSetArgumentValue(
            kernel_, arg_index, arg_size, arg_value);
}

ze_event_handle_t kernel_wrapper_t::create_out_event(
        const ze_context_handle_t context_ptr) {
    ze_event_pool_desc_t event_pool_desc = {};
    event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    event_pool_desc.pNext = nullptr;
    event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    event_pool_desc.count = 1;
    func_zeEventPoolCreate(
            context_ptr, &event_pool_desc, 0, nullptr, &event_pool_);

    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.pNext = nullptr;
    event_desc.index = 0;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    func_zeEventCreate(event_pool_, &event_desc, &event_);

    return event_;
}

// This class is to get around std::make_shared requirement to have a public
// constructor. We keep the original constructor as private but expose it here
// to use with std::make_shared.
class kernel_compat_t : public kernel_t {
public:
    template <typename... Args>
    kernel_compat_t(Args &&...args) : kernel_t(std::forward<Args>(args)...) {}
};

status_t kernel_t::make(compute::kernel_t &compute_kernel,
        const std::shared_ptr<module_wrapper_t> module_ptr,
        const ze_kernel_handle_t kernel_ptr) {
    compute_kernel = compute::kernel_t(
            std::make_shared<kernel_compat_t>(module_ptr, kernel_ptr));
    return status::success;
}

kernel_t::kernel_t(const std::shared_ptr<module_wrapper_t> module_ptr,
        const ze_kernel_handle_t kernel_ptr)
    : module_(module_ptr), main_kernel_(kernel_ptr) {
    l0::get_kernel_name(main_kernel_, kernel_name_);
    l0::get_kernel_binary(main_kernel_, kernel_binary_);
}

kernel_t::~kernel_t() {
    func_zeKernelDestroy(main_kernel_);
}

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

kernel_wrapper_t *kernel_t::get() {
    auto id = std::this_thread::get_id();
    {
        utils::lock_read_t lock_read(mutex_);
        auto it = kernels_.find(id);
        if (it != kernels_.end()) { return it->second.get(); }
    }

    // No copy for this thread, clone the original kernel and save the
    // copy.
    auto new_kernel = std::unique_ptr<kernel_wrapper_t>(
            new kernel_wrapper_t(kernel_name_.c_str(), *(module_.get())));

    utils::lock_write_t lock_write(mutex_);
    auto ret = kernels_.emplace(id, std::move(new_kernel));
    return ret.first->second.get();
}

status_t kernel_t::parallel_for(impl::stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list, const xpu::event_t &deps,
        xpu::event_t &out_dep) {
    CHECK(check_scalar_arguments(arg_list));
    CHECK(check_alignment(arg_list));

    auto l0_stream = utils::downcast<stream_t *>(&stream);
    auto l0_engine = l0_stream->l0_engine();
    auto l0_device_info = l0_engine->device_info();

    kernel_wrapper_t *kernel = get();
    const size_t pointer_size = l0_device_info->device_address_bits() / 8;

    size_t param_bytes = 0;
    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            if (!mem_storage->is_null()) {
                auto memory_storage_ctx
                        = utils::downcast<engine_t *>(mem_storage->engine())
                                  ->context();
                if (l0_engine->context() != memory_storage_ctx) {
                    VERROR(primitive, gpu,
                            "mismatched Level Zero context for "
                            "primitive/memory");
                    return status::invalid_arguments;
                }

                void *ptr = mem_storage->ptr();
                CHECK(kernel->set_arg(i, pointer_size, &ptr));
                param_bytes += pointer_size;
            } else {
                CHECK(kernel->set_arg(i, pointer_size, nullptr));
                param_bytes += pointer_size;
            }
        } else if (arg.is_local()) {
            CHECK(kernel->set_arg(i, arg.size(), arg.value()));
            param_bytes += pointer_size;
        } else {
            CHECK(kernel->set_arg(i, arg.size(), arg.value()));
            param_bytes += arg.size();
        }
    }
    if (param_bytes > l0_device_info->max_kernel_param_size()) {
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
        CHECK(func_zeKernelSuggestGroupSize(*kernel, global_size[0],
                global_size[1], global_size[2], &group_size[0], &group_size[1],
                &group_size[2]));
    }

    for (size_t i = 0; i < global_size.size(); i++) {
        if (global_size[i] % group_size[i] != 0) {
            VERROR(primitive, gpu, "only uniform work-groups are supported");
            return status::invalid_arguments;
        }
    }

    CHECK(func_zeKernelSetGroupSize(
            *kernel, group_size[0], group_size[1], group_size[2]));
    ze_group_count_t group_count = {global_size[0] / group_size[0],
            global_size[1] / group_size[1], global_size[2] / group_size[2]};

    std::vector<ze_event_handle_t> l0_deps
            = utils::downcast<const event_t *>(&deps)->events_;
    std::vector<ze_event_handle_t> l0_out_deps
            = utils::downcast<const event_t *>(&out_dep)->events_;

    ze_event_handle_t out_event
            = kernel->create_out_event(l0_engine->context());
    CHECK(func_zeCommandListAppendLaunchKernel(l0_stream->list(), *kernel,
            &group_count, out_event, l0_deps.size(),
            l0_deps.size() ? l0_deps.data() : nullptr));
    l0_out_deps.push_back(out_event);

    if (stream.is_profiling_enabled()) {
        l0_stream->profiler().register_event(
                utils::make_unique<event_t>(std::move(out_event)));
    }

    return status::success;
}

status_t kernel_t::get_kernel_binary(xpu::binary_t &binary) const {
    binary = kernel_binary_;
    return status::success;
}

std::string kernel_t::name() const {
    return kernel_name_;
}

status_t kernel_t::dump() const {
    return gpu::intel::gpu_utils::dump_kernel_binary(
            kernel_binary_, kernel_name_);
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
