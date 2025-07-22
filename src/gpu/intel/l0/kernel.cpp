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

kernel_wrapper_t::kernel_wrapper_t(ze_kernel_handle_t kernel)
    : kernel_(kernel) {}

kernel_wrapper_t::operator ze_kernel_handle_t() const {
    return kernel_;
}

status_t kernel_wrapper_t::set_arg(
        int arg_index, size_t arg_size, const void *arg_value) {
    return func_zeKernelSetArgumentValue(
            kernel_, arg_index, arg_size, arg_value);
}

kernel_cache_t::kernel_cache_t(
        ze_module_handle_t module, ze_kernel_handle_t main_kernel)
    : module_(module), main_kernel_(main_kernel) {}

kernel_cache_t::~kernel_cache_t() {
    for (auto &k : kernels_) {
        func_zeKernelDestroy(k.second);
    }
}

status_t kernel_cache_t::get(kernel_wrapper_t **kernel) {
    auto id = std::this_thread::get_id();
    {
        utils::lock_read_t lock_read(mutex_);
        auto it = kernels_.find(id);
        if (it != kernels_.end()) {
            *kernel = &it->second;
            return status::success;
        }
    }

    // No copy for this thread, clone the original kernel and save the
    // copy.
    ze_kernel_handle_t new_kernel;
    CHECK(clone_kernel(&new_kernel));

    utils::lock_write_t lock_write(mutex_);
    auto ret = kernels_.emplace(id, new_kernel);
    *kernel = &ret.first->second;
    return status::success;
}

status_t kernel_cache_t::clone_kernel(ze_kernel_handle_t *new_kernel) {
    std::string kernel_name;
    CHECK(get_kernel_name(main_kernel_, kernel_name));

    ze_kernel_desc_t kernel_desc = {};
    kernel_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernel_desc.pNext = nullptr;
    kernel_desc.flags = 0;
    kernel_desc.pKernelName = kernel_name.c_str();

    CHECK(func_zeKernelCreate(module_, &kernel_desc, new_kernel));

    return status::success;
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
        ze_module_handle_t module, ze_kernel_handle_t kernel,
        const compute::program_src_t &src) {
    compute_kernel = compute::kernel_t(
            std::make_shared<kernel_compat_t>(module, kernel, src));

    return status::success;
}

status_t kernel_t::parallel_for(impl::stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list, const xpu::event_t &deps,
        xpu::event_t &out_dep) {
    kernel_wrapper_t *kernel = nullptr;
    CHECK(cache_->get(&kernel));

    CHECK(check_scalar_arguments(arg_list));
    CHECK(check_alignment(arg_list));

    auto l0_device_info
            = utils::downcast<engine_t *>(stream.engine())->device_info();
    const size_t pointer_size = l0_device_info->device_address_bits() / 8;
    size_t param_bytes = 0;
    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            if (!mem_storage->is_null()) {
                auto stream_ctx = utils::downcast<engine_t *>(stream.engine())
                                          ->context();
                auto memory_storage_ctx
                        = utils::downcast<engine_t *>(mem_storage->engine())
                                  ->context();
                if (stream_ctx != memory_storage_ctx) {
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
    const size_t local_range_size = range.local_range().ndims();
    ze_group_count_t group_count = {};
    group_count.groupCountX = local_range_size > 0 ? range.local_range()[0] : 1;
    group_count.groupCountY = local_range_size > 1 ? range.local_range()[1] : 1;
    group_count.groupCountZ = local_range_size > 2 ? range.local_range()[2] : 1;

    auto *l0_stream = utils::downcast<stream_t *>(&stream);
    ze_event_handle_t event;
    if (l0_stream->flags() & stream_flags::out_of_order) {
        auto events = utils::downcast<const gpu::intel::l0::event_t *>(&deps)
                              ->events;
        auto events_size = events.size();
        auto event_data = events_size ? events.data() : nullptr;

        CHECK(func_zeCommandListAppendLaunchKernel(l0_stream->list(), *kernel,
                &group_count, event, events_size, event_data));

        event_t::from(out_dep).events = {event};
    } else {
        bool save_event = save_events_ || stream.is_profiling_enabled();

        CHECK(func_zeCommandListAppendLaunchKernel(l0_stream->list(), *kernel,
                &group_count, save_event ? event : nullptr, 0, nullptr));
    }

    if (stream.is_profiling_enabled()) {
        l0_stream->profiler().register_event(
                utils::make_unique<event_t>(std::move(event)));
    }

    return status::success;
}

status_t kernel_t::get_kernel_binary(xpu::binary_t &binary) const {
    size_t binary_size = 0;
    CHECK(func_zeGetKernelBinary(kernel_, &binary_size, nullptr));

    binary.resize(binary_size);
    CHECK(func_zeGetKernelBinary(kernel_, &binary_size, binary.data()));

    return status::success;
}

std::string kernel_t::name() const {
    std::string kernel_name;
    get_kernel_name(kernel_, kernel_name);

    return kernel_name;
}

status_t kernel_t::dump() const {
    xpu::binary_t binary;
    CHECK(get_kernel_binary(binary));

    return gpu::intel::gpu_utils::dump_kernel_binary(binary, name());
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

void kernel_t::save_output_events() {
    save_events_ = true;
};

kernel_t::kernel_t(const ze_module_handle_t module,
        const ze_kernel_handle_t kernel,
        const gpu::intel::compute::program_src_t &src)
    : module_(module), kernel_(kernel), src_(src), save_events_(false) {
    cache_ = std::make_shared<kernel_cache_t>(module_, kernel_);
}

kernel_t::~kernel_t() {
    func_zeKernelDestroy(kernel_);
    func_zeModuleDestroy(module_);
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
