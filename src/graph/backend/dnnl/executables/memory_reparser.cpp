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

#include "graph/backend/dnnl/executables/memory_reparser.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void memory_reparser_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    auto from = args.find(DNNL_ARG_FROM);
    auto to = args.find(DNNL_ARG_TO);
    if (from == args.end() || to == args.end()) return;

    if (from->second.get_data_handle() == to->second.get_data_handle())
        dummy_impl_t::execute(stream, args);
    else {
        const memory &dst_mem = to->second;
        const memory &src_mem = from->second;
        const memory temp_mem = make_dnnl_memory(dst_mem.get_desc(),
                src_mem.get_engine(), src_mem.get_data_handle());
        dnnl::reorder(temp_mem, dst_mem)
                .execute(stream, const_cast<memory &>(temp_mem),
                        const_cast<memory &>(dst_mem));
    }
}

#ifdef DNNL_WITH_SYCL
::sycl::event memory_reparser_t::execute_sycl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    auto from = args.find(DNNL_ARG_FROM);
    auto to = args.find(DNNL_ARG_TO);
    if (from == args.end() || to == args.end()) return {};

    if (from->second.get_data_handle() == to->second.get_data_handle())
        return dummy_impl_t::execute_sycl(stream, args, deps);
    else {
        const memory &src_mem = from->second;
        const memory &dst_mem = to->second;
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        auto e = sycl_queue.memcpy(dst_mem.get_data_handle(),
                src_mem.get_data_handle(), dst_mem.get_desc().get_size());
        return e;
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
cl_event memory_reparser_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps) const {
    auto from = args.find(DNNL_ARG_FROM);
    auto to = args.find(DNNL_ARG_TO);
    if (from == args.end() || to == args.end()) return {};

    if (from->second.get_data_handle() == to->second.get_data_handle())
        return dummy_impl_t::execute_ocl(stream, args, deps);
    else {
        const memory &src_mem = from->second;
        const memory &dst_mem = to->second;
        assert(deps.size() <= 1);
        // Passing the empty event to memcpy below causes failure.
        const bool empty = deps.empty() || deps[0] == nullptr;
        const cl_uint num = empty ? 0 : static_cast<cl_uint>(deps.size());
        cl_event e;
        UNUSED_STATUS(xpu::ocl::usm::memcpy(stream.get(),
                dst_mem.get_data_handle(), src_mem.get_data_handle(),
                dst_mem.get_desc().get_size(), num,
                empty ? nullptr : deps.data(), &e));
        return e;
    }
}
#endif

arg_indices_t memory_reparser_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;
    arg_indices.insert(
            {DNNL_ARG_FROM, indices_t {indices_t::type_t::input, 0}});
    arg_indices.insert({DNNL_ARG_TO, indices_t {indices_t::type_t::output, 0}});
    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
