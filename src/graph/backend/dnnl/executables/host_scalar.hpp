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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct host_scalar_executable_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER;

    host_scalar_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, pd_cache_t &pd_cache,
            const fpmath_t &fpmath, bool use_block_layout) {
        UNUSED(op);
        UNUSED(p_engine);
        UNUSED(pd_cache);
        UNUSED(fpmath);
        UNUSED(use_block_layout);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        auto it_src = args.find(DNNL_ARG_FROM);
        auto it_dst = args.find(DNNL_ARG_TO);

        if (it_src == args.end() || it_dst == args.end()) {
            assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
            return;
        }

        const memory &src_mem = it_src->second;
        const memory &dst_mem = it_dst->second;
        DNNL_HOST_SCALAR_TYPE_SWITCH(
                src_mem.get_desc().get_data_type(), DType, {
                    const DType val = src_mem.get_host_scalar_value<DType>();
                    std::memcpy(dst_mem.get_data_handle(), &val, sizeof(DType));
                });
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        auto it_src = args.find(DNNL_ARG_FROM);
        auto it_dst = args.find(DNNL_ARG_TO);

        if (it_src == args.end() || it_dst == args.end()) {
            assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
            return {};
        }

        const memory &src_mem = it_src->second;
        const memory &dst_mem = it_dst->second;

        // Use queue.memcpy() to copy the host scalar value to device memory. We
        // have to wait here as the val is on stack and will become invalid if
        // queue.memcpy() is asynchronous. A better solution may be supporting
        // host scalar memory to device memory reorder primitive.
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        const size_t size = src_mem.get_desc().get_size();
        const auto dt = src_mem.get_desc().get_data_type();
        assert(size
                == types::data_type_size(static_cast<impl::data_type_t>(dt)));
        DNNL_HOST_SCALAR_TYPE_SWITCH(dt, DType, {
            const DType val = src_mem.get_host_scalar_value<DType>();
            sycl_queue
                    .memcpy(dst_mem.get_data_handle(),
                            static_cast<const void *>(&val), size)
                    .wait();
        });
        return {};
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        auto it_src = args.find(DNNL_ARG_FROM);
        auto it_dst = args.find(DNNL_ARG_TO);

        if (it_src == args.end() || it_dst == args.end()) {
            assert(!"cannot find memory for DNNL_ARG_FROM or DNNL_ARG_TO");
            return {};
        }

        const memory &src_mem = it_src->second;
        const memory &dst_mem = it_dst->second;

        assert(deps.size() <= 1);
        // Passing the empty event to memcpy below causes failure.
        const bool empty = deps.empty() || deps[0] == nullptr;
        const cl_uint num = empty ? 0 : static_cast<cl_uint>(deps.size());
        const size_t size = src_mem.get_desc().get_size();
        const auto dt = src_mem.get_desc().get_data_type();
        assert(size
                == types::data_type_size(static_cast<impl::data_type_t>(dt)));
        cl_event e = nullptr;
        DNNL_HOST_SCALAR_TYPE_SWITCH(dt, DType, {
            const DType val = src_mem.get_host_scalar_value<DType>();
            UNUSED_STATUS(xpu::ocl::usm::memcpy(stream.get(),
                    dst_mem.get_data_handle(), static_cast<const void *>(&val),
                    size, num, empty ? nullptr : deps.data(), &e));
            clWaitForEvents(1, &e);
        });
        return e;
    }
#endif

    status_t reset_engine(const dnnl::engine &p_engine) override {
        UNUSED(p_engine);
        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_HOST_SCALAR_HPP
