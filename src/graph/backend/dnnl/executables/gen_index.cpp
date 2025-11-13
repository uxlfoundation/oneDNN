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

#include "graph/backend/dnnl/executables/gen_index.hpp"

#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

genindex_executable_t::genindex_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout)
    : axis_(op->get_attr<int64_t>(op_attr::axis)) {
    using ltw = logical_tensor_wrapper_t;
    const auto &input_lt = op->get_input_logical_tensor(0);
    nelems_ = ltw(input_lt).nelems();
    ndims_ = ltw(input_lt).ndims();
    const auto &output_lt = op->get_output_logical_tensor(0);
    for (int i = 0; i < ndims_; i++) {
        output_dims_[i] = output_lt.dims[i];
        output_strides_[i] = output_lt.layout.strides[i];
    }
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    if (p_engine.get_kind() == engine::kind::gpu) {
        compute::kernel_ctx_t kernel_ctx;
        kernel_ctx.define_int("NDIMS", ndims_);
        for (int d = 0; d < MAX_NDIMS; ++d) {
            dim_t dim = (d < ndims_) ? output_dims_[d] : 1;
            dim_t stride = (d < ndims_) ? output_strides_[d] : 0;
            kernel_ctx.define_int(dnnl::impl::utils::format("D%d", d), dim);
            kernel_ctx.define_int(dnnl::impl::utils::format("S%d", d), stride);
        }
        auto *intel_engine
                = dnnl::impl::utils::downcast<gpu::intel::engine_t *>(
                        p_engine.get());
        std::vector<compute::kernel_t> kernels(1);
        intel_engine->create_kernels(&kernels, {"gen_index"}, kernel_ctx);
        kernel_ = kernels[0];
    }
#endif
}

void genindex_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    const auto &it_dst = args.find(DNNL_ARG_DST);
    if (it_dst == args.end()) return;

    auto &output = it_dst->second;
    auto output_ptr = static_cast<int32_t *>(output.get_data_handle());

    stream.get()->before_exec_hook();
    dnnl::impl::parallel_nd(nelems_, [&](dim_t i) {
        dims_t input_dims; // decomposition for physical offsets
        dnnl::impl::utils::l_dims_by_l_offset(
                input_dims, i, output_dims_, ndims_);
        auto offset
                = utils::offset_compute(output_strides_, input_dims, ndims_);
        output_ptr[offset] = input_dims[axis_];
    });
    stream.get()->after_exec_hook();
}

#ifdef DNNL_WITH_SYCL
::sycl::event genindex_executable_t::execute_sycl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    if (stream.get_engine().get_kind() == engine::kind::cpu) {
        auto strm_t = stream.get();
        auto *sycl_stream_impl = dnnl::impl::utils::downcast<
                dnnl::impl::xpu::sycl::stream_impl_t *>(strm_t->impl());

        strm_t->before_exec_hook();
        if (!deps.empty()) { sycl_stream_impl->sycl_ctx().set_deps(deps); }

        execute(stream, args);

        // return output event
        ::sycl::event return_event = sycl_stream_impl->get_output_event();
        strm_t->after_exec_hook();
        return return_event;
    }
#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && (DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
    auto compute_stream
            = dnnl::impl::utils::downcast<gpu::intel::stream_t *>(stream.get());
    compute::range_t gws = {static_cast<size_t>(nelems_)};
    auto nd_range = compute::nd_range_t(gws);
    compute::kernel_arg_list_t arg_list;
    const auto &dst = *(args.at(DNNL_ARG_DST).get()->memory_storage());
    arg_list.set(0, dst);
    arg_list.set(1, axis_);
    auto *sycl_stream
            = dnnl::impl::utils::downcast<sycl::stream_t *>(compute_stream);
    sycl_stream->before_exec_hook();
    if (!deps.empty()) sycl_stream->sycl_ctx().set_deps(deps);

    kernel_.parallel_for(*compute_stream, nd_range, arg_list,
            sycl_stream->sycl_ctx().get_deps(),
            sycl_stream->sycl_ctx().get_deps());
    auto return_event = sycl_stream->get_output_event();

    sycl_stream->after_exec_hook();
    return return_event;
#else
    assertm(false,
            "genindex opexcutable is only implemented for intel vendor "
            "under SYCL runtime ");
    throw std::runtime_error("Unimplement");
#endif
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
cl_event genindex_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps) const {
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    auto compute_stream
            = dnnl::impl::utils::downcast<gpu::intel::stream_t *>(stream.get());

    compute::range_t gws = {static_cast<size_t>(nelems_)};

    auto nd_range = compute::nd_range_t(gws);
    compute::kernel_arg_list_t arg_list;
    const auto &dst = *(args.at(DNNL_ARG_DST).get()->memory_storage());
    arg_list.set(0, dst);
    arg_list.set(1, axis_);
    auto *ocl_stream = dnnl::impl::utils::downcast<gpu::intel::ocl::stream_t *>(
            compute_stream);

    ocl_stream->before_exec_hook();

    if (!deps.empty()) {
        std::vector<xpu::ocl::wrapper_t<cl_event>> events(deps.size());
        for (size_t i = 0; i < deps.size(); i++)
            events[i] = xpu::ocl::wrapper_t<cl_event>(deps[i], true);
        ocl_stream->ocl_ctx().set_deps(events);
    }

    kernel_.parallel_for(*compute_stream, nd_range, arg_list,
            compute_stream->ctx().get_deps(), compute_stream->ctx().get_deps());

    cl_event return_event = nullptr;
    if ((ocl_stream->flags() & stream_flags::in_order) == 0) {
        auto last = ocl_stream->get_output_event();
        return_event = last.release();
    }

    ocl_stream->after_exec_hook();
    return return_event;
#else
    assertm(false,
            "genindex opexcutable is only implemented for intel vendor "
            "under OCL runtime ");
    throw std::runtime_error("Unimplement");
#endif
}
#endif

arg_indices_t genindex_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);

    arg_indices_t args;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, 0}});
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
