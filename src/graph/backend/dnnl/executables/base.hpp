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
#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_BASE_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_BASE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "common/primitive.hpp"
#include "common/primitive_desc_iface.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#include "graph/utils/ocl_usm_utils.hpp"

#include "xpu/ocl/usm_utils.hpp"

#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"

#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && (DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
#include "gpu/intel/engine.hpp"
#include "gpu/intel/stream.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/stream.hpp"
#endif

#endif

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/stream.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct indices_t {
    // the type_t is used to indicate the indices is for input or output
    enum class type_t {
        input = 0,
        output = 1,
    };

    type_t type_;
    size_t value_;
};

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

// DNNL arg to in/outputs indices mapping. For example, <DNNL_ARG_SRC, {input,
// 0}> means the 0-th input of an op should be used as primitive's src argument.
// We should be able to know this map according the information on an op.
using arg_indices_t = std::unordered_map<int, indices_t>;

using arg_indices_getter_func = std::function<arg_indices_t(const op_t *)>;

void get_arg_indices_for_post_ops(
        const op_t *op, arg_indices_t &indices, size_t &base_index);

arg_indices_t get_arg_indices_for_siso_op(const op_t *op);
arg_indices_t get_arg_indices_for_miso_op(const op_t *op);
arg_indices_t get_arg_indices_for_conv_and_matmul(const op_t *op);
arg_indices_t get_arg_indices_for_lnorm_and_gnorm(const op_t *op);

// A dummy arg indices getter which is only used for those internal ops that are
// only for fusion purpose, like dnnl_add_zps and dnnl_sub_zps. The dummy getter
// should never be called.
inline arg_indices_t dummy_arg_indices_getter(const op_t *op) {
    UNUSED(op);
    assertm(false, "dummy getter should never be called");
    return arg_indices_t {};
}

// Used to declare the arg indices getter inside an op executable class. The
// getter can be used to generate the <dnnl_arg, in/output index> map. According
// to that, we can form the execution args by using the in/outputs list in op.
#define DECLARE_ARG_INDICES_GETTER \
    static arg_indices_t get_arg_indices(const op_t *op);

#define DECLARE_RESET_ENGINE(primitive) \
    status_t reset_engine(const dnnl::engine &p_engine) override { \
        const auto desc_t = prim_.get_primitive_desc()->impl(); \
        dnnl_primitive_desc new_pd_t(desc_t, p_engine.get()); \
        primitive::primitive_desc new_pd(&new_pd_t); \
        prim_ = primitive(new_pd); \
        return status::success; \
    } // namespace dnnl_impl

struct op_executable_t {
    virtual ~op_executable_t() = default;
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const = 0;
    virtual status_t reset_engine(const dnnl::engine &engine) = 0;
#ifdef DNNL_WITH_SYCL
    virtual ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const = 0;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    virtual cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const = 0;
#endif
};

using executable_creator_func = std::function<std::shared_ptr<op_executable_t>(
        std::shared_ptr<op_t> &, const dnnl::engine &, pd_cache_t &,
        const fpmath_t &, bool)>;

// A dummy executable creator which is only used for those internal ops that are
// only for fusion purpose, like dnnl_add_zps and dnnl_sub_zps. The dummy
// creator should never be called.
inline std::shared_ptr<op_executable_t> dummy_executable_creator(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);
    assertm(false, "dummy executable creator should never be called");
    return {};
}

// A general template executable fcreator function, which can be specialized by
// using different op executable class types
template <typename T>
inline std::shared_ptr<op_executable_t> executable_creator(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    return std::make_shared<T>(
            op, p_engine, pd_cache, fpmath, use_block_layout);
}

// Used to declare the desc_t class and the static create_desc method inside an
// op executable class
#define DECLARE_DESC_CLASS_AND_CREATOR(primitive_desc) \
    using type = primitive_desc; /* NOLINT */ \
    class desc_t : public type { \
        bool from_cache_; \
\
    public: \
        desc_t(const type &pd, bool from_cache) \
            : type(pd), from_cache_(from_cache) {} \
        bool is_from_cache() const { return from_cache_; } \
    }; \
    static desc_t create_desc(std::shared_ptr<op_t> &op, \
            const dnnl::engine &p_engine, pd_cache_t &pd_cache, \
            const fpmath_t &fpmath, bool use_block_layout);

// This class is a dummy executable which doesn't do any actual computation.
// This dummy executable can be used to:
// - support data formatting ops like permute/reshape/transpose
// - support zero-volume tensor (empty tensor) like (1024, 64)x(64, 0)
//
// In the execute_sycl function, we will run a dummy sycl kernel to gather all
// the input events
struct dummy_impl_t : public op_executable_t {
    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(stream);
        UNUSED(args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps) const override {
        UNUSED(stream);

        // Fast path: if no event, return an immediate event.
        if (deps.empty()) return {};

        // Fast path: if only one event, return it.
        if (deps.size() == 1) return deps[0];

        // Otherwise, we run a trivial kernel to gather all deps. The
        // dummy task is needed to not get an error related to empty
        // kernel.
        auto q = dnnl::sycl_interop::get_queue(stream);
        auto e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(deps);
            cgh.single_task<class dnnl_graph_dummy_kernel>([]() {});
        });
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps) const override {
        UNUSED(stream);

        // Fast path: if no event, return an immediate event.
        if (deps.empty()) return {};

        // Fast path: if only one event, return it.
        if (deps.size() == 1) return deps[0];

        // Otherwise, gather all dependencies.
        auto q = dnnl::ocl_interop::get_command_queue(stream);
        cl_event e;
        auto err = clEnqueueMarkerWithWaitList(
                q, static_cast<cl_uint>(deps.size()), deps.data(), &e);
        assert(err == CL_SUCCESS);
        MAYBE_UNUSED(err);
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

#endif
