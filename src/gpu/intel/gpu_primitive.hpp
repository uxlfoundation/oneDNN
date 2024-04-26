/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_GPU_PRIMITIVE_HPP
#define GPU_INTEL_GPU_PRIMITIVE_HPP

#include <cassert>
#include "gpu/intel/compute/utils.hpp"

#include "common/cache_blob.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/gemm/gpu_gemm_exec_types.hpp"
#include "gpu/intel/gpu_resource.hpp"
#include "gpu/intel/jit/jit_generator_base.hpp"
#include "gpu/intel/kernel_cache.hpp"
#include "gpu/intel/ocl/types_interop.hpp"
#include "hrt/utils.hpp"

#include "gpu/gpu_resource.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct primitive_t : public impl::primitive_t {
    using impl::primitive_t::primitive_t;

    struct compute_block_t {
        compute_block_t(impl::primitive_t *primitive) : primitive_(primitive) {}
        virtual ~compute_block_t() = default;

        status_t get_cache_blob_size(
                impl::engine_t *engine, size_t *size) const {
            if (primitive_)
                return primitive_->get_cache_blob_size(engine, size);
            return get_cache_blob_size_impl(engine, size);
        }

        status_t get_cache_blob(
                impl::engine_t *engine, cache_blob_t &blob) const {
            if (primitive_) return primitive_->get_cache_blob(engine, blob);
            return get_cache_blob_impl(engine, blob);
        }

        bool empty() const { return empty_impl(); }

        const impl::primitive_t *primitive() const { return primitive_; }

    private:
        virtual bool empty_impl() const { return !bool(primitive_); }

        virtual status_t get_cache_blob_size_impl(
                impl::engine_t *engine, size_t *size) const {
            assert(!"unexpected");
            return status::runtime_error;
        }
        virtual status_t get_cache_blob_impl(
                impl::engine_t *engine, cache_blob_t &blob) const {
            assert(!"unexpected");
            return status::runtime_error;
        }

        // "primitive" is a common compute block for all vendors and kernel
        // languages.
        impl::primitive_t *primitive_;
    };

    status_t create_nested_primitive(
            std::shared_ptr<impl::primitive_t> &primitive,
            const std::shared_ptr<primitive_desc_t> &pd,
            impl::engine_t *engine) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd->create_primitive_nested(p, engine, cache_blob()));

        if (p.second == cache_state_t::kernel_hit) {
            creation_cached_state_ = cache_state_t::nested_primitive_hit;
        }
        primitive = p.first;
        register_compute_block(new compute_block_t(primitive.get()));
        return status::success;
    }

    status_t get_cache_blob_size(
            impl::engine_t *engine, size_t *size) const override {
        if (!size) return status::invalid_arguments;
        // Query binary size for each created kernel.
        for (const auto &cb : compute_blocks()) {
            if (cb->empty()) continue;
            CHECK(cb->get_cache_blob_size(engine, size));
        }
        return status::success;
    }

    status_t get_cache_blob(
            impl::engine_t *engine, cache_blob_t &blob) const override {
        for (const auto &cb : compute_blocks()) {
            if (!cb) continue;

            switch (cb.kind()) {
                case compute_block_t::kind_t::kernel: {
                    // Get a binary for each kernel within current primitive.
                    hrt::binary_t binary;
                    CHECK(cb.kernel().get_binary(engine, binary));
                    CHECK(blob.add_binary(binary.data(), binary.size()));
                    break;
                }
                case compute_block_t::kind_t::primitive:
                    CHECK(cb.primitive()->get_cache_blob(engine, blob));
                    break;
                default: assert(!"unexpected"); return status::runtime_error;
            }
        }
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            jit::jit_generator_base *jitter, bool register_kernel = true) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernel(kernel, jitter, cache_blob()));
        if (register_kernel) CHECK(register_kernels({*kernel}));
        return status::success;
    }

    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernels(
                kernels, kernel_names, kernel_ctx, cache_blob()));
        CHECK(register_kernels(*kernels));
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx) {

        std::vector<compute::kernel_t> kernels(1);
        auto status
                = create_kernels(engine, &kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

    template <typename T>
    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names, const T &params) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        if (cache_blob())
            return compute_engine->create_kernels_from_cache_blob(
                    cache_blob(), kernels, kernel_names);

        auto key = std::make_shared<trivial_key_container_t<T>>(
                params, compute_engine->engine_id());
        gpu_assert(key->key.is_valid());

        CHECK(get_cached_kernels<typename trivial_key_t<T>::value_type>(
                std::move(key), engine, kernels, kernel_names));

        CHECK(register_kernels(kernels));

        return status::success;
    }

    template <typename T>
    status_t create_kernel(engine_t *engine, compute::kernel_t &kernel,
            const char *kernel_name, const T &params) {
        std::vector<compute::kernel_t> kernels(1);
        CHECK(create_kernels(engine, kernels, {kernel_name}, params));
        kernel = kernels[0];
        return status::success;
    }

    status_t create_nested_primitive(std::shared_ptr<primitive_t> &primitive,
            const std::shared_ptr<primitive_desc_t> &pd, engine_t *engine) {
        CHECK(pd->create_primitive(primitive, engine, cache_blob()));
        register_primitive(primitive.get());
        return status::success;
    }

    // TODO: use inheritance for exec_ctx_t to get rid of such places...
    static status_t parallel_for(const gemm_exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        return parallel_for(*compute_stream, range, kernel, arg_list,
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps());
    }

    static status_t parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());
        return parallel_for(*compute_stream, range, kernel, arg_list,
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps());
    }

    // Intel GPU hardware has a limitation on the size of work group dimensions to
    // be at most uint32_t. This function works around that by passing an offset
    // argument. The OpenCL native offset cannot be used due to lack of SYCL
    // interop support.
    static status_t large_parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &nd_range,
            const compute::kernel_t &kernel,
            compute::kernel_arg_list_t &arg_list, int offset_idx) {

        auto global_range = nd_range.global_range();
        auto local_range = nd_range.local_range();

        // Convert global_range to an equivalent 3D nd_range_t
        constexpr size_t range_ndims = 3;
        assert(global_range.ndims() <= range_ndims);
        auto gws = compute::range_t::one(range_ndims);
        for (size_t i = 0; i < global_range.ndims(); i++) {
            gws[i] = global_range[i];
        }

        compute::range_t off_inc(UINT32_MAX, UINT32_MAX, UINT32_MAX);
        if (local_range) {
            for (size_t i = 0; i < local_range.ndims(); i++) {
                off_inc[i] *= local_range[i];
            }
        }

        int64x3_t offset_arg = {};
        auto &offset = offset_arg.array;
        static_assert(range_ndims == 3,
                "Large parallel for loop doesn't match ndims.");
        for_(offset[2] = 0; static_cast<size_t>(offset[2]) < gws[2];
                offset[2] += off_inc[2])
        for_(offset[1] = 0; static_cast<size_t>(offset[1]) < gws[1];
                offset[1] += off_inc[1])
        for_(offset[0] = 0; static_cast<size_t>(offset[0]) < gws[0];
                offset[0] += off_inc[0])
        {
            arg_list.set(offset_idx, offset_arg);
            auto range = compute::range_t::empty(range_ndims);
            for (size_t i = 0; i < range_ndims; i++)
                range[i] = std::min(off_inc[i], gws[i] - offset[i]);

            CHECK(parallel_for(ctx, compute::nd_range_t(range, local_range),
                    kernel, arg_list));
        }
        return status::success;
    }

protected:
    int32_t version() const { return version_; }

    void set_version(int32_t version) { version_ = version; }

    void register_primitive(const primitive_t *primitive) {
        registered_compute_blocks_.emplace_back(primitive);
    }

    status_t register_kernels(const std::vector<compute::kernel_t> &kernels) {
        for (const auto &k : kernels) {
            if (k) CHECK(k.dump());
            registered_compute_blocks_.emplace_back(k);
        }
        return status::success;
    }

    virtual status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const {
        return status::success;
    }

    void register_compute_block(compute_block_t *cb) {
        compute_blocks_.emplace_back(cb);
    }

    const std::vector<std::unique_ptr<compute_block_t>> &
    compute_blocks() const {
        return compute_blocks_;
    }

private:
    void register_primitive(impl::primitive_t *primitive) {
        compute_blocks_.emplace_back(new compute_block_t(primitive));
    }

    std::vector<std::unique_ptr<compute_block_t>> compute_blocks_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
