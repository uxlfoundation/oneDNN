/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_PRIMITIVE_HPP
#define GPU_INTEL_PRIMITIVE_HPP

#include <cassert>
#include "gpu/intel/compute/utils.hpp"

#include "common/cache_blob.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/types_interop.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/kernel_cache.hpp"
#include "gpu/intel/stream.hpp"
#include "xpu/context.hpp"
#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct primitive_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct compute_block_t : public gpu::primitive_t::compute_block_t {
        compute_block_t(const compute::kernel_t &kernel)
            : gpu::primitive_t::compute_block_t(nullptr), kernel_(kernel) {}

        compute::kernel_t kernel() const { return kernel_; }

        uint64_t get_hash() const { return kernel_.get_hash(); };

    private:
        bool empty_impl() const override { return !bool(kernel_); }

        status_t get_cache_blob_size_impl(
                impl::engine_t *engine, size_t *size) const override {
            if (empty()) return status::success;
            size_t sz = 0;
            CHECK(kernel().get_binary_size(engine, &sz));
            // We need additional sizeof(size_t) bytes to store the size
            // of the binary when packing.
            (*size) += sz + sizeof(size_t);
            return status::success;
        }

        status_t get_cache_blob_impl(
                impl::engine_t *engine, cache_blob_t &blob) const override {
            if (empty()) return status::success;
            xpu::binary_t binary;
            CHECK(kernel().get_binary(engine, binary));
            CHECK(blob.add_binary(binary.data(), binary.size()));
            return status::success;
        }

        compute::kernel_t kernel_;
    };

    // This may need overridden when implementations perform dispatching via
    // runtime parameters.
    uint64_t get_id() const override {
        size_t hash = 0;
        for (auto &block : compute_blocks_) {
            if (block->primitive()) {
                auto id = block->primitive()->get_id();
                if (id == 0) return 0;
                hash = hash_combine(hash, id);
            } else {
                auto *b = utils::downcast<const compute_block_t *>(block.get());
                hash = hash_combine(hash, b->get_hash());
            }
        }
        return hash != 0 ? hash : 2147483647; // id is required to be non-zero
    }

    status_t get_cache_blob_size(
            impl::engine_t *engine, size_t *size) const override {
        if (!size) return status::invalid_arguments;
        if (version_ != -1) (*size) += sizeof(version_);
        return gpu::primitive_t::get_cache_blob_size(engine, size);
    }

    status_t get_cache_blob(
            impl::engine_t *engine, cache_blob_t &blob) const override {
        if (version_ != -1)
            CHECK(blob.add_value((const uint8_t *)&version_, sizeof(version_)));
        return gpu::primitive_t::get_cache_blob(engine, blob);
    }

    status_t create_kernel(impl::engine_t *engine, compute::kernel_t *kernel,
            jit::generator_base_t *jitter, bool register_kernel = true);

    status_t create_kernels(impl::engine_t *engine,
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx);

    status_t create_kernel(impl::engine_t *engine, compute::kernel_t *kernel,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx);

    template <typename T>
    status_t create_kernels(impl::engine_t *engine,
            std::vector<compute::kernel_t> &kernels,
            const std::vector<const char *> &kernel_names, const T &params) {
        auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
        if (cache_blob()) {
            CHECK(intel_engine->create_kernels_from_cache_blob(
                    cache_blob(), kernels, kernel_names));
            for (auto &k : kernels)
                k.hash_dump("blob");
            CHECK(register_kernels(kernels));
            return status::success;
        }

        auto key = std::make_shared<trivial_key_container_t<T>>(
                params, intel_engine->engine_id());
        gpu_assert(key->key.is_valid());

        cache_state_t kernel_cache_status;
        CHECK(get_cached_kernels<typename trivial_key_t<T>::value_type>(
                std::move(key), engine, kernels, kernel_names,
                kernel_cache_status));
        if (kernel_cache_status == cache_state_t::kernel_hit) {
            creation_cached_state_ = cache_state_t::kernel_hit;
        }

        for (auto &k : kernels)
            k.hash_dump("real");
        CHECK(register_kernels(kernels));

        return status::success;
    }

    template <typename T>
    status_t create_kernel(impl::engine_t *engine, compute::kernel_t &kernel,
            const char *kernel_name, const T &params) {
        std::vector<compute::kernel_t> kernels(1);
        VCHECK_KERNEL(create_kernels(engine, kernels, {kernel_name}, params),
                VERBOSE_KERNEL_CREATION_FAIL, kernel_name);
        kernel = kernels[0];
        return status::success;
    }

    // TODO: use inheritance for exec_ctx_t to get rid of such places...
    static status_t parallel_for(const gemm::exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());
        return parallel_for(*compute_stream, range, kernel, arg_list,
                compute_stream->ctx().get_deps(),
                compute_stream->ctx().get_deps());
    }

    static status_t parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) {
        auto compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());
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
            compute::kernel_arg_list_t &arg_list, int offset_idx);

protected:
    int32_t version() const { return version_; }

    void set_version(int32_t version) { version_ = version; }

    status_t register_kernels(const std::vector<compute::kernel_t> &kernels) {
        for (const auto &k : kernels) {
            if (k) CHECK(k.dump());
            register_compute_block(new compute_block_t(k));
        }
        return status::success;
    }

private:
    static status_t parallel_for(impl::stream_t &stream,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) {
        return kernel.parallel_for(stream, range, arg_list, deps, out_dep);
    }

    // Persistent cache versioning is not used by default. To enable versioning
    // the primitive should:
    // 1) Set the version via set_version() in case of non-cached initialization
    // 2) Retrieve the version from the cache blob and set it via set_version()
    //    in case of cached initialization
    int32_t version_ = -1;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
