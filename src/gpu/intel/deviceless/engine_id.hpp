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

#ifndef GPU_INTEL_DEVICELESS_ENGINE_ID_HPP
#define GPU_INTEL_DEVICELESS_ENGINE_ID_HPP

#include "common/engine_id.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

// The engine blob is the only identity a device-less engine has.
struct engine_id_impl_t : public impl::engine_id_impl_t {
    engine_id_impl_t(runtime_kind_t runtime_kind,
            const std::vector<uint8_t> &cache_blob, size_t index)
        : impl::engine_id_impl_t(engine_kind::gpu, runtime_kind, index)
        , cache_blob_(cache_blob) {}
    ~engine_id_impl_t() override = default;

private:
    bool compare_resource(
            const impl::engine_id_impl_t *id_impl) const override {
        const auto *typed_id
                = utils::downcast<const engine_id_impl_t *>(id_impl);
        return cache_blob_ == typed_id->cache_blob_;
    }

    size_t hash_resource() const override {
        return serialization_stream_t::from_data(cache_blob_).get_hash();
    }

    std::vector<uint8_t> cache_blob_;

    engine_id_impl_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_id_impl_t);
};

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_DEVICELESS_ENGINE_ID_HPP
