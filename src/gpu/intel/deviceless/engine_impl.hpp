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

#ifndef GPU_INTEL_DEVICELESS_ENGINE_IMPL_HPP
#define GPU_INTEL_DEVICELESS_ENGINE_IMPL_HPP

#include "common/engine_impl.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

class engine_impl_t : public impl::engine_impl_t {
public:
    engine_impl_t() = delete;
    engine_impl_t(runtime_kind_t runtime_kind,
            const std::vector<uint8_t> &cache_blob);

    ~engine_impl_t() override = default;

    engine_id_t engine_id() const override;

    status_t create_memory_storage(memory_storage_t **storage,
            impl::engine_t *engine, unsigned flags, size_t size,
            void *handle) const override;

    // Matches the OpenCL and Level Zero engines so that primitive descriptors
    // don't depend on whether the engine has a device.
    int get_buffer_alignment() const override { return 128; }

private:
    std::vector<uint8_t> cache_blob_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_impl_t);
};

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_DEVICELESS_ENGINE_IMPL_HPP
