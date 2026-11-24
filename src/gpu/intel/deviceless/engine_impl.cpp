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

#include "gpu/intel/deviceless/engine_impl.hpp"
#include "gpu/intel/deviceless/engine_id.hpp"

#include "common/verbose.hpp"

#include <limits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

namespace {
// Runtime engines never use this index, so `engine_id_impl_t::compare()` never
// pairs a device-less ID with a runtime ID and downcasts it to the wrong type.
constexpr size_t engine_id_index = std::numeric_limits<size_t>::max();
} // namespace

engine_impl_t::engine_impl_t(
        runtime_kind_t runtime_kind, const std::vector<uint8_t> &cache_blob)
    : impl::engine_impl_t(engine_kind::gpu, runtime_kind, 0)
    , cache_blob_(cache_blob) {}

engine_id_t engine_impl_t::engine_id() const {
    return engine_id_t(
            new engine_id_impl_t(runtime_kind(), cache_blob_, engine_id_index));
}

status_t engine_impl_t::create_memory_storage(memory_storage_t **storage,
        impl::engine_t *engine, unsigned flags, size_t size,
        void *handle) const {
    VERROR(common, runtime, "device-less engine has no memory");
    return status::unimplemented;
}

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
