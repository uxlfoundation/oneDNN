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

#include "common/stream.hpp"

#include "xpu/ze/memory_storage.hpp"
#include "xpu/ze/stream_impl.hpp"
#include "xpu/ze/usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

namespace {
status_t fill(impl::stream_t *stream, void *ptr, uint8_t pattern, size_t size) {
    if (size == 0) return status::success;

    auto *ze_stream_impl
            = utils::downcast<xpu::ze::stream_impl_t *>(stream->impl());

    CHECK(ze::zeCommandListAppendMemoryFill(ze_stream_impl->list(), ptr,
            &pattern, sizeof(pattern), size, nullptr, 0, nullptr));

    return status::success;
}

status_t copy(impl::stream_t *stream, void *dst, const void *src, size_t size) {
    if (size == 0) return status::success;

    auto *ze_stream_impl
            = utils::downcast<xpu::ze::stream_impl_t *>(stream->impl());

    CHECK(ze::zeCommandListAppendMemoryCopy(
            ze_stream_impl->list(), dst, src, size, nullptr, 0, nullptr));

    return status::success;
}
} // namespace

void *malloc_device(impl::engine_t *engine, size_t size) {
    return memory_storage_t::malloc_device(engine, size);
}

void *malloc_shared(impl::engine_t *engine, size_t size) {
    return memory_storage_t::malloc_shared(engine, size);
}
void free(impl::engine_t *engine, void *ptr) {
    memory_storage_t::free(engine, ptr);
}

status_t memset(impl::stream_t *stream, void *ptr, int value, size_t size) {
    uint8_t pattern = (uint8_t)value;
    return fill(stream, ptr, pattern, size);
}

status_t memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size) {
    return copy(stream, dst, src, size);
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
