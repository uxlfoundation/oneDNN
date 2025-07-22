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

#include "oneapi/dnnl/dnnl_ze.h"

#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/ze/stream_impl.hpp"

using namespace dnnl::impl;

status_t dnnl_ze_interop_stream_create(stream_t **stream, engine_t *engine,
        ze_command_list_handle_t list, int profiling) {
    bool args_ok = !utils::any_null(stream, engine, list)
            && engine->runtime_kind() == runtime_kind::ze;
    if (!args_ok) return status::invalid_arguments;

    unsigned flags = stream_flags::default_flags;
    if (profiling) {
#ifdef DNNL_EXPERIMENTAL_PROFILING
        flags |= stream_flags::profiling;
#endif
    }

    std::unique_ptr<stream_impl_t> stream_impl(
            new xpu::ze::stream_impl_t(flags, list));
    if (!stream_impl) return status::out_of_memory;

    CHECK(engine->create_stream(stream, stream_impl.get()));
    stream_impl.release();

    return status::success;
}

status_t dnnl_ze_interop_stream_get_list(
        stream_t *stream, ze_command_list_handle_t *list) {
    bool args_ok = !utils::any_null(list, stream)
            && stream->engine()->runtime_kind() == runtime_kind::ze;
    if (!args_ok) return status::invalid_arguments;

    auto *ze_stream_impl
            = utils::downcast<const xpu::ze::stream_impl_t *>(stream->impl());
    *list = ze_stream_impl->list();

    return status::success;
}
