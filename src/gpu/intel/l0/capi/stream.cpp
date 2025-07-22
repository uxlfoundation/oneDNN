/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_l0.h"

#include "common/utils.hpp"
#include "gpu/intel/l0/stream.hpp"

using namespace dnnl::impl;

dnnl_status_t dnnl_l0_interop_stream_create(dnnl_stream_t *stream,
        dnnl_engine_t engine, ze_command_list_handle_t list) {
    bool args_ok = !utils::any_null(stream, engine, list)
            && engine->runtime_kind() == runtime_kind::l0;
    if (!args_ok) return status::invalid_arguments;

    std::unique_ptr<stream_impl_t> stream_impl(
            new gpu::intel::l0::stream_impl_t(
                    stream_flags::default_flags, list));
    if (!stream_impl) return status::out_of_memory;

    CHECK(engine->create_stream(stream, stream_impl.get()));
    stream_impl.release();

    return status::success;
}

dnnl_status_t dnnl_l0_interop_stream_get_list(
        dnnl_stream_t stream, ze_command_list_handle_t list) {
    bool args_ok = !utils::any_null(list, stream)
            && stream->engine()->runtime_kind() == runtime_kind::l0;
    if (!args_ok) return status::invalid_arguments;

    auto *l0_stream = utils::downcast<const gpu::intel::l0::stream_t *>(stream);
    list = l0_stream->list();

    return status::success;
}
