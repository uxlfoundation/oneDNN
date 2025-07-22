/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "common/primitive_desc_iface.hpp"
#include "common/primitive_iface.hpp"
#include "common/utils.hpp"
#include "gpu/intel/l0/stream.hpp"

using namespace dnnl::impl;

dnnl_status_t dnnl_l0_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, dnnl_stream_t stream,
        size_t nargs, const dnnl_exec_arg_t *args, size_t ndeps,
        const ze_event_handle_t *deps, ze_event_handle_t *return_event) {
    const bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::l0
            && IMPLICATION(nargs > 0, args != nullptr)
            && IMPLICATION(ndeps > 0, deps != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *l0_stream = utils::downcast<gpu::intel::l0::stream_t *>(stream);
    stream->before_exec_hook();

    if (deps != nullptr) {
        std::vector<ze_event_handle_t> events(ndeps);
        for (size_t i = 0; i < ndeps; i++)
            events[i] = deps[i];
        l0_stream->l0_ctx().set_deps(events);
    }

    // run primitive
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(primitive_iface->pd()->impl().get(),
            static_cast<int>(nargs), args, exec_args));

    exec_ctx_t ctx(stream, std::move(exec_args));
    CHECK(primitive_execute(primitive_iface, ctx));

    // return output event
    if (return_event != nullptr) {
        if (l0_stream->impl()->flags() & stream_flags::in_order) {
            *return_event = nullptr;
        } else {
            *return_event = l0_stream->get_output_event();
        }
    }

    stream->after_exec_hook();

    return status::success;
}
