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

#include "graph/backend/dnnl/executables/host_scalar.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

arg_indices_t host_scalar_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);
    arg_indices_t arg_indices;

    arg_indices.insert(
            {DNNL_ARG_FROM, indices_t {indices_t::type_t::input, 0}});
    arg_indices.insert({DNNL_ARG_TO, indices_t {indices_t::type_t::output, 0}});
    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
