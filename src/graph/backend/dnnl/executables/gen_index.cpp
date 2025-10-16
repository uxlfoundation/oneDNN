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

#include "graph/backend/dnnl/executables/gen_index.hpp"

#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void genindex_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    const auto &it_dst = args.find(DNNL_ARG_DST);
    if (it_dst == args.end()) return;

    auto &output = it_dst->second;
    auto output_ptr = static_cast<int32_t *>(output.get_data_handle());

    stream.get()->before_exec_hook();
    dnnl::impl::parallel_nd(nelems_, [&](dim_t i) {
        dims_t input_dims; // decomposition for physical offsets
        dnnl::impl::utils::l_dims_by_l_offset(
                input_dims, i, output_dims_, ndims_);
        auto offset
                = utils::offset_compute(output_strides_, input_dims, ndims_);
        output_ptr[offset] = input_dims[axis_];
    });
    stream.get()->after_exec_hook();
}

arg_indices_t genindex_executable_t::get_arg_indices(const op_t *op) {
    UNUSED(op);

    arg_indices_t arg_indices;
    arg_indices.insert({DNNL_ARG_SRC, indices_t {indices_t::type_t::input, 0}});
    arg_indices.insert(
            {DNNL_ARG_DST, indices_t {indices_t::type_t::output, 0}});

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
