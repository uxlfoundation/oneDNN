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

#ifndef GRAPH_BACKEND_DNNL_EXECUTABLES_DELETER_UTIL_HPP
#define GRAPH_BACKEND_DNNL_EXECUTABLES_DELETER_UTIL_HPP

#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct pd_deleter_t {
    void operator()(dnnl_primitive_desc_t pd) const {
        dnnl_primitive_desc_destroy(pd);
    }
};

struct prim_deleter_t {
    void operator()(dnnl_primitive_t prim) const {
        dnnl_primitive_destroy(prim);
    }
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif // GRAPH_BACKEND_DNNL_EXECUTABLES_DELETER_UTIL_HPP
