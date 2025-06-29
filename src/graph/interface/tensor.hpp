/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GRAPH_INTERFACE_TENSOR_HPP
#define GRAPH_INTERFACE_TENSOR_HPP

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

struct dnnl_graph_tensor {
public:
    dnnl_graph_tensor() = default;

    dnnl_graph_tensor(const dnnl::impl::graph::logical_tensor_t &lt,
            const dnnl::impl::graph::engine_t *eng, void *handle);

    void *get_data_handle() const { return handle_.get(); }

    void set_data_handle(void *handle) {
        if (lt_.property == dnnl::impl::graph::property_type::host_scalar) {
            if (lt_.data_type == dnnl::impl::graph::data_type::s32) {
                scalar_.s32_value = *static_cast<int32_t *>(handle);
                handle_.reset(&scalar_.s32_value, dummy_destructor);
            } else {
                assertm(false, "Unsupported data type for host scalar");
            }
        } else {
            handle_.reset(handle, dummy_destructor);
        }
    }

    const dnnl::impl::graph::logical_tensor_t &get_logical_tensor() const {
        return lt_;
    }

    operator bool() const { return handle_ != nullptr; }

    const dnnl::impl::graph::engine_t *get_engine() const { return eng_; }

private:
    static dnnl::impl::graph::status_t dummy_destructor(void *) {
        return dnnl::impl::graph::status::success;
    }

    dnnl::impl::graph::logical_tensor_t lt_
            = dnnl::impl::graph::zero_logical_tensor();
    const dnnl::impl::graph::engine_t *eng_ {nullptr};

    std::shared_ptr<void> handle_ {nullptr};

    union {
        // this field is valid when logical tensor's
        // property is host_scalar
        int32_t s32_value = 0;
        // TODO: add more dtype support
    } scalar_;
};

#endif
