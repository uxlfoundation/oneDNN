/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_PRIMITIVE_ATTR_HPP
#define GPU_INTEL_PRIMITIVE_ATTR_HPP

#include "common/primitive_attr.hpp"
#include "common/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_primitive_attr_t : public primitive_attr_item_t {
    gpu_primitive_attr_t(int grf_per_thread = 0)
        : grf_per_thread_(grf_per_thread) {}

    std::unique_ptr<primitive_attr_item_t> clone() const override {
        return utils::make_unique<gpu_primitive_attr_t>(grf_per_thread_);
    }

    bool has_default_values() const override { return grf_per_thread_ == 0; }

    bool is_equal(const primitive_attr_item_t &other) const override {
        auto *other_ptr = utils::downcast<const gpu_primitive_attr_t *>(&other);
        return grf_per_thread_ == other_ptr->grf_per_thread_;
    }

    size_t get_hash() const override { return grf_per_thread_; }

    void serialize(serialization_stream_t &stream) const override {
        stream.append(grf_per_thread_);
    }

    int grf_per_thread() const { return grf_per_thread_; }

private:
    int grf_per_thread_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
