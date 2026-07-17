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

#include <string>

#include "common/primitive_attr.hpp"
#include "common/serialization.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_primitive_attr_t : public primitive_attr_item_t {
    gpu_primitive_attr_t(
            int grf_per_thread = 0, std::string kernel_override = {})
        : grf_per_thread_(grf_per_thread)
        , kernel_override_(std::move(kernel_override)) {}

    std::unique_ptr<primitive_attr_item_t> clone() const override {
        return utils::make_unique<gpu_primitive_attr_t>(
                grf_per_thread_, kernel_override_);
    }

    bool has_default_values() const override {
        return grf_per_thread_ == 0 && kernel_override_.empty();
    }

    bool is_equal(const primitive_attr_item_t &other) const override {
        auto *other_ptr = utils::downcast<const gpu_primitive_attr_t *>(&other);
        return grf_per_thread_ == other_ptr->grf_per_thread_
                && kernel_override_ == other_ptr->kernel_override_;
    }

    size_t get_hash() const override {
        size_t seed = grf_per_thread_;
        seed = hash_combine(seed, std::hash<std::string> {}(kernel_override_));
        return seed;
    }

    void serialize(serialization_stream_t &stream) const override {
        stream.append(grf_per_thread_);
        stream.append_array(kernel_override_.size(), kernel_override_.c_str());
    }

    int grf_per_thread() const { return grf_per_thread_; }

    const std::string &kernel_override() const { return kernel_override_; }

private:
    int grf_per_thread_;
    std::string kernel_override_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_set_kernel_override(
        dnnl_primitive_attr_t attr, const char *kernel);

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_override(
        const_dnnl_primitive_attr_t attr, const char **kernel);

#endif
