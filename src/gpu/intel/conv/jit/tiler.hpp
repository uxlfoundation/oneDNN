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

#ifndef GPU_INTEL_CONV_JIT_TILER_HPP
#define GPU_INTEL_CONV_JIT_TILER_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

class config_t;
class tuner_t;
class tiler_impl_t;

class tiler_t {
public:
    tiler_t(const config_t &cfg);
    void set_tuner(tuner_t *tuner);
    int configs() const;
    bool is_tuning_mode() const;
    bool is_valid() const;
    void move_next(const config_t &cfg);
    int32_t cur_version() const;
    void set_cur_version(int32_t idx);
    void set_params(config_t &cfg);
    void notify_out_of_registers(const config_t &cfg);
    bool is_grf_limit_ok(const config_t &cfg) const;
    static void after_create_hook(
            const config_t &cfg, const impl::primitive_t *primitive);
    static void before_exec_hook(
            const impl::primitive_t *primitive, impl::stream_t *stream);

private:
    std::shared_ptr<tiler_impl_t> impl_;
};

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
