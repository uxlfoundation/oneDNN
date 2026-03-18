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

#ifndef XPU_ZE_ENGINE_FACTORY_HPP
#define XPU_ZE_ENGINE_FACTORY_HPP

#include "common/engine.hpp"

#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

class engine_factory_t : public impl::engine_factory_t {
public:
    engine_factory_t(engine_kind_t engine_kind);

    ~engine_factory_t() override = default;

    size_t count() const override;

    status_t engine_create(
            impl::engine_t **engine, size_t index) const override;
    status_t engine_create(impl::engine_t **engine, ze_driver_handle_t adriver,
            ze_device_handle_t adevice, ze_context_handle_t acontext,
            size_t index) const;

private:
    engine_factory_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_factory_t);
};

inline std::unique_ptr<engine_factory_t> get_engine_factory(
        engine_kind_t engine_kind) {
    return std::unique_ptr<engine_factory_t>(new engine_factory_t(engine_kind));
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_ENGINE_FACTORY_HPP
