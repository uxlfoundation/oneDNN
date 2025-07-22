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

#include "gpu/intel/l0/context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

event_wrapper_t::event_wrapper_t(ze_context_handle_t context) {
    ze_event_pool_desc_t event_pool_desc = {};
    event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    event_pool_desc.pNext = nullptr;
    event_pool_desc.flags = 0;
    event_pool_desc.count = 1;
    func_zeEventPoolCreate(context, &event_pool_desc, 0, nullptr, &event_pool_);

    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.pNext = nullptr;
    event_desc.index = 0;
    event_desc.signal = 0;
    event_desc.wait = 0;
    func_zeEventCreate(event_pool_, &event_desc, &event_);
}

event_wrapper_t::~event_wrapper_t() {
    if (event_pool_) {
        func_zeEventDestroy(event_);
        func_zeEventPoolDestroy(event_pool_);
    }
}

void event_t::get_l0_events(std::vector<ze_event_handle_t> &l0_event) const {
    std::transform(events.cbegin(), events.cend(), std::back_inserter(l0_event),
            [](const event_wrapper_t &e) { return e.get(); });
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
