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

#include "utils/check.hpp"
#include "utils/engine.hpp"

runtime_kind_t default_runtime_kind {runtime_kind_t::undefined};
runtime_kind_t check_ref_impl {default_runtime_kind};

std::ostream &operator<<(std::ostream &s, runtime_kind_t runtime_kind) {
    if (runtime_kind == runtime_kind_t::undefined)
        s << "0";
    else if (runtime_kind == runtime_kind_t::cpu)
        s << "cpu";
    else if (runtime_kind == runtime_kind_t::gpu)
        s << "gpu";
    else if (runtime_kind == runtime_kind_t::all)
        s << "1";
    return s;
}

int check_ref_impl_hit(res_t *res) {
    if (check_ref_impl == default_runtime_kind) return OK;

    // Nvidia, AMD and Generic backends use reference implementations to fill
    // gaps in feature support.
    if (is_nvidia_gpu() || is_amd_gpu() || is_generic_gpu()) return OK;

    // Check that requested runtime corresponds to the primary engine.
    if (check_ref_impl == runtime_kind_t::cpu && !is_cpu()) return OK;
    if (check_ref_impl == runtime_kind_t::gpu && !is_gpu()) return OK;

    const auto &impl_name = res->impl_name;
    if (impl_name.find("ref") != std::string::npos) {
        res->state = FAILED;
        res->reason = reason_t::failed_ref_not_expected;
        return FAIL;
    }
    return OK;
}
