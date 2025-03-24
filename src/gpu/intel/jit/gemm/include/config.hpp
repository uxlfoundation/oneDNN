/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_CONFIG_HPP
#define GEMMSTONE_GUARD_CONFIG_HPP

#include "common/verbose.hpp"

#include "internal/namespace_start.hxx"

enum class GEMMVerbose {
    DebugInfo = dnnl::impl::verbose_t::debuginfo
};

inline int getVerbose(GEMMVerbose v) {
    return dnnl::impl::get_verbose(static_cast<dnnl::impl::verbose_t::flag_kind>(v));
}

template <typename... Args>
inline void verbosePrintf(const char *fmtStr, Args... args) {
    return dnnl::impl::verbose_printf(fmtStr, args...);
}

#include "internal/namespace_end.hxx"

#endif /* header guard */
