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

#if (defined(__has_include) && __has_include("gemmstone_config.hpp")) || defined(GEMMSTONE_CONFIG)
#include "gemmstone_config.hpp"
#else

#include "entrance_agent.hpp"
#include "package.hpp"

#endif

#ifndef GEMMSTONE_NAMESPACE_START
#define GEMMSTONE_NAMESPACE_START namespace gemmstone {
#endif

#ifndef GEMMSTONE_NAMESPACE_END
#define GEMMSTONE_NAMESPACE_END }
#endif

#endif /* header guard */
