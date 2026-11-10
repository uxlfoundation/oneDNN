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

#ifndef UTILS_ZE_LOADER_HPP
#define UTILS_ZE_LOADER_HPP

#ifdef BENCHDNN_WITH_ZE_LOADER
// Looks up a Level Zero loader symbol at run time, nullptr if unavailable.
void *load_ze_symbol(const char *name);

template <typename F>
F load_ze_fn(const char *name) {
    return reinterpret_cast<F>(load_ze_symbol(name));
}
#endif

#endif
