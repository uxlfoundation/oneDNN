/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef NGEN_DYNAMIC_HPP
#define NGEN_DYNAMIC_HPP

#include "ngen_config_internal.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#else
#include <dlfcn.h>
#endif

namespace NGEN_NAMESPACE {
namespace dynamic {

inline void *findSymbol(const char *lib, const char *symbol)
{
    // In nGEN usage, the caller has always initialized the runtime library (OCL, L0)
    //   prior to invoking nGEN, and is responsible for the lifetime of this library.
    // Hence we can always rely here on the library being loaded and initialized.

#ifdef _WIN32
    HMODULE handle = GetModuleHandleA(lib);
    if (!handle) return nullptr;
    return reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#else
    void *handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return nullptr;
    return dlsym(handle, symbol);
#endif
}

} /* namespace dynamic */
} /* namespace NGEN_NAMESPACE */

#endif
