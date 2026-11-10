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

#include "utils/ze_loader.hpp"

#ifdef BENCHDNN_WITH_ZE_LOADER
#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#endif

void *load_ze_symbol(const char *name) {
#ifdef _WIN32
    static HMODULE lib = [] {
        HMODULE h = GetModuleHandleA("ze_loader.dll");
        return h ? h
                 : LoadLibraryExA("ze_loader.dll", nullptr,
                           LOAD_LIBRARY_SEARCH_SYSTEM32);
    }();
    return lib ? reinterpret_cast<void *>(GetProcAddress(lib, name)) : nullptr;
#else
    static void *lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? dlsym(lib, name) : nullptr;
#endif
}
#endif
