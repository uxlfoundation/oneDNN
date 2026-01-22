/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "xpu/utils.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <tuple>
#include <vector>

namespace dnnl {
namespace impl {
namespace xpu {

#ifndef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
size_t device_uuid_hasher_t::operator()(const device_uuid_t &uuid) const {
    const size_t seed = hash_combine(0, std::get<0>(uuid));
    return hash_combine(seed, std::get<1>(uuid));
}
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

void *find_symbol(const char *library_name, const char *symbol) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryExA(
            library_name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!handle) {
        LPSTR error_text = nullptr;
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM
                        | FORMAT_MESSAGE_ALLOCATE_BUFFER
                        | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&error_text,
                0, nullptr);
        VERROR(common, runtime, "error while opening %s library: %s",
                library_name, error_text);
        LocalFree(error_text);
        return nullptr;
    }
    void *symbol_address
            = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
    if (!symbol_address) {
        LPSTR error_text = nullptr;
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM
                        | FORMAT_MESSAGE_ALLOCATE_BUFFER
                        | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&error_text,
                0, nullptr);
        VERROR(common, runtime,
                "error while searching for a %s symbol address in %s library: "
                "%s",
                symbol, library_name, error_text);
        LocalFree(error_text);
        return nullptr;
    }
    return symbol_address;
#elif defined(__linux__)
    // To clean the error string
    dlerror();
    void *handle = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        VERROR(common, runtime, "error while opening %s library: %s",
                library_name, dlerror());
        return nullptr;
    }
    // To clean the error string
    dlerror();
    void *symbol_address = dlsym(handle, symbol);
    if (!symbol_address) {
        VERROR(common, runtime,
                "error while searching for a %s symbol address in %s library: "
                "%s",
                symbol, library_name, dlerror());
        // See a comment below.
        // dlclose(handle);
        return nullptr;
    }
    // Note: `dlclose` invalidates `symbol_address` if the application hadn't
    // had a `handle` opened before. The solution to put a handle in some
    // `static` object leads to other problems with unloading a library from
    // applications linked with oneDNN when it comes to managing global
    // resources passed between libraries such as contexts.
    //
    // Thus, the recommendation is not to `dlclose` the handle and let it slide.
    //
    // dlclose(handle);
    return symbol_address;
#endif
}

} // namespace xpu
} // namespace impl
} // namespace dnnl
