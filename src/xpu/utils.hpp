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

#ifndef XPU_UTILS_HPP
#define XPU_UTILS_HPP

#include <tuple>
#include <vector>

#include "common/verbose.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#elif defined(__linux__)
#include <dlfcn.h>
#endif

// This file contains utility functionality for heterogeneous runtimes such
// as OpenCL and SYCL.

namespace dnnl {
namespace impl {
namespace xpu {

using binary_t = std::vector<uint8_t>;
using device_uuid_t = std::tuple<uint64_t, uint64_t>;

#ifndef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
struct device_uuid_hasher_t {
    size_t operator()(const device_uuid_t &uuid) const;
};
#endif // DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER

struct runtime_version_t {
    int major;
    int minor;
    int build;

    runtime_version_t(int major = 0, int minor = 0, int build = 0)
        : major {major}, minor {minor}, build {build} {}

    bool operator==(const runtime_version_t &other) const {
        return (major == other.major) && (minor == other.minor)
                && (build == other.build);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        if (major < other.major) return true;
        if (major > other.major) return false;
        if (minor < other.minor) return true;
        if (minor > other.minor) return false;
        return (build < other.build);
    }

    bool operator>(const runtime_version_t &other) const {
        return (other < *this);
    }

    bool operator<=(const runtime_version_t &other) const {
        return !(*this > other);
    }

    bool operator>=(const runtime_version_t &other) const {
        return !(*this < other);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }

    std::string str() const {
        return utils::format("%d.%d.%d", major, minor, build);
    }
};

#if defined(_WIN32)
inline void *find_symbol(const char *library_name, const char *symbol) {
    HMODULE handle = LoadLibraryExA(
            library_name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!handle) {
        LPSTR error_text = nullptr;
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM
                        | FORMAT_MESSAGE_ALLOCATE_BUFFER
                        | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr, GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), error_text, 0,
                nullptr);
        VERROR(common, runtime, "error while opening %s library: %s",
                library_name, error_text);
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
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), error_text, 0,
                nullptr);
        VERROR(common, runtime,
                "error while searching for a %s symbol address in %s library: "
                "%s",
                symbol, library_name, error_text);
        return nullptr;
    }
    return symbol_address;
}
#elif defined(__linux__)
inline void *find_symbol(const char *library_name, const char *symbol) {
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
        return nullptr;
    }
    return symbol_address;
}
#endif

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
