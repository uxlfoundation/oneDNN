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

#include <cstring>
#include <tuple>
#include <vector>

#include "common/verbose.hpp"

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
    int revision; // Optional 4th field, e.g. 32.0.101.<revision> on Windows.

    runtime_version_t(
            int major = 0, int minor = 0, int build = 0, int revision = 0)
        : major {major}, minor {minor}, build {build}, revision {revision} {}

    bool operator==(const runtime_version_t &other) const {
        return std::tie(major, minor, build, revision)
                == std::tie(
                        other.major, other.minor, other.build, other.revision);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        return std::tie(major, minor, build, revision) < std::tie(
                       other.major, other.minor, other.build, other.revision);
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

        revision = 0;
        i += (int)strspn(&s[i], "0123456789");
        if (s[i] != '.') return status::success;

        // Expect a 1-5 digit revision, e.g. 32.0.101.8970 on Windows.
        auto len = strspn(&s[++i], "0123456789");
        bool ok = len >= 1 && len <= 5 && s[i + len] != '.';
        assert(ok && "unexpected driver version format");
        if (ok) revision = atoi(&s[i]);

        return status::success;
    }

    std::string str() const {
        auto s = utils::format("%d.%d.%d", major, minor, build);
        if (revision) s += utils::format(".%d", revision);
        return s;
    }
};

void *find_symbol(const char *library_name, const char *symbol);

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
