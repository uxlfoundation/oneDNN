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

#include "gpu/intel/l0/compiler.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#else
#include <dlfcn.h>
#endif

#include "ocloc_api.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

inline void *find_symbol(const char *symbol) {
#ifdef _WIN32
    static const char *ocloc = "ocloc64.dll";
    HMODULE handle = GetModuleHandleA(ocloc);
    if (!handle) return nullptr;
    return reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#else
    static const char *ocloc = "libocloc.so";
    void *handle = dlopen(ocloc, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return nullptr;
    return dlsym(handle, symbol);
#endif
}

template <typename F>
F find_symbol(const char *symbol) {
    return (F)find_symbol(symbol);
}

status_t ocloc_invoke(uint32_t NumArgs, const char *Argv[], uint32_t NumSources,
        const uint8_t **DataSources, const uint64_t *LenSources,
        const char **NameSources, uint32_t NumInputHeaders,
        const uint8_t **DataInputHeaders, const uint64_t *LenInputHeaders,
        const char **NameInputHeaders, uint32_t *NumOutputs,
        uint8_t ***DataOutputs, uint64_t **LenOutputs, char ***NameOutputs) {
    static auto f = find_symbol<decltype(&oclocInvoke)>("oclocInvoke");
    if (!f) return status::runtime_error;

    if (f(NumArgs, Argv, NumSources, DataSources, LenSources, NameSources,
                NumInputHeaders, DataInputHeaders, LenInputHeaders,
                NameInputHeaders, NumOutputs, DataOutputs, LenOutputs,
                NameOutputs))
        return status::runtime_error;

    return status::success;
}

status_t ocloc_free(uint32_t *numOutputs, uint8_t ***dataOutputs,
        uint64_t **lenOutputs, char ***nameOutputs) {
    static auto f = find_symbol<decltype(&oclocFreeOutput)>("oclocFreeOutput");
    if (!f) return status::runtime_error;

    if (f(numOutputs, dataOutputs, lenOutputs, nameOutputs))
        return status::runtime_error;

    return status::success;
}

status_t ocloc_get_extensions(std::string &extensions) {
    std::vector<const char *> args = {"ocloc", "query", "CL_DEVICE_EXTENSIONS"};

    uint32_t num_outputs = 0;
    uint8_t **data_outputs = nullptr;
    uint64_t *len_outputs = nullptr;
    char **name_outputs = nullptr;

    CHECK(ocloc_invoke(static_cast<uint32_t>(args.size()), args.data(), 0,
            nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, &num_outputs,
            &data_outputs, &len_outputs, &name_outputs));

    for (uint32_t i = 0; i < num_outputs; i++) {
        if (!strcmp(name_outputs[i], "stdout.log")) {
            if (len_outputs[i] > 0) {
                extensions = std::string(
                        reinterpret_cast<const char *>(data_outputs[i]));
                break;
            }
        }
    }

    CHECK(ocloc_free(&num_outputs, &data_outputs, &len_outputs, &name_outputs));

    return status::success;
}

bool ocloc_mayiuse_microkernels(const std::string &kernel_code) {
    std::vector<const char *> args
            = {"ocloc", "compile", "-q", "-file", "test.cl"};
    const uint8_t *data_sources[]
            = {reinterpret_cast<const uint8_t *>(kernel_code.c_str())};
    const uint64_t len_sources[] = {kernel_code.length() + 1};
    const char *name_sources[] = {"test.cl"};

    uint32_t num_outputs = 0;
    uint8_t **data_outputs = nullptr;
    uint64_t *len_outputs = nullptr;
    char **name_outputs = nullptr;

    bool compilation_successful = true;
    if (ocloc_invoke(static_cast<uint32_t>(args.size()), args.data(), 1,
                data_sources, len_sources, name_sources, 0, nullptr, nullptr,
                nullptr, &num_outputs, &data_outputs, &len_outputs,
                &name_outputs))
        compilation_successful = false;
    ocloc_free(&num_outputs, &data_outputs, &len_outputs, &name_outputs);

    return compilation_successful;
}

status_t ocloc_build_kernels(const std::string &kernel_code,
        const std::string &options, const std::string &ip_version,
        xpu::binary_t &binary) {
    std::vector<const char *> args = {"ocloc", "compile", "-q", "--format",
            "zebin", "-exclude_ir", "-output_no_suffix", "-file", "main.cl",
            "-device", ip_version.c_str(), "-options", options.c_str()};
    const uint8_t *data_sources[]
            = {reinterpret_cast<const uint8_t *>(kernel_code.c_str())};
    const uint64_t len_sources[] = {kernel_code.length() + 1};
    const char *name_sources[] = {"main.cl"};

    uint32_t num_outputs = 0;
    uint8_t **data_outputs = nullptr;
    uint64_t *len_outputs = nullptr;
    char **name_outputs = nullptr;

    status_t ret = ocloc_invoke(static_cast<uint32_t>(args.size()), args.data(),
            1, data_sources, len_sources, name_sources, 0, nullptr, nullptr,
            nullptr, &num_outputs, &data_outputs, &len_outputs, &name_outputs);
    if (ret != status::success) {
        std::string output_string;
        for (uint32_t i = 0; i < num_outputs; i++) {
            if (!strcmp(name_outputs[i], "stdout.log")) {
                if (len_outputs[i] > 0) {
                    output_string = std::string(
                            reinterpret_cast<const char *>(data_outputs[i]));
                }
            }
        }
        CHECK(ocloc_free(
                &num_outputs, &data_outputs, &len_outputs, &name_outputs));
        throw std::runtime_error(output_string);
    }

    for (uint32_t i = 0; i < num_outputs; i++) {
        if (!strcmp(name_outputs[i], "main.bin")) {
            if (len_outputs[i] > 0) {
                binary.resize(len_outputs[i]);
                std::memcpy(binary.data(), data_outputs[i], len_outputs[i]);
                break;
            }
        }
    }

    CHECK(ocloc_free(&num_outputs, &data_outputs, &len_outputs, &name_outputs));

    return status::success;
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
