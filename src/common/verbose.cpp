/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#include <atomic>
#include <regex>
#include <sstream>
#include <type_traits>

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl_version_hash.h"

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/verbose.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "xpu/sycl/verbose.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "xpu/ze/verbose.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL
#include "common/experimental.hpp"
#endif

#ifdef ONEDNN_BUILD_GRAPH
#include "graph/utils/verbose.hpp"
#endif

namespace dnnl {
namespace impl {

// Versioning is used to communicate breaking verbose lines changes to
// verbose_converter tool for maintaining compatibility smoother.
// Numeration uses integers only and goes linearly from 0 to infinity.
static constexpr char verbose_version[] = "v1";

static setting_t<uint32_t> verbose {0};

// Component filters help manage verbose output by parsing and printing for
// matching components. The filter status is tracked from verbose initializaton,
// allowing queries for the component type during verbose printing.
filter_status_t &filter_status() {
    static filter_status_t filter_status;
    return filter_status;
}

void print_header() noexcept {
    static std::atomic_flag version_printed = ATOMIC_FLAG_INIT;
    if (!version_printed.test_and_set()) {
        verbose_printf("info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        verbose_printf("info,cpu,runtime:%s,nthr:%d\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime),
                dnnl_get_max_threads());
        verbose_printf("info,cpu,isa:%s\n", cpu::platform::get_isa_info());
#endif
        verbose_printf("info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
        // Printing the header generally requires iterating over devices/backends,
        // which may involve an allocation. Use a try/catch block in case
        // these fail (not printing a header is reasonable in this case)
        try {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            xpu::ocl::print_verbose_header();
#endif
#ifdef DNNL_WITH_SYCL
            xpu::sycl::print_verbose_header();
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
            xpu::ze::print_verbose_header();
#endif
#ifdef ONEDNN_BUILD_GRAPH
            graph::utils::print_verbose_header();
#endif
        } catch (...) {
            verbose_printf("info,exception while printing verbose header\n");
        }
#ifdef DNNL_EXPERIMENTAL
        verbose_printf("info,experimental features are enabled\n");
        verbose_printf("info,use batch_normalization stats one pass is %s\n",
                experimental::use_bnorm_stats_one_pass() ? "enabled"
                                                         : "disabled");
        verbose_printf("info,GPU convolution v2 is %s\n",
                experimental::use_gpu_conv_v2() ? "enabled" : "disabled");
#endif
#ifdef DNNL_EXPERIMENTAL_LOGGING
        const log_manager_t &log_manager = log_manager_t::get_log_manager();
        if (log_manager.is_logger_enabled())
            verbose_printf(
                    "info,experimental functionality for logging is enabled\n");
#endif
#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
        verbose_printf("info,experimental SYCL kernel compiler is enabled\n");
#endif
        verbose_printf(
                "primitive,info,template:%soperation,engine,primitive,"
                "implementation,prop_kind,memory_descriptors,attributes,"
                "auxiliary,problem_desc,exec_time\n",
                get_verbose_timestamp() ? "timestamp," : "");

#ifdef ONEDNN_BUILD_GRAPH
        verbose_printf(
                "graph,info,template:%soperation,engine,partition_id,"
                "partition_kind,op_names,data_formats,logical_tensors,fpmath_"
                "mode,implementation,backend,exec_time\n",
                get_verbose_timestamp() ? "timestamp," : "");
#endif
        if (filter_status().status == filter_status_t::flags::valid)
            verbose_printf(
                    "common,info,filter format is enabled, hit components: "
                    "%s\n",
                    filter_status().components.c_str());
        else if (filter_status().status == filter_status_t::flags::invalid)
            verbose_printf(
                    "common,error,filter format is ill-formed and is not "
                    "applied, error: %s\n",
                    filter_status().err_msg.c_str());
    }
}

// hint parameter is the kind of verbose we are querying for
uint32_t get_verbose(verbose_t::flag_kind verbosity_kind,
        component_t::flag_kind filter_kind) noexcept {
#if defined(DISABLE_VERBOSE)
    return verbose_t::none;
#endif
    // we print all verbose by default
    static int flags = component_t::all;

    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        static std::string user_opt = getenv_string_user("VERBOSE");
        auto update_kind = [&](const std::string &s, int &k) {
            // Legacy: we accept values 0,1,2
            // 0 and none erase previously set flags, including error
            if (s == "0" || s == "none") k = verbose_t::none;
            if (s == "1") k |= verbose_t::level1;
            if (s == "2") k |= verbose_t::level2;
            if (s == "all" || s == "-1") k |= verbose_t::all;
            if (s == "error") k |= verbose_t::error;
            if (s == "check")
                k |= verbose_t::create_check | verbose_t::exec_check;
            if (s == "dispatch") k |= verbose_t::create_dispatch;
            if (s == "profile")
                k |= verbose_t::create_profile | verbose_t::exec_profile;
            if (s == "profile_create") k |= verbose_t::create_profile;
            if (s == "profile_exec") k |= verbose_t::exec_profile;
            // Enable profiling to external libraries
            if (s == "profile_externals") k |= verbose_t::profile_externals;
            if (s == "warn") k |= verbose_t::warn;
            // we extract debug info debuginfo=XX. ignore if debuginfo is invalid.
            if (s.rfind("debuginfo=", 0) == 0)
                k |= verbose_t::make_debuginfo(
                        std::strtol(s.c_str() + 10, nullptr, 10));
        };

        auto update_filter = [&](const std::string &s) -> int {
            int k = component_t::none;
            try {
                std::regex regexp = std::regex(s);

#define REGEX_SEARCH(k, component, regexp) \
    if (std::regex_search("" #component "", regexp)) { \
        (k) |= component_t::component; \
        filter_status().components += "" #component ","; \
    }
                REGEX_SEARCH(k, primitive, regexp);
                REGEX_SEARCH(k, reorder, regexp);
                REGEX_SEARCH(k, shuffle, regexp);
                REGEX_SEARCH(k, concat, regexp);
                REGEX_SEARCH(k, sum, regexp);
                REGEX_SEARCH(k, convolution, regexp);
                REGEX_SEARCH(k, deconvolution, regexp);
                REGEX_SEARCH(k, eltwise, regexp);
                REGEX_SEARCH(k, lrn, regexp);
                REGEX_SEARCH(k, batch_normalization, regexp);
                REGEX_SEARCH(k, inner_product, regexp);
                REGEX_SEARCH(k, rnn, regexp);
                REGEX_SEARCH(k, binary, regexp);
                REGEX_SEARCH(k, matmul, regexp);
                REGEX_SEARCH(k, resampling, regexp);
                REGEX_SEARCH(k, pooling, regexp);
                REGEX_SEARCH(k, reduction, regexp);
                REGEX_SEARCH(k, prelu, regexp);
                REGEX_SEARCH(k, softmax, regexp);
                REGEX_SEARCH(k, layer_normalization, regexp);
                REGEX_SEARCH(k, group_normalization, regexp);
                REGEX_SEARCH(k, graph, regexp);
                REGEX_SEARCH(k, gemm_api, regexp);
                REGEX_SEARCH(k, ukernel, regexp);
#undef REGEX_SEARCH
            } catch (const std::exception &e) {
                filter_status().status = filter_status_t::flags::invalid;
                filter_status().err_msg = e.what();
                return component_t::all;
            }

            // filter enabled and at least one component is hit
            if (!filter_status().components.empty()) {
                // pop out the last comma
                filter_status().components.pop_back();
                filter_status().status = filter_status_t::flags::valid;
            } else {
                filter_status().status = filter_status_t::flags::invalid;
                filter_status().err_msg
                        = "component with name \'" + s + "\' not found";
            }
            return k;
        };

        // we always enable error by default
        int val = verbose_t::error;
        for (size_t pos_st = 0, pos_en = user_opt.find_first_of(',', pos_st);
                true; pos_st = pos_en + 1,
                    pos_en = user_opt.find_first_of(',', pos_st)) {
            std::string tok = user_opt.substr(pos_st, pos_en - pos_st);
            // update verbose flags
            update_kind(tok, val);
            // update filter flags
            if (tok.rfind("filter=", 0) == 0) {
                auto filter_str = tok.substr(7);
                if (!filter_str.empty()) { flags = update_filter(filter_str); }
            }
            if (pos_en == std::string::npos) break;
        }

        // We parse for explicit flags
        verbose.set(val);

#ifdef DNNL_EXPERIMENTAL_LOGGING
        const log_manager_t &log_manager = log_manager_t::get_log_manager();
        if (log_manager.is_logger_enabled())
            log_manager.set_log_level(user_opt);
#endif
    }

    int result = verbose.get() & verbosity_kind;
    if (verbosity_kind == verbose_t::debuginfo)
        result = verbose_t::get_debuginfo(verbose.get());
    bool filter_result = flags & filter_kind;
    return filter_result ? result : 0;
}

static setting_t<bool> verbose_timestamp {false};
bool get_verbose_timestamp() {
#if defined(DISABLE_VERBOSE)
    return false;
#endif
    if (verbose.get() == 0) return false;

    if (!verbose_timestamp.initialized()) {
        // Assumes that all threads see the same environment
        static bool val
                = getenv_int_user("VERBOSE_TIMESTAMP", verbose_timestamp.get());
        verbose_timestamp.set(val);
    }
    return verbose_timestamp.get();
}

// Designated function to prepend a verbose marker and correspondent version.
// Note: intended to be called inside `verbose_printf_impl` only!
std::string prepend_identifier_and_version(const char *fmt_str) {
    assert(std::string(fmt_str).find("onednn_verbose") == std::string::npos);
    std::string s
            = "onednn_verbose," + std::string(verbose_version) + "," + fmt_str;
    return s;
}

void verbose_printf_impl(const char *raw_fmt_str, verbose_t::flag_kind kind) {
#if defined(DISABLE_VERBOSE)
    return;
#endif

    if (get_verbose(kind)) print_header();

    const auto &fmt_str = prepend_identifier_and_version(raw_fmt_str);

#ifdef DNNL_EXPERIMENTAL_LOGGING
    const log_manager_t &log_manager = log_manager_t::get_log_manager();

    if (log_manager.is_logger_enabled())
        log_manager.log(fmt_str.c_str(), align_verbose_mode_to_log_level(kind));
    if (log_manager.is_console_enabled()) {
        printf("%s", fmt_str.c_str());
        fflush(stdout);
    }
#else
    printf("%s", fmt_str.c_str());
    fflush(stdout);
#endif
}

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_set_verbose(int level) {
    using namespace dnnl::impl::status;
    using namespace dnnl::impl;
    if (level < 0 || level > 2) return invalid_arguments;

    uint32_t verbose_level = verbose_t::none;
    if (level == 1) verbose_level = verbose_t::level1;
    if (level == 2) verbose_level = verbose_t::level2;
    // we put the lower byte of level as devinfo to preserve backward
    // compatibility with historical VERBOSE={1,2}
    if (level == 1 || level == 2) verbose_level |= (level << 24);
    verbose.set(verbose_level);

#ifdef DNNL_EXPERIMENTAL_LOGGING
    // if logging is enabled, this also updates the logging level for the outputs
    const log_manager_t &log_manager = log_manager_t::get_log_manager();
    if (log_manager.is_logger_enabled())
        log_manager.set_log_level(std::to_string(level));
#endif

    return success;
}

const dnnl_version_t *dnnl_version(void) {
    static const dnnl_version_t ver
            = {DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                    DNNL_VERSION_HASH, DNNL_CPU_RUNTIME, DNNL_GPU_RUNTIME};
    return &ver;
}
