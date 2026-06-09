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

#ifndef UTILS_ENGINE_HPP
#define UTILS_ENGINE_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "common.hpp"
#include "dnnl_debug.hpp"

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            if ((s) == CRIT || (s) == WARN) { \
                BENCHDNN_PRINT(0, \
                        "Error: Function '%s' at (%s:%d) returned '%s'\n", \
                        __FUNCTION__, __FILE__, __LINE__, \
                        status2str(status__)); \
                fflush(0); \
                if ((s) == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status__ = (f); \
        if (status__ != dnnl_success) { \
            BENCHDNN_PRINT(0, \
                    "Error: Function '%s' at (%s:%d) returned '%s'\n", \
                    __FUNCTION__, __FILE__, __LINE__, status2str(status__)); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

// Unlike `DNN_SAFE` this one returns `dnnl_status_t`, not `OK/FAIL`.
#define DNN_SAFE_STATUS(f) \
    do { \
        dnnl_status_t status__ = (f); \
        if (status__ != dnnl_success) { return status__; } \
    } while (0)

extern dnnl_engine_kind_t engine_tgt_kind;
extern size_t engine_index;

struct engine_t {
    engine_t(dnnl_engine_kind_t engine_kind);
    engine_t(dnnl_engine_t engine);
    engine_t(const dnnl::engine &engine);
    // `recreate_on_copy=true` forces to construct a brand new engine_t object
    // with underlying interop objects from `other`.
    // When `recreate_on_copy=false`, copy follows a weak_ptr semantics that
    // `dnnl::engine` provides.
    engine_t(const engine_t &other, bool recreate_on_copy = false);
    operator dnnl_engine_t() const { return engine_.get(); }
    operator const dnnl::engine &() const { return engine_; }

    bool is_cpu() const;
    bool is_gpu() const;

private:
    dnnl::engine::kind get_kind() const;
    engine_t &operator=(engine_t &other) = delete;
    dnnl::engine engine_;
};

// Engine used to run oneDNN primitives for testing.
const engine_t &get_test_engine();

// Engine used to run all reference native implementations and CPU
// implementations used by `--fast-ref` option.
const engine_t &get_cpu_engine();

bool is_cpu(const engine_t &engine = get_test_engine());
bool is_gpu(const engine_t &engine = get_test_engine());
bool is_async(const engine_t &engine = get_test_engine());
bool is_sycl_engine(const engine_t &engine = get_test_engine());
bool is_opencl_engine(const engine_t &engine = get_test_engine());
bool is_ze_engine(const engine_t &engine = get_test_engine());
bool is_nvidia_gpu(const engine_t &engine = get_test_engine());
bool is_amd_gpu(const engine_t &engine = get_test_engine());
bool is_generic_gpu(const engine_t &engine = get_test_engine());

#endif
