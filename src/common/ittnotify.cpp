/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "ittnotify.hpp"
#include "utils.hpp"
#include "common/profiler.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "dnnl_debug.h"
#include "ittnotify/ittnotify.h"
#endif

namespace dnnl {
namespace impl {
namespace itt {

static setting_t<int> itt_task_level {__itt_task_level_high};

bool get_itt(__itt_task_level level) {
    if (!itt_task_level.initialized()) {
        // Assumes that all threads see the same environment
        static int val
                = getenv_int_user("ITT_TASK_LEVEL", itt_task_level.get());
        itt_task_level.set(val);
    }
    return level <= itt_task_level.get();
}

#if defined(DNNL_ENABLE_ITT_TASKS)

namespace {

thread_local primitive_kind_t thread_primitive_kind;
thread_local const char *thread_primitive_info;
thread_local const char *thread_primitive_log_kind;
__itt_string_handle *thread_primitive_meta_fmt
        = __itt_string_handle_create("%s");

__itt_domain *itt_domain(const char *log_kind) {

    const bool is_exec = (std::strncmp(log_kind, "exec", sizeof("exec")) == 0);
    assert(is_exec
            || (std::strncmp(log_kind, "create", sizeof("create")) == 0));

    if (is_exec) {
        static __itt_domain *d_exec
                = __itt_domain_create("dnnl::primitive::execute");
        return d_exec;
    } else {
        static __itt_domain *d_create
                = __itt_domain_create("dnnl::primitive::create");
        return d_create;
    }
}

} // namespace

void primitive_task_start(primitive_kind_t kind, const char *pd_info,
        const char *log_kind, double *mss[4]) {
    if (kind == primitive_kind::undefined) return;
    static double ms_domain = 0;
    auto ms3 = get_msec();
    __itt_domain *pd_domain = itt_domain(log_kind);
    auto ms4 = get_msec();

#define CASE(x) \
    __itt_string_handle_create(dnnl_prim_kind2str(primitive_kind::x))
    static __itt_string_handle *prim_kind_itt_strings[] = {
            CASE(undefined),
            CASE(reorder),
            CASE(shuffle),
            CASE(concat),
            CASE(sum),
            CASE(convolution),
            CASE(deconvolution),
            CASE(eltwise),
            CASE(lrn),
            CASE(batch_normalization),
            CASE(inner_product),
            CASE(rnn),
            CASE(gemm),
            CASE(binary),
            CASE(matmul),
            CASE(resampling),
            CASE(pooling),
            CASE(reduction),
            CASE(prelu),
            CASE(softmax),
            CASE(layer_normalization),
            CASE(group_normalization),
            CASE(sdpa),
    };
#undef CASE
    static double ms_begin = 0;
    auto ms5 = get_msec();
    int kind_idx = (int)kind;
    assert(kind_idx >= 0);
    if (kind_idx < primitive_kind::internal_only_start) {
        assert((size_t)kind_idx < sizeof(prim_kind_itt_strings)
                        / sizeof(prim_kind_itt_strings[0]));
        __itt_task_begin(pd_domain, __itt_null, __itt_null,
                prim_kind_itt_strings[kind_idx]);
    }
    auto ms6 = get_msec();

    static double ms_meta = 0;
    static double ms_upd = 0;
    auto ms0 = get_msec();
    __itt_formatted_metadata_add(pd_domain, thread_primitive_meta_fmt, pd_info);
    auto ms1 = get_msec();
    thread_primitive_kind = kind;
    thread_primitive_info = pd_info;
    thread_primitive_log_kind = log_kind;
    auto ms2 = get_msec();
    if (!mss) return;

    ms_meta += ms1 - ms0;
    ms_upd += ms2 - ms1;
    ms_domain += ms4 - ms3;
    ms_begin += ms6 - ms5;

    if (mss[0]) { (*mss)[0] = ms_meta; }
    if (mss[1]) { (*mss)[1] = ms_upd; }
    if (mss[2]) { (*mss)[2] = ms_domain; }
    if (mss[3]) { (*mss)[3] = ms_begin; }
}

primitive_kind_t primitive_task_get_current_kind() {
    return thread_primitive_kind;
}

const char *primitive_task_get_current_info() {
    return thread_primitive_info;
}

const char *primitive_task_get_current_log_kind() {
    return thread_primitive_log_kind;
}

void primitive_task_end(const char *log_kind) {
    if (thread_primitive_kind != primitive_kind::undefined) {
        __itt_task_end(itt_domain(log_kind));
        thread_primitive_kind = primitive_kind::undefined;
        thread_primitive_info = nullptr;
        thread_primitive_log_kind = nullptr;
    }
}
#else
void primitive_task_start(primitive_kind_t kind) {
    UNUSED(kind);
}
primitive_kind_t primitive_task_get_current_kind() {
    return primitive_kind::undefined;
}
void primitive_task_end(const char *log_kind) {}
#endif

} // namespace itt
} // namespace impl
} // namespace dnnl
