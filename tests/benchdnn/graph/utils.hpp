/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_UTILS_HPP
#define BENCHDNN_GRAPH_UTILS_HPP

#include "dnnl_memory.hpp"
#include "utils/engine.hpp"
#include "utils/timer.hpp"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include <string>
#include <utility>
#include <vector>

namespace graph {

struct deserialized_lt_t;

struct bdnn_state_t {
    res_state_t state;
    reason_t reason;
};

enum class dnnl_driver_t {
    binary,
    bnorm,
    concat,
    conv,
    custom,
    deconv,
    eltwise,
    lnorm,
    matmul,
    pool,
    prelu,
    reduction,
    reorder,
    resampling,
    softmax,
    gnorm,
    others
};

enum class graph_recognized_pattern_t {
    ordinary,
    sdpa_fwd,
    sdpa_bwd,
    gmlp,
};

extern bdnn_state_t convert_state(const dnnl_status_t &s);

// Flags that controls the behavior for handling exceptions. The logic
// relies on the fact that the values not intersect with each other.
enum { CRIT = 0x001, WARN = 0x002, NEED_CLEANUP = 0x004 };

// For now, there are some feature gaps between Nvidia GPU and Intel GPU, as
// fitting those gaps is a long-term task, many cases will be unsupported on
// Nvidia GPU, here we directly skip those cases on Nvidia GPU.
#define DNN_GRAPH_SAFE(f, s, ss) \
    do { \
        try { \
            (f); \
        } catch (const dnnl::error &e) { \
            if (((s) & CRIT) || ((s) & WARN)) { \
                bdnn_state_t bs = convert_state(e.status); \
                (ss)->state = bs.state; \
                if ((ss)->state == res_state_t::SKIPPED) { \
                    (ss)->reason = bs.reason; \
                } else if ((((ss)->state == res_state_t::UNIMPLEMENTED) \
                                   || ((ss)->state \
                                           == res_state_t::INVALID_ARGUMENTS)) \
                        && is_nvidia_gpu()) { \
                    (ss)->state = SKIPPED; \
                    (ss)->reason = reason_t::skip_not_supported; \
                    BENCHDNN_PRINT(2, \
                            "SKIP: Function '%s' at (%s:%d) returned '%s'\n", \
                            __FUNCTION__, __FILE__, __LINE__, e.what()); \
                } else { \
                    BENCHDNN_PRINT(0, \
                            "Error: Function '%s' at (%s:%d) returned '%s'\n", \
                            __FUNCTION__, __FILE__, __LINE__, e.what()); \
                } \
                fflush(0); \
                if ((s) & CRIT) exit(2); \
            } \
            if (!((s) & NEED_CLEANUP)) return FAIL; \
        } \
    } while (0)

using perf_function_t = std::function<void(dnnl::stream &,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs,
        const dnnl::graph::tensor &scratchpad)>;

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::stream &stream, const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs,
        const dnnl::graph::tensor &scratchpad);

int measure_perf(timer::timer_t &t,
        const std::vector<dnnl::graph::compiled_partition> &cp_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &inputs_v,
        const std::vector<std::vector<dnnl::graph::tensor>> &outputs_v,
        const std::vector<dnnl::graph::tensor> &scratchpads, res_t *res);

dnnl::graph::op::kind opstr2kind(const std::string &kind);
bool is_unary(const std::string &kind);
bool is_unary(dnnl::graph::op::kind akind);
bool is_backward(const std::string &kind);
bool is_backward(dnnl::graph::op::kind akind);
dnnl::graph::op::attr attrstr2kind(const std::string &attr_name);
const std::string &attrstr2type(const std::string &attr_name);

std::string get_default_tag(size_t length);
std::string strides2memory_tag(const dnnl::graph::logical_tensor::dims &dims,
        const dnnl::graph::logical_tensor::dims &strides,
        bool use_x_tag = true);
dnnl::graph::logical_tensor::dims memory_tag2strides(
        const dnnl::graph::logical_tensor::dims &shape, const std::string &tag);
bool is_contiguous_memory(const dnnl::graph::logical_tensor::dims &strides,
        const dnnl::graph::logical_tensor::dims &shape, const std::string &tag);

inline bool is_plain(dnnl_format_tag_t fmt_tag) {
    return fmt_tag >= dnnl_a && fmt_tag <= dnnl_abcdefghijlk;
}

dnnl::graph::op::kind opstr2kind(const std::string &kind);
dnnl::graph::op::attr attrstr2kind(const std::string &attr_name);

// permute md based on permutation
void permute_md(dnn_mem_t &mem, std::vector<int64_t> permutation);

// get primitive's arg name according to graph op's output offset
// i.e. If BatchNormForwardTraining's 2-nd output is ReLU's 1-st input
//      the output offset of 2 needs to be mapped to primitive's
//      output arg of DNNL_ARG_VARIANCE
int get_prim_arg_name_from_graph_op_output_offset(
        dnnl::graph::op::kind op_kind, size_t output_offset);
// get primitive's arg name according to graph op's input offset
int get_prim_arg_name_from_graph_op_input_offset(dnnl::graph::op::kind op_kind,
        size_t input_offset, bool use_dst = false);

/// Get logical tensor layout type based on string
///
/// @param layout_type a string of layout type from deserialized
/// logical tensor
dnnl::graph::logical_tensor::layout_type str2layout(
        const std::string &layout_type);

void change_format_to_ncx(dims_t &dims);

// For a given vector of partitions provide a string with number of ops in
// every partition in format: `{N} {M} ...`.
std::string verbose_partitions_n_ops(
        const std::vector<dnnl::graph::partition> &partitions);

// Returns logical dims as a string object in dims_t format
std::string lt_dims2str(const dnnl::graph::logical_tensor::dims &dims);

template <typename First, typename... Rest>
void change_format_to_ncx(First &first, Rest &...rest) {
    change_format_to_ncx(first);
    change_format_to_ncx(rest...);
}

// Creates the graph engine wrapped into the common `engine_t` abstraction. The
// graph library requires an allocator attached to the engine to allocate memory
// for constant cache and scratchpad, hence the engine is built via
// `make_engine_with_allocator` and adopted by `engine_t`.
engine_t make_graph_engine(bool use_host);

// Engine used for the graph library. It wraps a `dnnl::engine` created with an
// allocator (see `make_graph_engine`) into the common `engine_t` abstraction so
// that the graph driver shares the same engine type as the rest of benchdnn
// while still exposing the C++ engine interface required by the graph API.
inline const engine_t &get_graph_engine() {
    static const engine_t instance(make_graph_engine(/*use_host*/ false));
    return instance;
}

inline const engine_t &get_graph_host_engine() {
    // return `get_graph_engine` for `is_cpu` to avoid different engine instances.
    const dnnl::engine &g_eng = get_graph_engine();
    if (is_cpu(g_eng.get())) { return get_graph_engine(); }

    static const engine_t instance(make_graph_engine(/*use_host*/ true));
    return instance;
}

dnnl_data_type_t convert_dt(const dnnl::graph::logical_tensor::data_type dt);

inline double GB(double bytes) {
    return bytes / powf(2, 30);
}

struct graph_fpmath_mode_t {
    graph_fpmath_mode_t() = default;
    graph_fpmath_mode_t(const std::string &mode, bool apply_to_int,
            bool override_json_value)
        : mode_(mode)
        , apply_to_int_(apply_to_int)
        , override_json_value_(override_json_value) {}

    bool operator==(const graph_fpmath_mode_t &rhs) const {
        return mode_ == rhs.mode_ && apply_to_int_ == rhs.apply_to_int_
                && override_json_value_ == rhs.override_json_value_;
    }

    std::string mode_ = "strict";
    bool apply_to_int_ = false;
    // Since fpmath_mode doesn't provide an "undef" value that would indicate
    // it was not set externally to the json case, need to maintain this flag.
    bool override_json_value_ = false;
};

} // namespace graph
#endif
