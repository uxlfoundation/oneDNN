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

/// @example cpu_deconv_binary.cpp
/// @brief This example demonstrates INT8 deconvolution + binary post-ops with
/// broadcast, implemented in three ways:
///   1. Graph API (Dequant + ConvTranspose + Binary + Quantize)
///   2. Primitive API with binary post-op (fused int8 deconv)
///   3. Primitive API with separate primitives (unfused)
///
/// Five binary operations are demonstrated:
///   - binary_mul (Multiply)
///   - binary_max (Maximum)
///   - binary_min (Minimum)
///   - binary_div (Divide)
///   - binary_sub (Subtract)
///
/// Quantization config (matching test_convtranspose int8 pattern):
///   src:  u8, scale=1/255, zp=-4 (asymmetric)
///   wei:  s8, scale=1/127 per-channel, zp=0
///   dst:  s8, scale=1, zp=78
///   binary_src1: f32 {1, 1, 1} (broadcast scalar, value=1.0)
///
/// Shapes (1D / 3D tensors):
///   src:  {1, 8, 12}  (NCW)
///   wei:  {8, 8, 3}   (IOW: IC, OC, KW)
///   dst:  {1, 8, 14}  (NCW)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "example_utils.hpp"
#include "graph/graph_example_utils.hpp"

using namespace dnnl;
namespace graph = dnnl::graph;

// Helper: compare two int8 buffers with tolerance
bool allclose_int8(const std::vector<int8_t> &a, const std::vector<int8_t> &b,
        int atol = 1) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (diff > atol) {
            std::cout << "  Mismatch at index " << i << ": "
                      << static_cast<int>(a[i]) << " vs "
                      << static_cast<int>(b[i]) << " (diff=" << diff << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
}

// Helper: map primitive algorithm to graph op kind
graph::op::kind to_graph_binary_kind(algorithm alg) {
    switch (alg) {
        case algorithm::binary_mul: return graph::op::kind::Multiply;
        case algorithm::binary_max: return graph::op::kind::Maximum;
        case algorithm::binary_min: return graph::op::kind::Minimum;
        case algorithm::binary_div: return graph::op::kind::Divide;
        case algorithm::binary_sub: return graph::op::kind::Subtract;
        default:
            throw std::runtime_error("Unsupported binary algorithm for graph");
    }
}

// Helper: algorithm name string
const char *binary_alg_name(algorithm alg) {
    switch (alg) {
        case algorithm::binary_mul: return "Multiply";
        case algorithm::binary_max: return "Maximum";
        case algorithm::binary_min: return "Minimum";
        case algorithm::binary_div: return "Divide";
        case algorithm::binary_sub: return "Subtract";
        default: return "Unknown";
    }
}

// =============================================================================
// Method 1: Graph API
//   Dequantize(u8->f32) -> Dequantize(s8->f32) -> ConvTranspose -> Binary
//       -> Quantize(f32->s8)
// =============================================================================
std::vector<int8_t> run_graph_api(engine &eng, stream &strm,
        const memory::dims &src_shape, const memory::dims &wei_shape,
        const memory::dims &dst_shape, const memory::dims &binary_shape,
        std::vector<uint8_t> &src_data, std::vector<int8_t> &wei_data,
        std::vector<float> &binary_data, float src_scale, int64_t src_zp,
        std::vector<float> &wei_scales, float dst_scale, int64_t dst_zp,
        algorithm binary_alg) {

    using lt = graph::logical_tensor;
    using dt = graph::logical_tensor::data_type;
    using lt_type = graph::logical_tensor::layout_type;

    int64_t OC = wei_shape[1]; // IOW format: {IC, OC, KW}

    // Logical tensors
    lt src_u8_lt {0, dt::u8, src_shape, lt_type::strided};
    lt src_f32_lt {1, dt::f32, src_shape, lt_type::strided};
    lt wei_s8_lt {2, dt::s8, wei_shape, lt_type::strided};
    lt wei_f32_lt {3, dt::f32, wei_shape, lt_type::strided};
    lt deconv_dst_lt {4, dt::f32, dst_shape, lt_type::strided};
    lt binary_src_lt {5, dt::f32, binary_shape, lt_type::strided};
    lt binary_dst_lt {6, dt::f32, dst_shape, lt_type::strided};
    lt dst_s8_lt {7, dt::s8, dst_shape, lt_type::strided};

    // Op 0: Dequantize src (u8 -> f32)
    graph::op dequant_src(0, graph::op::kind::Dequantize, "dequant_src");
    dequant_src.set_attr<std::vector<float>>(
            graph::op::attr::scales, {src_scale});
    dequant_src.set_attr<std::vector<int64_t>>(graph::op::attr::zps, {src_zp});
    dequant_src.set_attr<std::string>(graph::op::attr::qtype, "per_tensor");
    dequant_src.set_attr<int64_t>(graph::op::attr::axis, 0);
    dequant_src.add_input(src_u8_lt);
    dequant_src.add_output(src_f32_lt);

    // Op 1: Dequantize weights (s8 -> f32), per-channel on axis 1 (OC in IOW)
    graph::op dequant_wei(1, graph::op::kind::Dequantize, "dequant_wei");
    dequant_wei.set_attr<std::vector<float>>(
            graph::op::attr::scales, wei_scales);
    dequant_wei.set_attr<std::vector<int64_t>>(
            graph::op::attr::zps, std::vector<int64_t>(OC, 0));
    dequant_wei.set_attr<std::string>(graph::op::attr::qtype, "per_channel");
    dequant_wei.set_attr<int64_t>(graph::op::attr::axis, 1);
    dequant_wei.add_input(wei_s8_lt);
    dequant_wei.add_output(wei_f32_lt);

    // Op 2: ConvTranspose
    graph::op convtranspose_op(
            2, graph::op::kind::ConvTranspose, "convtranspose");
    convtranspose_op.set_attr<std::vector<int64_t>>(
            graph::op::attr::strides, {1});
    convtranspose_op.set_attr<std::vector<int64_t>>(
            graph::op::attr::pads_begin, {0});
    convtranspose_op.set_attr<std::vector<int64_t>>(
            graph::op::attr::pads_end, {0});
    convtranspose_op.set_attr<std::vector<int64_t>>(
            graph::op::attr::dilations, {1});
    convtranspose_op.set_attr<int64_t>(graph::op::attr::groups, 1);
    convtranspose_op.set_attr<std::string>(graph::op::attr::data_format, "NCX");
    convtranspose_op.set_attr<std::string>(
            graph::op::attr::weights_format, "IOX");
    convtranspose_op.add_input(src_f32_lt);
    convtranspose_op.add_input(wei_f32_lt);
    convtranspose_op.add_output(deconv_dst_lt);

    // Op 3: Binary (Multiply or Maximum)
    graph::op binary_op(3, to_graph_binary_kind(binary_alg), "binary");
    binary_op.add_input(deconv_dst_lt);
    binary_op.add_input(binary_src_lt);
    binary_op.add_output(binary_dst_lt);

    // Op 4: Quantize output (f32 -> s8)
    graph::op quant_dst(4, graph::op::kind::Quantize, "quant_dst");
    quant_dst.set_attr<std::vector<float>>(
            graph::op::attr::scales, {dst_scale});
    quant_dst.set_attr<std::vector<int64_t>>(graph::op::attr::zps, {dst_zp});
    quant_dst.set_attr<std::string>(graph::op::attr::qtype, "per_tensor");
    quant_dst.set_attr<int64_t>(graph::op::attr::axis, 0);
    quant_dst.add_input(binary_dst_lt);
    quant_dst.add_output(dst_s8_lt);

    // Build graph
    graph::graph g(engine::kind::cpu);
    g.add_op(dequant_src);
    g.add_op(dequant_wei);
    g.add_op(convtranspose_op);
    g.add_op(binary_op);
    g.add_op(quant_dst);
    g.finalize();

    // Get partitions
    auto partitions = g.get_partitions();
    std::cout << "  Graph partitions: " << partitions.size()
              << ", ops in first: " << partitions[0].get_ops_num() << std::endl;

    // Compile
    auto &partition = partitions[0];
    std::vector<lt> compile_inputs = {src_u8_lt, wei_s8_lt, binary_src_lt};
    std::vector<lt> compile_outputs = {dst_s8_lt};
    auto cp = partition.compile(compile_inputs, compile_outputs, eng);

    // Execute
    std::vector<int8_t> dst_data(product(dst_shape), 0);

    graph::tensor src_ts {compile_inputs[0], eng, src_data.data()};
    graph::tensor wei_ts {compile_inputs[1], eng, wei_data.data()};
    graph::tensor binary_ts {compile_inputs[2], eng, binary_data.data()};
    graph::tensor dst_ts {compile_outputs[0], eng, dst_data.data()};

    cp.execute(strm, {src_ts, wei_ts, binary_ts}, {dst_ts});
    strm.wait();

    return dst_data;
}

// =============================================================================
// Method 2: Primitive API - Int8 deconv with binary post-op (fused)
//   u8 src + s8 weights -> s8 dst, with scales, zero_points, binary post-op
//   Uses format_tag::any to let the library pick optimal layouts, then
//   reorders from plain format to optimal before execution.
// =============================================================================
std::vector<int8_t> run_primitive_fused(engine &eng, stream &strm,
        const memory::dims &src_shape, const memory::dims &wei_shape,
        const memory::dims &dst_shape, const memory::dims &binary_shape,
        std::vector<uint8_t> &src_data, std::vector<int8_t> &wei_data,
        std::vector<float> &binary_data, float src_scale, int32_t src_zp,
        std::vector<float> &wei_scales, float dst_scale, int32_t dst_zp,
        algorithm binary_alg) {

    int64_t OC = wei_shape[1]; // IOW: {IC, OC, KW}
    memory::dims prim_wei_shape = {wei_shape[1], wei_shape[0], wei_shape[2]};

    // Use format_tag::any to let the primitive choose optimal layouts
    auto src_md_any = memory::desc(
            src_shape, memory::data_type::u8, memory::format_tag::any);
    auto wei_md_any = memory::desc(
            prim_wei_shape, memory::data_type::s8, memory::format_tag::any);
    auto dst_md_any = memory::desc(
            dst_shape, memory::data_type::s8, memory::format_tag::any);

    // Plain (user-side) memory descriptors for data in known layouts
    auto src_md_plain = memory::desc(
            src_shape, memory::data_type::u8, memory::format_tag::ncw);
    auto wei_md_plain = memory::desc(
            prim_wei_shape, memory::data_type::s8, memory::format_tag::bac);
    auto dst_md_plain = memory::desc(
            dst_shape, memory::data_type::s8, memory::format_tag::ncw);
    auto binary_md = memory::desc(
            binary_shape, memory::data_type::f32, memory::format_tag::abc);

    // Configure quantization attributes
    primitive_attr attr;
    attr.set_scales_mask(DNNL_ARG_SRC, 0);
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 0);
    attr.set_scales_mask(DNNL_ARG_DST, 0);
    attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    attr.set_zero_points_mask(DNNL_ARG_DST, 0);

    // Binary post-op
    post_ops pops;
    pops.append_binary(binary_alg, binary_md);
    attr.set_post_ops(pops);

    // Create primitive descriptor with format_tag::any
    auto deconv_pd = deconvolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::deconvolution_direct,
            src_md_any, wei_md_any, dst_md_any,
            /*strides=*/ {1}, /*padding_l=*/ {0}, /*padding_r=*/ {0}, attr);

    // Query optimal memory descriptors chosen by the primitive
    auto src_md_opt = deconv_pd.src_desc();
    auto wei_md_opt = deconv_pd.weights_desc();
    auto dst_md_opt = deconv_pd.dst_desc();

    // Create user (plain) memory and fill with data
    auto src_mem_plain = memory(src_md_plain, eng);
    auto wei_mem_plain = memory(wei_md_plain, eng);
    write_to_dnnl_memory(src_data.data(), src_mem_plain);
    write_to_dnnl_memory(wei_data.data(), wei_mem_plain);

    // Create memory in optimal format and reorder if needed
    auto src_mem = memory(src_md_opt, eng);
    auto wei_mem = memory(wei_md_opt, eng);
    auto dst_mem = memory(dst_md_opt, eng);

    if (src_md_opt != src_md_plain) {
        reorder(src_mem_plain, src_mem).execute(strm, src_mem_plain, src_mem);
    } else {
        src_mem = src_mem_plain;
    }
    if (wei_md_opt != wei_md_plain) {
        reorder(wei_mem_plain, wei_mem).execute(strm, wei_mem_plain, wei_mem);
    } else {
        wei_mem = wei_mem_plain;
    }

    auto binary_mem = memory(binary_md, eng);
    write_to_dnnl_memory(binary_data.data(), binary_mem);

    // Create memory objects for scales and zero points
    auto src_sc_mem
            = memory({{1}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto wei_sc_mem = memory(
            {{OC}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto dst_sc_mem
            = memory({{1}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto src_zp_mem
            = memory({{1}, memory::data_type::s32, memory::format_tag::x}, eng);
    auto dst_zp_mem
            = memory({{1}, memory::data_type::s32, memory::format_tag::x}, eng);

    write_to_dnnl_memory(&src_scale, src_sc_mem);
    write_to_dnnl_memory(wei_scales.data(), wei_sc_mem);
    write_to_dnnl_memory(&dst_scale, dst_sc_mem);
    write_to_dnnl_memory(&src_zp, src_zp_mem);
    write_to_dnnl_memory(&dst_zp, dst_zp_mem);

    // Execute
    auto deconv = deconvolution_forward(deconv_pd);
    deconv.execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_sc_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            binary_mem}});
    strm.wait();

    // Reorder output back to plain layout if needed, then read
    std::vector<int8_t> dst_data(product(dst_shape));
    if (dst_md_opt != dst_md_plain) {
        auto dst_mem_plain = memory(dst_md_plain, eng);
        reorder(dst_mem, dst_mem_plain).execute(strm, dst_mem, dst_mem_plain);
        strm.wait();
        read_from_dnnl_memory(dst_data.data(), dst_mem_plain);
    } else {
        read_from_dnnl_memory(dst_data.data(), dst_mem);
    }
    return dst_data;
}

// =============================================================================
// Method 3: Primitive API - Separate primitives (unfused)
//   Step 1: deconv (u8 src + s8 wei -> f32 dst) with scales/zps
//   Step 2: binary op (f32)
//   Step 3: quantize via reorder (f32 -> s8) with dst scale/zp
//   Uses format_tag::any and reorders for optimal layouts.
// =============================================================================
std::vector<int8_t> run_primitive_unfused(engine &eng, stream &strm,
        const memory::dims &src_shape, const memory::dims &wei_shape,
        const memory::dims &dst_shape, const memory::dims &binary_shape,
        std::vector<uint8_t> &src_data, std::vector<int8_t> &wei_data,
        std::vector<float> &binary_data, float src_scale, int32_t src_zp,
        std::vector<float> &wei_scales, float dst_scale, int32_t dst_zp,
        algorithm binary_alg) {

    int64_t OC = wei_shape[1]; // IOW: {IC, OC, KW}
    memory::dims prim_wei_shape = {wei_shape[1], wei_shape[0], wei_shape[2]};

    // Use format_tag::any to let the primitive choose optimal layouts
    auto src_md_any = memory::desc(
            src_shape, memory::data_type::u8, memory::format_tag::any);
    auto wei_md_any = memory::desc(
            prim_wei_shape, memory::data_type::s8, memory::format_tag::any);
    auto dst_f32_md_any = memory::desc(
            dst_shape, memory::data_type::f32, memory::format_tag::any);

    // Plain memory descriptors
    auto src_md_plain = memory::desc(
            src_shape, memory::data_type::u8, memory::format_tag::ncw);
    auto wei_md_plain = memory::desc(
            prim_wei_shape, memory::data_type::s8, memory::format_tag::bac);
    auto dst_f32_md_plain = memory::desc(
            dst_shape, memory::data_type::f32, memory::format_tag::ncw);
    auto binary_md = memory::desc(
            binary_shape, memory::data_type::f32, memory::format_tag::abc);

    // Step 1: Deconvolution (u8 + s8 -> f32) with src/wei scales and src zp
    primitive_attr deconv_attr;
    deconv_attr.set_scales_mask(DNNL_ARG_SRC, 0);
    deconv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 0);
    deconv_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);

    auto deconv_pd = deconvolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::deconvolution_direct,
            src_md_any, wei_md_any, dst_f32_md_any,
            /*strides=*/ {1}, /*padding_l=*/ {0}, /*padding_r=*/ {0},
            deconv_attr);

    // Query optimal formats
    auto src_md_opt = deconv_pd.src_desc();
    auto wei_md_opt = deconv_pd.weights_desc();
    auto dst_f32_md_opt = deconv_pd.dst_desc();

    // Create user (plain) memory and fill with data
    auto src_mem_plain = memory(src_md_plain, eng);
    auto wei_mem_plain = memory(wei_md_plain, eng);
    write_to_dnnl_memory(src_data.data(), src_mem_plain);
    write_to_dnnl_memory(wei_data.data(), wei_mem_plain);

    // Reorder to optimal format if needed
    auto src_mem = memory(src_md_opt, eng);
    auto wei_mem = memory(wei_md_opt, eng);
    auto deconv_dst_mem = memory(dst_f32_md_opt, eng);

    if (src_md_opt != src_md_plain) {
        reorder(src_mem_plain, src_mem).execute(strm, src_mem_plain, src_mem);
    } else {
        src_mem = src_mem_plain;
    }
    if (wei_md_opt != wei_md_plain) {
        reorder(wei_mem_plain, wei_mem).execute(strm, wei_mem_plain, wei_mem);
    } else {
        wei_mem = wei_mem_plain;
    }

    // Scales and zero points for deconv
    auto src_sc_mem
            = memory({{1}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto wei_sc_mem = memory(
            {{OC}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto src_zp_mem
            = memory({{1}, memory::data_type::s32, memory::format_tag::x}, eng);

    write_to_dnnl_memory(&src_scale, src_sc_mem);
    write_to_dnnl_memory(wei_scales.data(), wei_sc_mem);
    write_to_dnnl_memory(&src_zp, src_zp_mem);

    auto deconv = deconvolution_forward(deconv_pd);
    deconv.execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_DST, deconv_dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_sc_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem}});
    strm.wait();

    // Reorder deconv output to plain f32 ncw for the binary step
    memory deconv_dst_plain = memory(dst_f32_md_plain, eng);
    if (dst_f32_md_opt != dst_f32_md_plain) {
        reorder(deconv_dst_mem, deconv_dst_plain)
                .execute(strm, deconv_dst_mem, deconv_dst_plain);
        strm.wait();
    } else {
        deconv_dst_plain = deconv_dst_mem;
    }

    // Step 2: Binary op (f32 x f32 -> f32)
    auto binary_pd = binary::primitive_desc(
            eng, binary_alg, dst_f32_md_plain, binary_md, dst_f32_md_plain);

    auto binary_src1_mem = memory(binary_md, eng);
    auto binary_dst_mem = memory(dst_f32_md_plain, eng);
    write_to_dnnl_memory(binary_data.data(), binary_src1_mem);

    auto binary_prim = binary(binary_pd);
    binary_prim.execute(strm,
            {{DNNL_ARG_SRC_0, deconv_dst_plain},
                    {DNNL_ARG_SRC_1, binary_src1_mem},
                    {DNNL_ARG_DST, binary_dst_mem}});
    strm.wait();

    // Step 3: Quantize via reorder (f32 -> s8) with dst scale and zero point
    auto dst_s8_md = memory::desc(
            dst_shape, memory::data_type::s8, memory::format_tag::ncw);

    primitive_attr quant_attr;
    quant_attr.set_scales_mask(DNNL_ARG_DST, 0);
    quant_attr.set_zero_points_mask(DNNL_ARG_DST, 0);

    auto quant_pd = reorder::primitive_desc(
            eng, dst_f32_md_plain, eng, dst_s8_md, quant_attr);

    auto dst_s8_mem = memory(dst_s8_md, eng);
    auto dst_sc_mem
            = memory({{1}, memory::data_type::f32, memory::format_tag::x}, eng);
    auto dst_zp_mem
            = memory({{1}, memory::data_type::s32, memory::format_tag::x}, eng);
    write_to_dnnl_memory(&dst_scale, dst_sc_mem);
    write_to_dnnl_memory(&dst_zp, dst_zp_mem);

    auto quant_reorder = reorder(quant_pd);
    quant_reorder.execute(strm,
            {{DNNL_ARG_SRC, binary_dst_mem}, {DNNL_ARG_DST, dst_s8_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem}});
    strm.wait();

    std::vector<int8_t> dst_data(product(dst_shape));
    read_from_dnnl_memory(dst_data.data(), dst_s8_mem);
    return dst_data;
}

// =============================================================================
// Run all 3 methods for a given binary algorithm and compare results
// =============================================================================
void run_and_compare(engine &eng, stream &strm, const memory::dims &src_shape,
        const memory::dims &wei_shape, const memory::dims &dst_shape,
        const memory::dims &binary_shape, std::vector<uint8_t> &src_data,
        std::vector<int8_t> &wei_data, std::vector<float> &binary_data,
        float src_scale, int64_t src_zp, std::vector<float> &wei_scales,
        float dst_scale, int64_t dst_zp, algorithm binary_alg) {

    std::cout << "--- Deconv + Binary " << binary_alg_name(binary_alg) << " ---"
              << std::endl;

    std::cout << "Method 1: Graph API (Dequant->ConvTranspose->"
              << binary_alg_name(binary_alg) << "->Quant)" << std::endl;
    auto result_graph = run_graph_api(eng, strm, src_shape, wei_shape,
            dst_shape, binary_shape, src_data, wei_data, binary_data, src_scale,
            src_zp, wei_scales, dst_scale, dst_zp, binary_alg);

    std::cout << "Method 2: Primitive API with binary post-op (fused int8)"
              << std::endl;
    auto result_fused = run_primitive_fused(eng, strm, src_shape, wei_shape,
            dst_shape, binary_shape, src_data, wei_data, binary_data, src_scale,
            static_cast<int32_t>(src_zp), wei_scales, dst_scale,
            static_cast<int32_t>(dst_zp), binary_alg);

    std::cout << "Method 3: Primitive API with separate primitives (unfused)"
              << std::endl;
    auto result_unfused = run_primitive_unfused(eng, strm, src_shape, wei_shape,
            dst_shape, binary_shape, src_data, wei_data, binary_data, src_scale,
            static_cast<int32_t>(src_zp), wei_scales, dst_scale,
            static_cast<int32_t>(dst_zp), binary_alg);

    // Compare results (allow atol=1 for int8 rounding differences)
    std::cout << std::endl
              << "Comparing results (atol=1 for int8):" << std::endl;

    bool pass_graph_vs_fused = allclose_int8(result_graph, result_fused, 1);
    std::cout << "  Graph API vs Primitive (fused):   "
              << (pass_graph_vs_fused ? "PASS" : "FAIL") << std::endl;

    bool pass_graph_vs_unfused = allclose_int8(result_graph, result_unfused, 1);
    std::cout << "  Graph API vs Primitive (unfused): "
              << (pass_graph_vs_unfused ? "PASS" : "FAIL") << std::endl;

    bool pass_fused_vs_unfused = allclose_int8(result_fused, result_unfused, 1);
    std::cout << "  Primitive (fused) vs (unfused):   "
              << (pass_fused_vs_unfused ? "PASS" : "FAIL") << std::endl;

    // Print first few output values
    std::cout << std::endl << "First 20 output values (s8):" << std::endl;
    std::cout << "  Graph:    ";
    for (int i = 0; i < 20; i++)
        std::cout << static_cast<int>(result_graph[i]) << " ";
    std::cout << std::endl;
    std::cout << "  Fused:    ";
    for (int i = 0; i < 20; i++)
        std::cout << static_cast<int>(result_fused[i]) << " ";
    std::cout << std::endl;
    std::cout << "  Unfused:  ";
    for (int i = 0; i < 20; i++)
        std::cout << static_cast<int>(result_unfused[i]) << " ";
    std::cout << std::endl << std::endl;

    if (!pass_graph_vs_fused || !pass_graph_vs_unfused
            || !pass_fused_vs_unfused) {
        throw std::runtime_error(std::string("Results mismatch for binary ")
                + binary_alg_name(binary_alg) + "!");
    }
}

// =============================================================================
// Main
// =============================================================================
void deconv_binary_example() {
    // Configuration matching test_convtranspose int8 pattern
    // src: {N=1, IC=8, IW=12}, wei: {IC=8, OC=8, KW=3} in IOW format
    // dst: {N=1, OC=8, OW=14}, binary: {1, 1, 1} scalar broadcast
    memory::dims src_shape = {1, 8, 12};
    memory::dims wei_shape = {8, 8, 3}; // IOW: {IC, OC, KW}
    memory::dims dst_shape = {1, 8, 14};
    memory::dims binary_shape = {1, 1, 1};

    int64_t OC = 8;

    // Quantization parameters (matching test_convtranspose.cpp)
    float src_scale = 1.f / 255.f;
    int64_t src_zp = -4;
    std::vector<float> wei_scales(OC, 1.f / 127.f);
    float dst_scale = 1.f;
    int64_t dst_zp = 78;

    // Generate test data with ranges matching test_convtranspose.cpp
    std::default_random_engine gen(7);
    std::uniform_real_distribution<float> u8_dist(0.0f, 25.0f);
    std::uniform_real_distribution<float> s8_dist(-1.0f, 25.0f);

    std::vector<uint8_t> src_data(product(src_shape));
    std::vector<int8_t> wei_data(product(wei_shape));
    std::vector<float> binary_data(product(binary_shape));

    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_dist(gen)); });
    std::generate(wei_data.begin(), wei_data.end(),
            [&]() { return static_cast<int8_t>(s8_dist(gen)); });
    // Binary value matching test_convtranspose.cpp
    std::fill(binary_data.begin(), binary_data.end(), 1.0f);

    std::cout << "INT8 Deconv + Binary Example" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  src:  u8 {1, 8, 12} (NCW), scale=" << src_scale
              << ", zp=" << src_zp << std::endl;
    std::cout << "  wei:  s8 {8, 8, 3}  (IOW), scale=1/127 per-channel"
              << std::endl;
    std::cout << "  dst:  s8 {1, 8, 14} (NCW), scale=" << dst_scale
              << ", zp=" << dst_zp << std::endl;
    std::cout << "  binary: f32 {1, 1, 1} (broadcast), value=" << binary_data[0]
              << std::endl;
    std::cout << std::endl;

    engine eng(engine::kind::cpu, 0);
    stream strm(eng);

    dnnl_set_verbose(1);

    // Case 1: Deconv + Binary Multiply
    run_and_compare(eng, strm, src_shape, wei_shape, dst_shape, binary_shape,
            src_data, wei_data, binary_data, src_scale, src_zp, wei_scales,
            dst_scale, dst_zp, algorithm::binary_mul);

    // Case 2: Deconv + Binary Maximum
    run_and_compare(eng, strm, src_shape, wei_shape, dst_shape, binary_shape,
            src_data, wei_data, binary_data, src_scale, src_zp, wei_scales,
            dst_scale, dst_zp, algorithm::binary_max);

    // Case 3: Deconv + Binary Minimum
    run_and_compare(eng, strm, src_shape, wei_shape, dst_shape, binary_shape,
            src_data, wei_data, binary_data, src_scale, src_zp, wei_scales,
            dst_scale, dst_zp, algorithm::binary_min);

    // Case 4: Deconv + Binary Divide
    run_and_compare(eng, strm, src_shape, wei_shape, dst_shape, binary_shape,
            src_data, wei_data, binary_data, src_scale, src_zp, wei_scales,
            dst_scale, dst_zp, algorithm::binary_div);

    // Case 5: Deconv + Binary Subtract
    run_and_compare(eng, strm, src_shape, wei_shape, dst_shape, binary_shape,
            src_data, wei_data, binary_data, src_scale, src_zp, wei_scales,
            dst_scale, dst_zp, algorithm::binary_sub);

    dnnl_set_verbose(0);
}

int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, deconv_binary_example);
}
