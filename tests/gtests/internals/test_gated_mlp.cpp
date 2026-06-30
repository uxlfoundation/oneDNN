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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include "test_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>

#include <cassert>
#include <random>

#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2

#include "common/gated_mlp_iface.hpp"

// uncomment to dump cpu memory buffers
//#define ENABLE_PRINT_MEM

// uncomment to disable everything except Up
//#define ENABLE_UP_ONLY

static bool verbose = false; // enable for debug
static const int min_runs = 4;

namespace dnnl {
namespace impl {

using tag = memory::format_tag;
using mdt = memory::data_type;

/// Gated MLP (gmlp) internal primitive.
struct gmlp_t : public dnnl::primitive {
    /// Primitive descriptor for a gmlp primitive.
    struct pd_t : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        pd_t() = default;

        /// Constructs a primitive descriptor for a gmlp primitive from a C
        /// API primitive descriptor that must have a matching kind.
        ///
        /// @param pd C API primitive descriptor for a gmlp primitive.
        pd_t(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::undef) {}

        pd_t(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &W_gate_desc, const memory::desc &W_up_desc,
                const memory::desc &W_down_desc,
                const memory::desc &output_desc, const alg_kind_t &activation,
                const primitive_attr &attr = default_attr()) {
            dnnl_primitive_desc_t pd = nullptr;
            dnnl_status_t status = dnnl_gated_mlp_primitive_desc_create(&pd,
                    aengine.get(), src_desc.get(), W_gate_desc.get(),
                    W_up_desc.get(), W_down_desc.get(), output_desc.get(),
                    activation, attr.get());

            dnnl::error::wrap_c_api(status,
                    "could not create a primitive descriptor for a gmlp "
                    "primitive");
            reset(pd);
        }
    };

    /// Default constructor. Produces an empty object.
    gmlp_t() = default;

    /// Constructs a gmlp primitive.
    /// @param pd Primitive descriptor for a gmlp primitive.
    gmlp_t(const pd_t &pd) : primitive(pd) {}
};

struct mlp_dims_t {
    dim_t mb;
    dim_t ic;
    dim_t oc;

    int src_group_size;
    int gateup_group_size;
    int down_group_size;

    quantize_type qtype;
    dnnl_alg_kind_t activation;

    memory::data_type dst_dt;

    memory::data_type src_dt;
    memory::data_type src_s_dt;
    memory::data_type src_zp_dt;

    memory::data_type wgu_wt;
    memory::data_type wgu_s_dt;
    memory::data_type wgu_zp_dt;

    memory::data_type wd_wt;
    memory::data_type wd_s_dt;
    memory::data_type wd_zp_dt;
};

struct gmlp_tensors_t {
    memory m_x, m_w_gate, m_w_up, m_w_down;
    memory m_x_quant, m_w_gate_quant, m_w_up_quant, m_w_down_quant;
    memory m_x_scales, m_w_gate_scales, m_w_up_scales, m_w_down_scales;
    memory m_x_zp, m_w_gate_zp, m_w_up_zp, m_w_down_zp;
    memory m_fc_gate, m_fc_up, m_fc_down;
    memory m_fc_retn_t;

    primitive_attr gateup_attr_quantized, down_attr_quantized;
    memory::dims src_groups, wgu_groups, wd_groups;
};

std::ostream &operator<<(std::ostream &ss, const dnnl_alg_kind_t &act) {
    switch (act) {
        case dnnl_alg_kind_t::dnnl_eltwise_gelu_erf:
            ss << "_activation_gelu_erf";
            break;
        case dnnl_alg_kind_t::dnnl_eltwise_gelu_tanh:
            ss << "_activation_gelu_tanh";
            break;
        case dnnl_alg_kind_t::dnnl_eltwise_swish:
            ss << "_activation_swish";
            break;
        default: ss << "_activation_unknown"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const mlp_dims_t &p) {
    const bool has_quant = p.qtype != quantize_type::no_quantization;

    ss << "mb_" << p.mb;
    ss << "_ic_" << p.ic;
    ss << "_oc_" << p.oc;

    ss << ((has_quant) ? "_quant_" : "_noquant_");
    if (has_quant) {
        ss << "_qtype_" << p.qtype;
        ss << "_src_gs_" << p.src_group_size;
        ss << "_gu_gs_" << p.gateup_group_size;
        ss << "_d_gs_" << p.down_group_size;
    }

    ss << "_dst_dt_" << dnnl_dt2str(memory::convert_to_c(p.dst_dt));

    ss << "_src_dt_" << dnnl_dt2str(memory::convert_to_c(p.src_dt));
    if (has_quant) {
        ss << "_src_sdt_" << dnnl_dt2str(memory::convert_to_c(p.src_s_dt));
        ss << "_src_zpdt_" << dnnl_dt2str(memory::convert_to_c(p.src_zp_dt));
    }

    ss << "_wgu_wt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_wt));
    if (has_quant) {
        ss << "_wgu_sdt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_s_dt));
        ss << "_wgu_zpdt_" << dnnl_dt2str(memory::convert_to_c(p.wgu_zp_dt));
    }

    ss << "_wd_wt_" << dnnl_dt2str(memory::convert_to_c(p.wd_wt));
    if (has_quant) {
        ss << "_wd_sdt_" << dnnl_dt2str(memory::convert_to_c(p.wd_s_dt));
        ss << "_wd_zpdt_" << dnnl_dt2str(memory::convert_to_c(p.wd_zp_dt));
    }

    ss << p.activation;
    return ss;
}

std::string PrintToString(const ::testing::TestParamInfo<mlp_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

gmlp_tensors_t get_descriptors(engine &eng, stream &strm, mlp_dims_t p) {
    gmlp_tensors_t out;

    // Prepare input and output shapes to construct the swiglu graph.
    const memory::dims O_proj_sz = {p.mb, p.ic};
    const memory::dims W_gate_sz = {p.ic, p.oc};
    const memory::dims W_up_sz = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic};
    const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};

    const memory::dims qnt_src_sz = [&]() {
        switch (p.qtype) {
            case quantize_type::no_quantization: return memory::dims {1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {O_proj_sz[0],
                        O_proj_sz[1] / std::max(p.src_group_size, 1)};
            case quantize_type::per_token:
                return memory::dims {1, O_proj_sz[1]};
            case quantize_type::per_tensor: return memory::dims {1, 1};
            default: return memory::dims {0, 0};
        }
    }();
    const memory::dims qnt_gu_sz = [&]() {
        switch (p.qtype) {
            case quantize_type::no_quantization: return memory::dims {1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        W_gate_sz[0] / std::max(p.gateup_group_size, 1),
                        W_gate_sz[1]};
            case quantize_type::per_token:
                return memory::dims {W_gate_sz[0], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1};
            default: return memory::dims {0, 0};
        }
    }();
    const memory::dims qnt_d_sz = [&]() {
        switch (p.qtype) {
            case quantize_type::no_quantization: return memory::dims {1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        W_down_sz[0] / std::max(p.down_group_size, 1),
                        W_down_sz[1]};
            case quantize_type::per_token:
                return memory::dims {W_down_sz[0], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1};
            default: return memory::dims {0, 0};
        }
    }();
    auto maybe_memory = [](const memory::desc &md, const engine &aengine) {
        if (md && (md.get_data_type() != mdt::undef))
            return memory(md, aengine);
        return memory();
    };
    auto maybe_product = [](const memory &mem) {
        if (mem) return product(mem.get_desc().get_padded_dims());
        return dim_t(0);
    };

    auto FC_gate_md = memory::desc(FC_gate_sz, mdt::f32, tag::ab);
    auto FC_up_md = memory::desc(FC_up_sz, mdt::f32, tag::ab);
    auto FC_down_md = memory::desc(FC_down_sz, p.dst_dt, tag::ab);
    auto FC_retn_md_t = memory::desc(FC_down_sz, p.dst_dt, tag::ab);

    // clang-format off
    auto x_md      = memory::desc(O_proj_sz, mdt::f32, tag::ab);
    auto w_gate_md = memory::desc(W_gate_sz, mdt::f32, tag::ba);
    auto w_up_md   = memory::desc(W_up_sz,   mdt::f32, tag::ba);
    auto w_down_md = memory::desc(W_down_sz, mdt::f32, tag::ba);

    auto x_qnt_md      = memory::desc(O_proj_sz, p.src_dt, tag::ab);
    auto w_gate_qnt_md = memory::desc(W_gate_sz, p.wgu_wt, tag::ba);
    auto w_up_qnt_md   = memory::desc(W_up_sz,   p.wgu_wt, tag::ba);
    auto w_down_qnt_md = memory::desc(W_down_sz,  p.wd_wt, tag::ba);

    auto x_scales_md      = memory::desc(qnt_src_sz, p.src_s_dt, tag::ab, true);
    auto w_gate_scales_md = memory::desc(qnt_gu_sz,  p.wgu_s_dt, tag::ab, true);
    auto w_up_scales_md   = memory::desc(qnt_gu_sz,  p.wgu_s_dt, tag::ab, true);
    auto w_down_scales_md = memory::desc(qnt_d_sz,    p.wd_s_dt, tag::ab, true);

    auto x_zp_md      = memory::desc(qnt_src_sz, p.src_zp_dt, tag::ab, true);
    auto w_gate_zp_md = memory::desc(qnt_gu_sz,  p.wgu_zp_dt, tag::ab, true);
    auto w_up_zp_md   = memory::desc(qnt_gu_sz,  p.wgu_zp_dt, tag::ab, true);
    auto w_down_zp_md = memory::desc(qnt_d_sz,    p.wd_zp_dt, tag::ab, true);
    // clang-format on

    // Create memory objects
    out.m_x = memory(x_md, eng);
    out.m_w_gate = memory(w_gate_md, eng);
    out.m_w_up = memory(w_up_md, eng);
    out.m_w_down = memory(w_down_md, eng);

    out.m_x_quant = memory(x_qnt_md, eng);
    out.m_w_gate_quant = memory(w_gate_qnt_md, eng);
    out.m_w_up_quant = memory(w_up_qnt_md, eng);
    out.m_w_down_quant = memory(w_down_qnt_md, eng);

    out.m_x_scales = maybe_memory(x_scales_md, eng);
    out.m_w_gate_scales = maybe_memory(w_gate_scales_md, eng);
    out.m_w_up_scales = maybe_memory(w_up_scales_md, eng);
    out.m_w_down_scales = maybe_memory(w_down_scales_md, eng);

    out.m_x_zp = maybe_memory(x_zp_md, eng);
    out.m_w_gate_zp = maybe_memory(w_gate_zp_md, eng);
    out.m_w_up_zp = maybe_memory(w_up_zp_md, eng);
    out.m_w_down_zp = maybe_memory(w_down_zp_md, eng);

    out.m_fc_gate = memory(FC_gate_md, eng);
    out.m_fc_up = memory(FC_up_md, eng);
    out.m_fc_down = memory(FC_down_md, eng);

    out.m_fc_retn_t = memory(FC_retn_md_t, eng);

    // Allocate user data.
    std::vector<float> x_data(product(O_proj_sz));
    std::vector<float> w_gate_data(product(W_gate_sz));
    std::vector<float> w_up_data(product(W_up_sz));
    std::vector<float> w_down_data(product(W_down_sz));

    std::vector<float> src_quantized_data(product(O_proj_sz), 1.f);
    std::vector<float> w_gate_quantized_data(product(W_gate_sz), 1.f);
    std::vector<float> w_up_quantized_data(product(W_up_sz), 1.f);
    std::vector<float> w_down_quantized_data(product(W_down_sz), 1.f);

    std::vector<float> src_scales_data(maybe_product(out.m_x_scales), 1.f);
    std::vector<float> w_gate_scales_data(
            maybe_product(out.m_w_gate_scales), 1.f);
    std::vector<float> w_up_scales_data(maybe_product(out.m_w_up_scales), 1.f);
    std::vector<float> w_down_scales_data(
            maybe_product(out.m_w_down_scales), 1.f);

    std::vector<int> src_zp_data(maybe_product(out.m_x_zp), 0);
    std::vector<int> w_gate_zp_data(maybe_product(out.m_w_gate_zp), 0);
    std::vector<int> w_up_zp_data(maybe_product(out.m_w_up_zp), 0);
    std::vector<int> w_down_zp_data(maybe_product(out.m_w_down_zp), 0);

    out.src_groups = {};
    out.wgu_groups = {};
    out.wd_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups: {
            out.src_groups = {1, p.src_group_size};
            out.wgu_groups = {p.gateup_group_size, 1};
            out.wd_groups = {p.down_group_size, 1};
            break;
        }
        case quantize_type::per_token: {
            // TODO: add
            break;
        }
        case quantize_type::per_tensor: {
            // TODO: add
            break;
        }
        default: break;
    }

    int src_group_size = p.src_group_size;
    int wgu_group_size = p.gateup_group_size;
    int wd_group_size = p.down_group_size;

    //if (p.qtype == quantize_type::per_tensor) {
    //    wgu_group_size = W_gate_sz[0] * W_gate_sz[1];
    //    wd_group_size = W_down_sz[0] * W_down_sz[1];
    //}

    if (p.qtype == quantize_type::no_quantization) {
        if (verbose) printf("no quant init\n");
        fill_random(x_data, x_md, -.25f, .25f);
        fill_random(w_gate_data, w_gate_md, -1.f, 1.f);
        fill_random(w_up_data, w_up_md, -1.f, 1.f);
        fill_random(w_down_data, w_down_md, -1.f, 1.f);
    } else {
        fill_random_quantized(src_quantized_data, x_qnt_md,
                (p.src_dt == mdt::u4 || p.src_dt == mdt::u8));
        fill_random_quantized(w_gate_quantized_data, w_gate_qnt_md,
                (p.wgu_wt == mdt::u4 || p.wgu_wt == mdt::u8));
        fill_random_quantized(w_up_quantized_data, w_up_qnt_md,
                (p.wgu_wt == mdt::u4 || p.wgu_wt == mdt::u8));
        fill_random_quantized(w_down_quantized_data, w_down_qnt_md,
                (p.wd_wt == mdt::u4 || p.wd_wt == mdt::u8));

        if (x_scales_md) fill_random_scales(src_scales_data, x_scales_md);
        if (w_gate_scales_md)
            fill_random_scales(w_gate_scales_data, w_gate_scales_md);
        if (w_up_scales_md)
            fill_random_scales(w_up_scales_data, w_up_scales_md);
        if (w_down_scales_md)
            fill_random_scales(w_down_scales_data, w_down_scales_md);

        bool src_zp_unsigned
                = (p.src_zp_dt == mdt::u4 || p.src_zp_dt == mdt::u8);
        if (verbose) {
            if (src_zp_unsigned)
                printf("unsigned src quant init\n");
            else
                printf("signed src quant init\n");
        }
        if (x_zp_md)
            fill_random_quantized(src_zp_data, x_zp_md, src_zp_unsigned);

        if (!x_scales_md && x_zp_md) {
            x_scales_md = x_zp_md;
            src_scales_data = std::vector<float>(src_zp_data.size(), 1.f);
        }
        if (x_scales_md && !x_zp_md) {
            src_zp_data = std::vector<int>(src_scales_data.size(), 0);
        }
        if (x_scales_md) {
            x_data = dequantize(src_quantized_data, x_md, x_scales_md,
                    src_zp_data, src_scales_data, src_group_size, p.qtype,
                    out.src_groups, 1);
        } else {
            x_data = src_quantized_data;
        }

        bool wgu_zp_unsigned
                = (p.wgu_zp_dt == mdt::u4 || p.wgu_zp_dt == mdt::u8);
        if (verbose) {
            if (wgu_zp_unsigned)
                printf("unsigned gateup quant init\n");
            else
                printf("signed gateup quant init\n");
        }
        if (w_gate_zp_md)
            fill_random_quantized(
                    w_gate_zp_data, w_gate_zp_md, wgu_zp_unsigned);
        if (w_up_zp_md)
            fill_random_quantized(w_up_zp_data, w_up_zp_md, wgu_zp_unsigned);

        if (!w_gate_scales_md && w_gate_zp_md) {
            w_gate_scales_md = w_gate_zp_md;
            w_gate_scales_data = std::vector<float>(w_gate_zp_data.size(), 1.f);
        }
        if (w_gate_scales_md && !w_gate_zp_md) {
            w_gate_zp_data = std::vector<int>(w_gate_scales_data.size(), 0);
        }
        if (w_gate_scales_md) {
            w_gate_data = dequantize(w_gate_quantized_data, w_gate_md,
                    w_gate_scales_md, w_gate_zp_data, w_gate_scales_data,
                    wgu_group_size, p.qtype, out.wgu_groups, 0);
        } else {
            w_gate_data = w_gate_quantized_data;
        }

        if (!w_up_scales_md && w_up_zp_md) {
            w_up_scales_md = w_up_zp_md;
            w_up_scales_data = std::vector<float>(w_up_zp_data.size(), 1.f);
        }
        if (w_up_scales_md && !w_up_zp_md) {
            w_up_zp_data = std::vector<int>(w_up_scales_data.size(), 0);
        }
        if (w_up_scales_md) {
            w_up_data = dequantize(w_up_quantized_data, w_up_md, w_up_scales_md,
                    w_up_zp_data, w_up_scales_data, wgu_group_size, p.qtype,
                    out.wgu_groups, 0);
        } else {
            w_up_data = w_up_quantized_data;
        }

        bool wd_zp_unsigned = (p.wd_zp_dt == mdt::u4 || p.wd_zp_dt == mdt::u8);
        if (verbose) {
            if (wd_zp_unsigned)
                printf("unsigned down quant init\n");
            else
                printf("signed down quant init\n");
        }
        if (w_down_zp_md)
            fill_random_quantized(w_down_zp_data, w_down_zp_md, wd_zp_unsigned);

        if (!w_down_scales_md && w_down_zp_md) {
            w_down_scales_md = w_down_zp_md;
            w_down_scales_data = std::vector<float>(w_down_zp_data.size(), 1.f);
        }
        if (w_down_scales_md && !w_down_zp_md) {
            w_down_zp_data = std::vector<int>(w_down_scales_data.size(), 0);
        }
        if (w_down_scales_md) {
            w_down_data = dequantize(w_down_quantized_data, w_down_md,
                    w_down_scales_md, w_down_zp_data, w_down_scales_data,
                    wd_group_size, p.qtype, out.wd_groups, 0);
        } else {
            w_down_data = w_down_quantized_data;
        }
    }

    // Write data to tensor object's handle.
    write_to_dnnl_memory(x_data.data(), out.m_x, eng, strm);
    write_to_dnnl_memory(w_gate_data.data(), out.m_w_gate, eng, strm);
    write_to_dnnl_memory(w_up_data.data(), out.m_w_up, eng, strm);
    write_to_dnnl_memory(w_down_data.data(), out.m_w_down, eng, strm);

    if (p.qtype == quantize_type::no_quantization) {
        write_to_dnnl_memory(x_data.data(), out.m_x_quant, eng, strm);
        write_to_dnnl_memory(w_gate_data.data(), out.m_w_gate_quant, eng, strm);
        write_to_dnnl_memory(w_up_data.data(), out.m_w_up_quant, eng, strm);
        write_to_dnnl_memory(w_down_data.data(), out.m_w_down_quant, eng, strm);
    } else {
        write_to_dnnl_memory(
                src_quantized_data.data(), out.m_x_quant, eng, strm);
        write_to_dnnl_memory(
                w_gate_quantized_data.data(), out.m_w_gate_quant, eng, strm);
        write_to_dnnl_memory(
                w_up_quantized_data.data(), out.m_w_up_quant, eng, strm);
        write_to_dnnl_memory(
                w_down_quantized_data.data(), out.m_w_down_quant, eng, strm);

        if (out.m_x_zp)
            write_to_dnnl_memory(src_zp_data.data(), out.m_x_zp, eng, strm);
        if (out.m_w_gate_zp)
            write_to_dnnl_memory(
                    w_gate_zp_data.data(), out.m_w_gate_zp, eng, strm);
        if (out.m_w_up_zp)
            write_to_dnnl_memory(w_up_zp_data.data(), out.m_w_up_zp, eng, strm);
        if (out.m_w_down_zp)
            write_to_dnnl_memory(
                    w_down_zp_data.data(), out.m_w_down_zp, eng, strm);

        if (out.m_x_scales)
            write_to_dnnl_memory(
                    src_scales_data.data(), out.m_x_scales, eng, strm);
        if (out.m_w_gate_scales)
            write_to_dnnl_memory(
                    w_gate_scales_data.data(), out.m_w_gate_scales, eng, strm);
        if (out.m_w_up_scales)
            write_to_dnnl_memory(
                    w_up_scales_data.data(), out.m_w_up_scales, eng, strm);
        if (out.m_w_down_scales)
            write_to_dnnl_memory(
                    w_down_scales_data.data(), out.m_w_down_scales, eng, strm);
    }
    return out;
}

void bench_gated_mlp_primitives(std::vector<float> &res, double &avg_time,
        gmlp_tensors_t &t, engine &eng, stream &strm, const mlp_dims_t &p,
        double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // extract memory objects
    auto m_O_proj = t.m_x;
    auto m_W_gate = t.m_w_gate;
    auto m_W_up = t.m_w_up;
    auto m_W_down = t.m_w_down;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    // extract memory descriptors
    auto O_proj_md = t.m_x.get_desc();
    auto W_gate_md = t.m_w_gate.get_desc();
    auto W_up_md = t.m_w_up.get_desc();
    auto W_down_md = t.m_w_down.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    auto m_FC_retn_t = t.m_fc_retn_t;

    auto gen_default_attr = [](quantize_type qtype) {
        primitive_attr attr;
        switch (qtype) {
            default: break;
            case quantize_type::per_token_with_groups:
            case quantize_type::per_tensor:
                attr.set_fpmath_mode(
                        static_cast<enum fpmath_mode>(fpmath_mode::any), true);
                break;
        }
        return attr;
    };

    // fc_up
    auto bmm0_pd = matmul::primitive_desc(
            eng, O_proj_md, W_up_md, FC_up_md, gen_default_attr(p.qtype));
    auto bmm0_prim = matmul(bmm0_pd);

    // fc_gate -> swish -> mul
    auto bmm1_attr = gen_default_attr(p.qtype);
    post_ops bmm1_po;
    if (p.activation == dnnl_eltwise_swish) {
        bmm1_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    } else if (p.activation == dnnl_eltwise_gelu_erf) {
        bmm1_po.append_eltwise(algorithm::eltwise_gelu_erf, 0.f, 0.f);
    } else if (p.activation == dnnl_eltwise_gelu_tanh) {
        bmm1_po.append_eltwise(algorithm::eltwise_gelu_tanh, 0.f, 0.f);
    }
    bmm1_po.append_binary(algorithm::binary_mul, m_FC_up.get_desc());
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, O_proj_md, W_gate_md, FC_gate_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    auto bmm2_pd = matmul::primitive_desc(
            eng, FC_gate_md, W_down_md, FC_down_md, gen_default_attr(p.qtype));
    auto bmm2_prim = matmul(bmm2_pd);

    const auto loop = [&](bool print = false) {
#ifdef ENABLE_PRINT_MEM
#define PRINT_MEM(mem) \
    if (print) { print_mem(mem, #mem " "); }
#else
#define PRINT_MEM(mem)
#endif
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                        {DNNL_ARG_DST, m_FC_up}});
#ifndef ENABLE_UP_ONLY
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_FC_up}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_FC_gate}, {DNNL_ARG_WEIGHTS, m_W_down},
                        {DNNL_ARG_DST, m_FC_down}});
#endif
        PRINT_MEM(m_O_proj)
        PRINT_MEM(m_W_up)
#ifndef ENABLE_UP_ONLY
        PRINT_MEM(m_W_gate)
        PRINT_MEM(m_W_down)
#endif
        PRINT_MEM(m_FC_up)
#ifndef ENABLE_UP_ONLY
        PRINT_MEM(m_FC_gate)
        PRINT_MEM(m_FC_down)
#endif
#undef PRINT_MEM
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop(true);

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = (duration.count() - dur_first.count()) / runs;
    if (verbose) {
        std::cout << "primitive runs: " << runs + 1 << "; ";
        std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    }

#ifndef ENABLE_UP_ONLY
    res.resize(product(m_FC_down.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_down, false);
#else
    res.resize(product(m_FC_up.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_up, false);
#endif
}

void bench_gated_mlp_internal(std::vector<float> &res, double &avg_time,
        gmlp_tensors_t &t, engine &eng, stream strm, const mlp_dims_t &p,
        double time_limit = 0.) {
    if (verbose) printf("eng?\n");
    const bool quick_test = (time_limit == 0.);

    auto FC_down_md = t.m_fc_down.get_desc();

#ifdef ENABLE_UP_ONLY
    const memory::dims FC_retn_sz_t = {p.mb, p.oc};
#else
    const memory::dims FC_retn_sz_t = {p.mb, p.ic};
#endif
    auto FC_retn_md_t
            = memory::desc(FC_retn_sz_t, FC_down_md.get_data_type(), tag::ab);
    auto m_FC_gate_t = memory(FC_retn_md_t, eng);

    primitive_attr attr;
    int mask = 0;
    switch (p.qtype) {
        case quantize_type::per_token_with_groups: mask = (1 << 0) + (1 << 1);
        case quantize_type::per_tensor: {
            attr.set_fpmath_mode(
                    static_cast<enum fpmath_mode>(fpmath_mode::any), true);
            // src scale+zp
            if (t.m_x_scales)
                attr.set_scales(DNNL_ARG_SRC, mask, t.src_groups,
                        t.m_x_scales.get_desc().get_data_type());
            if (t.m_x_zp)
                attr.set_zero_points(DNNL_ARG_SRC, mask, t.src_groups,
                        t.m_x_zp.get_desc().get_data_type());

            // wts_gate scale+zp
            if (t.m_w_gate_scales)
                attr.set_scales(DNNL_ARG_WEIGHTS_GATE, mask, t.wgu_groups,
                        t.m_w_gate_scales.get_desc().get_data_type());
            if (t.m_w_gate_zp)
                attr.set_zero_points(DNNL_ARG_WEIGHTS_GATE, mask, t.wgu_groups,
                        t.m_w_gate_zp.get_desc().get_data_type());

            // wts_up scale+zp
            if (t.m_w_up_scales)
                attr.set_scales(DNNL_ARG_WEIGHTS_UP, mask, t.wgu_groups,
                        t.m_w_up_scales.get_desc().get_data_type());
            if (t.m_w_up_zp)
                attr.set_zero_points(DNNL_ARG_WEIGHTS_UP, mask, t.wgu_groups,
                        t.m_w_up_zp.get_desc().get_data_type());

            // wts_down scale+zp
            if (t.m_w_down_scales)
                attr.set_scales(DNNL_ARG_WEIGHTS_DOWN, mask, t.wd_groups,
                        t.m_w_down_scales.get_desc().get_data_type());
            if (t.m_w_down_zp)
                attr.set_zero_points(DNNL_ARG_WEIGHTS_DOWN, mask, t.wd_groups,
                        t.m_w_down_zp.get_desc().get_data_type());
            break;
        }
        default: break;
    }

    auto gmlp_pd = [&]() {
        dnnl_alg_kind_t activation;
        activation = p.activation;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_swish;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_gelu_erf;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_gelu_tanh;
        //activation = dnnl_alg_kind_t::dnnl_eltwise_exp; // should fail
        return gmlp_t::pd_t(eng, t.m_x_quant.get_desc(),
                t.m_w_gate_quant.get_desc(), t.m_w_up_quant.get_desc(),
                t.m_w_down_quant.get_desc(), FC_retn_md_t, activation, attr);
    }();

    auto prim_fused_internal = gmlp_t(gmlp_pd);

    const auto loop = [&](bool print = false) {
#ifdef ENABLE_PRINT_MEM
#define PRINT_MEM(mem) \
    if (print) { print_mem(mem, #mem " "); }
#else
#define PRINT_MEM(mem)
#endif
        if (p.qtype == quantize_type::no_quantization) {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC, t.m_x_quant},
                            {DNNL_ARG_WEIGHTS_GATE, t.m_w_gate_quant},
                            {DNNL_ARG_WEIGHTS_UP, t.m_w_up_quant},
                            {DNNL_ARG_WEIGHTS_DOWN, t.m_w_down_quant},
                            {DNNL_ARG_DST, m_FC_gate_t}});
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(t.m_x_quant)
            PRINT_MEM(t.m_w_up_quant)
            PRINT_MEM(t.m_w_gate_quant)
            PRINT_MEM(t.m_w_down_quant)
#endif
            PRINT_MEM(m_FC_gate_t)
        } else {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC, t.m_x_quant},
                            {DNNL_ARG_WEIGHTS_GATE, t.m_w_gate_quant},
                            {DNNL_ARG_WEIGHTS_UP, t.m_w_up_quant},
                            {DNNL_ARG_WEIGHTS_DOWN, t.m_w_down_quant},
                            {DNNL_ARG_DST, m_FC_gate_t},
                            {DNNL_ARG_SRC | DNNL_ARG_ATTR_SCALES, t.m_x_scales},
                            {DNNL_ARG_SRC | DNNL_ARG_ATTR_ZERO_POINTS,
                                    t.m_x_zp},
                            {DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_SCALES,
                                    t.m_w_gate_scales},
                            {DNNL_ARG_WEIGHTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS,
                                    t.m_w_gate_zp},
                            {DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_SCALES,
                                    t.m_w_up_scales},
                            {DNNL_ARG_WEIGHTS_UP | DNNL_ARG_ATTR_ZERO_POINTS,
                                    t.m_w_up_zp},
                            {DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_SCALES,
                                    t.m_w_down_scales},
                            {DNNL_ARG_WEIGHTS_DOWN | DNNL_ARG_ATTR_ZERO_POINTS,
                                    t.m_w_down_zp}});
#ifndef ENABLE_UP_ONLY
            PRINT_MEM(t.m_x_quant)
            PRINT_MEM(t.m_w_up_quant)
            PRINT_MEM(t.m_w_gate_quant)
            PRINT_MEM(t.m_w_down_quant)
            PRINT_MEM(t.m_x_scales)
            PRINT_MEM(t.m_x_zp)
            PRINT_MEM(t.m_w_up_scales)
            PRINT_MEM(t.m_w_up_zp)
            PRINT_MEM(t.m_w_gate_scales)
            PRINT_MEM(t.m_w_gate_zp)
            PRINT_MEM(t.m_w_down_scales)
            PRINT_MEM(t.m_w_down_zp)
#else
            PRINT_MEM(t.m_w_up_quant)
            PRINT_MEM(t.m_w_up_scales)
            PRINT_MEM(t.m_w_up_zp)
#endif
            PRINT_MEM(m_FC_gate_t)
        }
#undef PRINT_MEM
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop(true);

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
        //print_mem(m_W_gate, "-ilolloopafnternal");
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = (duration.count() - dur_first.count()) / runs;
    if (verbose) {
        std::cout << "internal gmlp primitive runs: " << runs + 1 << "; ";
        std::cout << "avg_time: " << avg_time << " ms" << std::endl;
    }

    res.resize(product(m_FC_gate_t.get_desc().get_dims()));
    move_data(eng, strm, res, m_FC_gate_t, false);
}

enum class api_kind { primitive, graph, internal_hack };

template <typename T>
void bench(std::vector<T> &res, double &avg_time, gmlp_tensors_t &t,
        api_kind api, engine &eng, stream &strm, const mlp_dims_t &p,
        double time_limit = 0.) {

    try {
        if (api == api_kind::primitive) {
            bench_gated_mlp_primitives(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        } else if (api == api_kind::graph) {
            // TODO: add graph
        } else {
            bench_gated_mlp_internal(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        }
    } catch (error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported mlp" << std::endl;
        } else
            throw;
    }
}

class mlp_test_t : public ::testing::TestWithParam<mlp_dims_t> {
public:
    void SetUp() override {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "GMLP primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "GMLP primitive tests do not support HIP";
#endif
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "GMLP tests require gpus.");
        p = GetParam();
        eng = engine(engine::kind::gpu, 0);
        strm = stream(eng);
        t = get_descriptors(eng, strm, p);
    }

protected:
    mlp_dims_t p;
    engine eng;
    stream strm;
    gmlp_tensors_t t;
};

TEST_P(mlp_test_t, compare) {
    auto tensors = t;
    auto params = p;

    std::vector<float> resph, resih;
    double avg_time_int, avg_time_prim;

    if (verbose) printf("PRIMITIVE\n");
    bench(resph, avg_time_prim, tensors, api_kind::primitive, eng, strm, params,
            2000.0 /*ms*/);

    if (verbose) printf("INTERNAL\n");
    bench(resih, avg_time_int, tensors, api_kind::internal_hack, eng, strm,
            params, 2000.0 /*ms*/);

    if (resih.empty()) {
        if (verbose)
            printf("[WARNING] Empty output: internal kernel failure!\n");
        EXPECT_TRUE(false);
    }
    int n_mismatches = 0, n_matches = 0;
    if (verbose) printf("resih.size() %zu\n", resih.size());
    float max_diff = 0.0f, max_val, max_gold;
    for (int i = 0; i < int(resih.size()); ++i) {
        float abs_diff = std::abs(resih[i] - resph[i]);
        float rel_diff = std::abs((resih[i] - resph[i]) / resih[i]);
        if (abs_diff > 1e-4 && rel_diff > 5e-2) {

            if (isfinite(rel_diff) && (abs_diff) > max_diff) {
                max_diff = abs_diff;
                max_val = resih[i];
                max_gold = resph[i];
            }

            n_mismatches++;
            if (verbose && (n_mismatches < 10))
                printf("mismatch @ %d, %f != %f\n", i, float(resih[i]),
                        float(resph[i])); //TODO: improve
        } else {
            if (std::abs(float16_t(resih[i])) > 5e-3) {
                n_matches++;
                if (verbose && (n_matches < 10))
                    printf("vs @ %d, %f == %f\n", i, float(resih[i]),
                            float(resph[i])); //TODO: improve
            }
        }
    }
    int total_size = int(resph.size());
    int threshold = total_size * 0.0006;

    if (verbose) {
        printf("total mismatches: %d, allowed: %d\n", n_mismatches, threshold);
        printf("avg time internal vs primitive: %f vs %f, w/speedup of %f\n",
                avg_time_int, avg_time_prim, avg_time_prim / avg_time_int);
        if (n_mismatches > 0) {
            std::cout << "max diff: " << max_diff << ":  " << max_val
                      << " != " << max_gold << std::endl;
        }
    }
    ASSERT_LE(n_mismatches, threshold) << "out of: " << total_size;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(VEC, mlp_test_t, ::testing::Values(

mlp_dims_t{1024, 3584, 18944, 0, 128, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::f16,   mdt::undef, mdt::undef,
mdt::u4,    mdt::f16,   mdt::u8,
mdt::u4,    mdt::undef, mdt::undef}
,
mlp_dims_t{1024, 4096, 14336, 0, 4096, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::f16,   mdt::undef, mdt::undef,
mdt::u8,    mdt::f16,   mdt::u8,
mdt::u8,    mdt::undef, mdt::undef}
,

mlp_dims_t{1032, 2560, 6912, 2560, 64, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::s8,    mdt::f16,   mdt::undef,
mdt::s4,    mdt::f16,   mdt::undef,
mdt::s4,    mdt::undef, mdt::undef}
,

mlp_dims_t{1086, 1024, 4096, 1024, 128, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::s8,    mdt::f16,   mdt::undef,
mdt::u4,    mdt::f16,   mdt::u8,
mdt::u4,    mdt::undef, mdt::undef}
,
mlp_dims_t{1024, 2048, 11008, 64, 64, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::s8,    mdt::f16,   mdt::undef,
mdt::u4,    mdt::f16,   mdt::u8,
mdt::u4,    mdt::undef, mdt::undef}
,

mlp_dims_t{1086, 1024, 4096, 1024, 1024, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::s8,    mdt::f16,   mdt::undef,
mdt::u8,    mdt::f16,   mdt::u8,
mdt::u8,    mdt::undef, mdt::undef}
,
mlp_dims_t{1024, 2048, 11008, 128, 2048, 0,
quantize_type::per_token_with_groups, dnnl_eltwise_swish, mdt::f16,
mdt::s8,    mdt::f16,   mdt::undef,
mdt::u8,    mdt::f16,   mdt::u8,
mdt::u8,    mdt::undef, mdt::undef}

), &PrintToString);
// clang-format on

} // namespace impl
} // namespace dnnl
