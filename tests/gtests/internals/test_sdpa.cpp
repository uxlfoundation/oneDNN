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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include "sdpa_internal.hpp"
#include "test_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#include <memory>
#include <random>

using mdt = memory::data_type;

enum class mask_type { no_mask, oneD, twoD, causal_br, causal_tl };

struct sdpa_dims_t {
    memory::dim mb;
    memory::dim head_num;
    memory::dim kv_head_num;
    memory::dim seq_len;
    memory::dim query_num;
    memory::dim head_size;

    int kgroup_size;
    int vgroup_size;

    memory::data_type dt;

    memory::data_type qdt;

    memory::data_type kdt;
    memory::data_type ksdt;
    memory::data_type kzpdt;

    memory::data_type vdt;
    memory::data_type vsdt;
    memory::data_type vzpdt;

    memory::data_type mskdt;

    quantize_type qtype;
    bool with_key_transposed;
    mask_type mask;
};

struct sdpa_tensors_t {
    memory m_query, m_key, m_scale, m_mask, m_value, m_output;
    memory m_query_test;
    memory m_key_quantized, m_value_quantized, m_output_quantized;
    memory m_key_t, m_value_t;
    memory m_key_t_quantized, m_value_t_quantized;

    memory m_reorder_scale_attr, m_key_scales, m_key_scales_t, m_key_zp,
            m_value_scales, m_value_zp;
    dnnl::primitive_attr sdpa_attr, sdpa_attr_quantized, sdpa_kq_attr_quantized,
            sdpa_vs_attr_quantized;

    int kq_mask, vs_mask;
    memory::dims kq_groups, vs_groups;
};

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t &p) {
    ss << "mb_" << p.mb;
    if (p.kv_head_num != p.head_num) { ss << "_KVN_" << p.kv_head_num; }
    ss << "_N_" << p.head_num;
    ss << "_D_" << p.head_size;
    if (p.with_key_transposed)
        ss << "_T";
    else
        ss << "_";
    ss << "K_" << p.seq_len;
    ss << "_Q_" << p.query_num;
    ss << "_Qdt_" << p.qdt;
    ss << "_Kdt_" << p.kdt;
    if (!(p.kdt == mdt::f16 || p.kdt == mdt::bf16)
            || p.qtype != quantize_type::no_quantization) {
        ss << "_Ksdt_" << p.ksdt;
        ss << "_Kzpdt_" << p.kzpdt;
    }
    ss << "_Vdt_" << p.vdt;
    if (!(p.vdt == mdt::f16 || p.vdt == mdt::bf16)
            || p.qtype != quantize_type::no_quantization) {
        ss << "_Vsdt_" << p.vsdt;
        ss << "_Vzpdt_" << p.vzpdt;
    }
    switch (p.mask) {
        case mask_type::no_mask: ss << "_no_mask"; break;
        case mask_type::oneD: ss << "_mask1D"; break;
        case mask_type::twoD: ss << "_mask2D"; break;
        case mask_type::causal_br: ss << "_maskcausalbr"; break;
        case mask_type::causal_tl: ss << "_maskcausaltl"; break;
    }
    if ((!(p.kdt == mdt::f16 || p.vdt == mdt::f16)
                || !(p.kdt == mdt::bf16 || p.vdt == mdt::bf16))
            && p.qtype != quantize_type::no_quantization) {
        ss << "_" << p.qtype;
    }
    return ss;
}

std::string print_to_string(const ::testing::TestParamInfo<sdpa_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

void print_table_header() {
    std::cout << "| mb | Q Heads | KV Heads |   D |    K  |    Q | Kdt | Vdt | "
                 "mask | quant |  time (us) | BW eff/actual (Gbps) | "
                 "gemm/total FLOPs (GFLOPs) |\n";
}

std::string print_row(const sdpa_dims_t &p) {
    std::stringstream ss;

    ss << "|" << p.mb;
    ss << "|" << p.head_num;
    ss << "|" << p.kv_head_num;
    ss << "|" << p.head_size;
    ss << "|" << p.seq_len;
    ss << "|" << p.query_num;
    ss << "|" << p.kdt;
    if (!(p.kdt == mdt::f16 || p.vdt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.ksdt;
        ss << "/" << p.kzpdt;
    }
    ss << "|" << p.vdt;
    if (!(p.vdt == mdt::f16 || p.vdt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.vsdt;
        ss << "/" << p.vzpdt;
    }
    ss << "|";
    switch (p.mask) {
        case mask_type::no_mask: ss << "no"; break;
        case mask_type::oneD: ss << "1D"; break;
        case mask_type::twoD: ss << "2D"; break;
        case mask_type::causal_br: ss << "causalbr"; break;
        case mask_type::causal_tl: ss << "causaltl"; break;
    }
    ss << "|" << p.qtype;
    return ss.str();
}

using dnnl::algorithm;
using dnnl::matmul;
using dnnl::memory;
using dnnl::primitive_attr;
using dnnl::softmax_forward;

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#undef CHECK
#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, const memory::desc &desc) {
    size_t seq_len = desc.get_dims()[3];
    size_t query_num = desc.get_dims()[2];
    size_t batches = desc.get_dims()[1] * desc.get_dims()[0];
    for (size_t b = 0; b < batches; b++) {
        for (size_t q = 0; q < query_num; q++) {
            for (size_t i = 0; i < seq_len; i++) {
                if (i <= q) {
                    mask[b * query_num * seq_len + q * seq_len + i] = 0;
                    // = (float)i + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + i]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)i + (float)q / 100.f);
                }
            }
        }
    }
}

void fill_causal_mask(
        std::vector<float> &mask, const memory::desc &desc, mask_type mask_t) {
    int64_t seq_len = desc.get_dims()[3];
    int64_t query_num = desc.get_dims()[2];
    int64_t batches = desc.get_dims()[1] * desc.get_dims()[0];
    for (int64_t b = 0; b < batches; b++) {
        for (int64_t q = 0; q < query_num; q++) {
            for (int64_t k = 0; k < seq_len; k++) {
                if (mask_t == mask_type::causal_br
                                ? ((q + seq_len - query_num) >= k)
                                : (q >= k)) {
                    mask[b * query_num * seq_len + q * seq_len + k] = 0;
                    // = (float)k + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + k]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)k + (float)q / 100.f);
                }
            }
        }
    }
}

memory::dims double_mb(const memory::dims &dims) {
    memory::dims ret = dims;
    ret[0] *= 2;
    return ret;
}

/// This function creates a large tensor double the size requested by /p desc and
/// fills it with NaN values. It then creates a new memory object backed by
/// the first memory handle but with the size of the original memory descriptor.
///
/// This function allows us to identify situations where the SDPA kernel is
/// accessing data out-of-bounds
memory double_and_resize(const memory::desc &desc, dnnl::engine &eng,
        std::vector<dnnl_memory_t> &doubled_memory) {
    dnnl::stream s(eng);
    memory::dims dims2 = double_mb(desc.get_dims());
    auto desc2 = memory::desc(dims2, desc.get_data_type(), desc.get_strides());

    dnnl_memory_t mem2;
    CHECK(dnnl_memory_create(
            &mem2, desc2.get(), eng.get(), DNNL_MEMORY_ALLOCATE));
    doubled_memory.push_back(mem2);

    void *handle;
    CHECK(dnnl_memory_get_data_handle(mem2, &handle));
#ifdef DNNL_WITH_SYCL
    auto sycl_queue = dnnl::sycl_interop::get_queue(s);
    sycl_queue.fill<uint8_t>(handle, 0xFF, desc2.get_size());
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void *mapped_ptr = nullptr;
    CHECK(dnnl_memory_map_data(mem2, &mapped_ptr));
    memset(mapped_ptr, 0xFF, desc2.get_size());
    CHECK(dnnl_memory_unmap_data(mem2, mapped_ptr));
#endif

    auto out = memory(desc, eng, handle);
    return out;
}

sdpa_tensors_t get_descriptors(dnnl::engine &eng, const sdpa_dims_t &p,
        std::vector<dnnl_memory_t> &doubled_memory) {
    sdpa_tensors_t out;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.kv_head_num, p.head_size, p.seq_len};
    const memory::dims k_stride
            = {p.mb, p.kv_head_num, p.head_size, p.seq_len * 2};
    const memory::dims k_t_stride
            = {p.mb, p.kv_head_num, p.seq_len * 2, p.head_size};
    const memory::dims v_sz = {p.mb, p.kv_head_num, p.seq_len, p.head_size};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims key_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        k_sz[0], k_sz[1], k_sz[2] / p.kgroup_size, k_sz[3]};
            case quantize_type::per_token:
                return memory::dims {k_sz[0], k_sz[1], 1, k_sz[3]};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {k_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {k_sz[0], k_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();
    const memory::dims val_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        v_sz[0], v_sz[1], v_sz[2], v_sz[3] / p.vgroup_size};
            case quantize_type::per_token:
                return memory::dims {v_sz[0], v_sz[1], v_sz[2], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {v_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {v_sz[0], v_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();

    memory::dims mask_sz;
    switch (p.mask) {
        case mask_type::no_mask: mask_sz = {};
        case mask_type::oneD: mask_sz = {1, 1, 1, p.seq_len}; break;
        case mask_type::causal_br:
        case mask_type::causal_tl:
        case mask_type::twoD: mask_sz = {1, 1, p.query_num, p.seq_len}; break;
    }

    auto ksdt = p.ksdt == mdt::undef ? p.kdt : p.ksdt;
    auto kzpdt = p.kzpdt == mdt::undef ? mdt::s8 : p.kzpdt;
    auto vsdt = p.vsdt == mdt::undef ? p.vdt : p.vsdt;
    auto vzpdt = p.vzpdt == mdt::undef ? mdt::s8 : p.vzpdt;

    memory::format_tag abcd = memory::format_tag::abcd;
    memory::format_tag abdc = memory::format_tag::abdc;
    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    // clang-format off
    auto query_md            = memory::desc(q_sz,          p.qdt,   abcd);
    auto key_t_md            = memory::desc(k_sz,          p.dt,    abdc);
    auto key_md              = memory::desc(k_sz,          p.dt,    abcd);
    auto value_t_md          = memory::desc(v_sz,          p.dt,    abdc);
    auto value_md            = memory::desc(v_sz,          p.dt,    abcd);
    auto scale_md            = memory::desc(scale_sz,      p.qdt,    abcd);

    auto query_test_md       = memory::desc(q_sz,          p.qdt,   abcd);

    auto key_quantized_md    = memory::desc(k_sz,          p.kdt,   abcd);
    auto key_t_quantized_md  = memory::desc(k_sz,          p.kdt,   abdc);
    auto key_scales_md       = memory::desc(key_scales_sz, ksdt,    abcd);
    auto key_scales_t_md     = memory::desc(key_scales_sz, ksdt,    abdc);
    auto key_zp_md           = memory::desc(key_scales_sz, kzpdt,   abcd);

    auto val_quantized_md    = memory::desc(v_sz,          p.vdt,   abcd);
    auto val_t_quantized_md  = memory::desc(v_sz,          p.vdt,   abdc);
    auto val_scales_md       = memory::desc(val_scales_sz, vsdt,    abcd);
    auto val_zp_md           = memory::desc(val_scales_sz, vzpdt,   abcd);


    auto mask_md             = memory::desc(mask_sz,       p.mskdt, abcd);
    auto output_md           = memory::desc(q_sz,          p.qdt,   abcd);
    auto output_quantized_md = memory::desc(q_sz,          p.qdt,   abcd);
    // clang-format on

    // Create memory objects
    out.m_query = double_and_resize(query_md, eng, doubled_memory);
    out.m_query_test = double_and_resize(query_test_md, eng, doubled_memory);
    out.m_key = double_and_resize(key_md, eng, doubled_memory);
    out.m_key_t = double_and_resize(key_t_md, eng, doubled_memory);
    out.m_scale = double_and_resize(scale_md, eng, doubled_memory);
    out.m_key_quantized
            = double_and_resize(key_quantized_md, eng, doubled_memory);
    out.m_key_t_quantized
            = double_and_resize(key_t_quantized_md, eng, doubled_memory);
    out.m_key_scales = double_and_resize(key_scales_md, eng, doubled_memory);
    out.m_key_scales_t
            = double_and_resize(key_scales_t_md, eng, doubled_memory);
    out.m_key_zp = double_and_resize(key_zp_md, eng, doubled_memory);
    out.m_value_quantized
            = double_and_resize(val_quantized_md, eng, doubled_memory);
    out.m_value_t_quantized
            = double_and_resize(val_t_quantized_md, eng, doubled_memory);
    out.m_value_scales = double_and_resize(val_scales_md, eng, doubled_memory);
    out.m_value_zp = double_and_resize(val_zp_md, eng, doubled_memory);
    out.m_mask = double_and_resize(mask_md, eng, doubled_memory);
    out.m_value = double_and_resize(value_md, eng, doubled_memory);
    out.m_value_t = double_and_resize(value_t_md, eng, doubled_memory);
    out.m_output = double_and_resize(output_md, eng, doubled_memory);
    out.m_output_quantized
            = double_and_resize(output_quantized_md, eng, doubled_memory);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz), 0.f);
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> key_quantized_data(product(k_sz), 0);
    std::vector<float> val_quantized_data(product(v_sz), 0);
    std::vector<float> key_scale_data(product(key_scales_sz), std::nanf("1"));
    std::vector<float> val_scale_data(product(val_scales_sz), std::nanf("1"));

    std::vector<int> key_zp_data_signed(product(key_scales_sz), INT_MAX);
    std::vector<int> val_zp_data_signed(product(val_scales_sz), INT_MAX);

    std::vector<unsigned> key_zp_data_unsigned(product(key_scales_sz), INT_MAX);
    std::vector<unsigned> val_zp_data_unsigned(product(val_scales_sz), INT_MAX);

    std::vector<float> mask_data(product(mask_sz), NAN);
    std::vector<float> output_data(product(q_sz), NAN);

    out.sdpa_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    out.sdpa_attr_quantized.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    out.kq_mask = 0;
    out.vs_mask = 0;
    out.kq_groups = {};
    out.vs_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            out.kq_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.kq_groups = {p.kgroup_size, 1};
            out.vs_groups = {1, p.vgroup_size};
            break;
        case quantize_type::per_token:
            out.kq_mask = 1 << 3 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 0 | 1 << 1 | 1 << 2;
            break;
        case quantize_type::per_tensor3:
            out.kq_mask = 3;
            out.vs_mask = 3;
            break;
        case quantize_type::per_tensor1:
            out.kq_mask = 1;
            out.vs_mask = 1;
            break;
        case quantize_type::per_tensor:
            out.kq_mask = 0;
            out.vs_mask = 0;
            break;
        case quantize_type::no_quantization: break;
    }

    if (p.qtype != quantize_type::no_quantization) {
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.ksdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.ksdt);
        }

        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vsdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.vsdt);
        }

        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.kzpdt);
        }

        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.vzpdt);
        }
    }

    fill_random(query_data, query_md);
    fill_random_quantized(key_quantized_data, key_quantized_md,
            (p.kdt == mdt::u4 || p.kdt == mdt::u8));
    fill_random_quantized(val_quantized_data, val_quantized_md,
            (p.vdt == mdt::u4 || p.vdt == mdt::u8));
    if (p.qtype != quantize_type::no_quantization) {
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.ksdt != mdt::undef) {
            fill_random_scales(key_scale_data, key_scales_md);
        } else {
            fill_value(key_scale_data, key_scales_md, 1.f);
        }
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vsdt != mdt::undef) {
            fill_random_scales(val_scale_data, val_scales_md);
        } else {
            fill_value(val_scale_data, val_scales_md, 1.f);
        }
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_signed, key_zp_md);
        } else {
            fill_value(key_zp_data_signed, key_zp_md, 0);
        }
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_signed, val_zp_md);
        } else {
            fill_value(val_zp_data_signed, val_zp_md, 0);
        }
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_unsigned, key_zp_md);
        } else {
            fill_value(key_zp_data_unsigned, key_zp_md, 0U);
        }
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_unsigned, val_zp_md);
        } else {
            fill_value(val_zp_data_unsigned, val_zp_md, 0U);
        }
    }

    if (p.mask == mask_type::causal_br || p.mask == mask_type::causal_tl) {
        fill_causal_mask(mask_data, mask_md, p.mask);
    } else {
        fill_mask(mask_data, mask_md);
    }

/// This section allows setting the values of the tensors using environment variables.
/// Syntax:
///    <Tensor Name>[<S for scales, Z for zero points>]<R for row C for column>
///
/// KR=3 KC=1 Set the value in the  Key tensor at (3, 1) to 1 and all other values should be zero
/// VSR=1 VSC=2  Set the scale for the Value tensor at (1, 2) to 1 and all other values to zero
#if 0
    auto &Q = query_data;
    auto &K = key_quantized_data;
    auto &V = val_quantized_data;
    auto &Ks = key_scale_data;
    auto &Vs = val_scale_data;
    auto &Kz = key_zp_data_signed;
    auto &Vz = val_zp_data_signed;
    auto d = p.head_size;
    auto k = p.seq_len;
    auto q = p.query_num;

    int kr = -1, kc = -1, qr = -1, qc = -1, vr = -1, vc = -1, mr = -1, mc = -1,
        xb = 0;
    int ksr = -1, ksc = -1, kzr = -1, kzc = -1, vsr = -1, vscales = -1,
        vzr = -1, vzc = -1;
    if (getenv("KR")) kr = atoi(getenv("KR"));
    if (getenv("KC")) kc = atoi(getenv("KC"));
    if (getenv("KSR")) ksr = atoi(getenv("KSR"));
    if (getenv("KSC")) ksc = atoi(getenv("KSC"));
    if (getenv("KZR")) kzr = atoi(getenv("KZR"));
    if (getenv("KZC")) kzc = atoi(getenv("KZC"));
    if (getenv("QR")) qr = atoi(getenv("QR"));
    if (getenv("QC")) qc = atoi(getenv("QC"));
    if (getenv("VR")) vr = atoi(getenv("VR"));
    if (getenv("VC")) vc = atoi(getenv("VC"));
    if (getenv("VSR")) vsr = atoi(getenv("VSR"));
    if (getenv("VScaleC")) vscales = atoi(getenv("VScaleC"));
    if (getenv("VZR")) vzr = atoi(getenv("VZR"));
    if (getenv("VZC")) vzc = atoi(getenv("VZC"));
    if (getenv("XB")) xb = atoi(getenv("XB"));

    if (getenv("MR")) mr = atoi(getenv("MR"));
    if (getenv("MC")) mc = atoi(getenv("MC"));

    if (mr >= 0 || mc >= 0) {
        mr = std::max(mr, 0);
        mc = std::max(mc, 0);
        for (auto &m : mask_data)
            m = 0;
        mask_data[mr * p.seq_len + mc] = -999;
    }
    if (kr >= 0 || kc >= 0) {
        kr = std::max(kr, 0);
        kc = std::max(kc, 0);
        if (getenv("KX")) {
            for (int kr_ = 0; kr_ < d; kr_++)
                for (int kc_ = 0; kc_ < k; kc_++)
                    if (kr_ >= kr || kc_ >= kc) K[kr_ * k + kc_] = 0;
        } else {
            for (auto &k : K)
                k = 0;
            K[xb * d * k + kr * k + kc] = 1;
        }
    }
    if (ksr >= 0 || ksc >= 0) {
        ksr = std::max(ksr, 0);
        ksc = std::max(ksc, 0);
        for (auto &ks : Ks)
            ks = 0;
        Ks[(xb * d / p.kgroup_size * k + ksr * k) + ksc] = 1;
    }
    if (kzr >= 0 || kzc >= 0) {
        kzr = std::max(kzr, 0);
        kzc = std::max(kzc, 0);
        for (auto &kz : Kz)
            kz = 0;
        Kz[(xb * d * k + kzr * d) / p.kgroup_size + kzc] = 2;
    }
    if (qr >= 0 || qc >= 0) {
        qr = std::max(qr, 0);
        qc = std::max(qc, 0);
        if (getenv("QX")) {
            for (int qr_ = 0; qr_ < d; qr_++)
                for (int qc_ = 0; qc_ < q; qc_++)
                    if (qr_ >= qr || qc_ >= qc) Q[qr_ * d + qc_] = 0;
        } else {
            for (auto &q : Q)
                q = 0;
            Q[xb * d * q + qr * d + qc] = 1;
        }
    }
    if (vr >= 0 || vc >= 0) {
        vr = std::max(vr, 0);
        vc = std::max(vc, 0);
        if (getenv("VX")) {
            for (int vr_ = 0; vr_ < k; vr_++)
                for (int vc_ = 0; vc_ < d; vc_++)
                    if (vr_ >= vr || vc_ >= vc) V[vr_ * d + vc_] = 0;
        } else {
            for (auto &v : V)
                v = 0;
            V[xb * d * k + vr * d + vc] = 1;
        }
    }
    if (vsr >= 0 || vscales >= 0) {
        vsr = std::max(vsr, 0);
        vscales = std::max(vscales, 0);
        for (auto &vs : Vs)
            vs = 0;
        Vs[(xb * d * k + vscales * d) / p.vgroup_size + vsr] = 1;
    }
    if (vzr >= 0 || vzc >= 0) {
        vzr = std::max(vzr, 0);
        vzc = std::max(vzc, 0);
        for (auto &vz : Vz)
            vz = 0;
        Vz[(xb * d * k + vzc * d) / p.vgroup_size + vzr] = 1;
    }
#endif

    int group_size = p.kgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = k_sz[0] * k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = k_sz[2] * k_sz[3];
    }

    std::vector<float> key_data;
    if (p.kzpdt == mdt::s4 || p.kzpdt == mdt::s8) {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_signed, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    } else {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_unsigned, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    }

    group_size = p.vgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = v_sz[0] * v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = v_sz[2] * v_sz[3];
    }
    std::vector<float> value_data;
    if (p.vzpdt == mdt::s4 || p.vzpdt == mdt::s8) {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_signed, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    } else {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_unsigned, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    }

    write_to_dnnl_memory(mask_data.data(), out.m_mask);
    write_to_dnnl_memory(scale_data.data(), out.m_scale);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(key_data.data(), out.m_key);
    write_to_dnnl_memory(value_data.data(), out.m_value);
    write_to_dnnl_memory(query_data.data(), out.m_query);
    write_to_dnnl_memory(query_data.data(), out.m_query_test);

    write_to_dnnl_memory(key_quantized_data.data(), out.m_key_quantized);

    write_to_dnnl_memory(val_quantized_data.data(), out.m_value_quantized);
    if (p.kzpdt == mdt::s4 || p.kzpdt == mdt::s8) {
        write_to_dnnl_memory(key_zp_data_signed.data(), out.m_key_zp);
    } else {
        write_to_dnnl_memory(key_zp_data_unsigned.data(), out.m_key_zp);
    }
    if (p.vzpdt == mdt::s4 || p.vzpdt == mdt::s8) {
        write_to_dnnl_memory(val_zp_data_signed.data(), out.m_value_zp);
    } else {
        write_to_dnnl_memory(val_zp_data_unsigned.data(), out.m_value_zp);
    }
    write_to_dnnl_memory(key_scale_data.data(), out.m_key_scales);
    write_to_dnnl_memory(val_scale_data.data(), out.m_value_scales);
    write_to_dnnl_memory(output_data.data(), out.m_output);
    write_to_dnnl_memory(output_data.data(), out.m_output_quantized);

    transpose_strides(eng, out.m_key_scales_t, out.m_key_scales);
    transpose_strides(eng, out.m_key_t, out.m_key);
    transpose_strides(eng, out.m_key_t_quantized, out.m_key_quantized);
    transpose_strides(eng, out.m_value_t, out.m_value);
    transpose_strides(eng, out.m_value_t_quantized, out.m_value_quantized);

    return out;
}

static std::unique_ptr<dnnl::engine> sdpa_eng;

dnnl::engine get_sdpa_test_engine() {
    return *sdpa_eng;
}

class sdpa_test_t : public ::testing::TestWithParam<sdpa_dims_t> {
public:
    // Testing reusable functionality requires shared engine between tests.
    static void SetUpTestSuite() {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "SDPA primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "SDPA primitive tests do not support HIP";
#endif
#ifdef DNNL_SYCL_GENERIC
        GTEST_SKIP() << "no supported generic implementations for SDPA "
                        "primitive tests";
#endif
#ifndef DNNL_TEST_WITH_ENGINE_PARAM
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "SDPA tests require gpus.");
        sdpa_eng.reset(new dnnl::engine(engine::kind::gpu, 0));
#endif
    }

    void SetUp() override {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "SDPA primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "SDPA primitive tests do not support HIP";
#endif
#ifdef DNNL_SYCL_GENERIC
        GTEST_SKIP() << "no supported generic implementations for SDPA "
                        "primitive tests";
#endif
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
        SKIP_IF(get_test_engine_kind() != dnnl::engine::kind::gpu,
                "This test requires GPU engine");
        eng = get_test_engine();
#else
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "SDPA tests require gpus.");
        eng = get_sdpa_test_engine();
#endif
        strm = dnnl::stream(eng);
        p = GetParam();
        doubled_memory.reserve(30);
        t = get_descriptors(eng, p, doubled_memory);
    }

    void TearDown() override {
        for (dnnl_memory_t &mem : doubled_memory) {
            CHECK(dnnl_memory_destroy(mem));
        }
    }

    static void TearDownTestSuite() {
#ifndef DNNL_TEST_WITH_ENGINE_PARAM
        sdpa_eng.reset();
#endif
    }

protected:
    dnnl::engine eng;
    dnnl::stream strm;
    sdpa_dims_t p;
    sdpa_tensors_t t;
    std::vector<dnnl_memory_t> doubled_memory;
};

bool with_key_transposed = true;
bool no_key_transposed = false;

// clang-format off
INSTANTIATE_TEST_SUITE_P(AllMaskTypes,
    sdpa_test_t,
                               //  mb, hd_num,kv_hd_num,seq_len,qry_num, hd_size, kg_sz,  vgrp_sz,        dt,       qdt,       kdt,       ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,     mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::oneD},
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::oneD},
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::causal_br },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::causal_br },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,      128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,         no_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   1,      10,       10,     77,   2304,      64,    64,       64, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization, with_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   1,      10,       10,   2304,     77,      64,    64,       64, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization, with_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   1,      10,       10,     77,   2304,      64,    64,       64, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization, with_key_transposed, mask_type::causal_br },
                    sdpa_dims_t{   1,      10,       10,   2304,     77,      64,    64,       64, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization, with_key_transposed, mask_type::causal_br }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(DataTypes_bf16_s8,
    sdpa_test_t,
                              //  mb,  hd_num,kv_hd_num,seq_len,qry_num, hd_size, kg_sz, vgrp_sz,        dt,       qdt,       kdt,       ksdt,       kzpdt,       vdt,       vsdt,      vzpdt,     mskdt,                    qtype
    testing::Values(
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8, mdt::undef, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,  mdt::bf16, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,  mdt::bf16, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,  mdt::bf16,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,  mdt::bf16,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },

                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    32,      32,     32,     32,      32, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::no_mask },
                    //sdpa_dims_t{   1,       2,        2,    33,       1,     32,     32,      32, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32, mdt::undef, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::no_mask },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //*sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, mdt::undef, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8, mdt::undef, mdt::undef, mdt::bf16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD }
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32, mdt::undef, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16, mdt::bf16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32,    mdt::s8,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    //sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128, mdt::bf16, mdt::bf16,   mdt::s8,   mdt::f32,    mdt::s8,   mdt::s8,   mdt::f32,    mdt::s8, mdt::bf16, quantize_type::per_token,        no_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s8,
    sdpa_test_t,
                              //  mb,  hd_num,kv_hd_num,seq_len,qry_num, hd_size, kg_sz, vgrp_sz,        dt,       qdt,       kdt,       ksdt,       kzpdt,       vdt,       vsdt,      vzpdt,     mskdt,                    qtype
    testing::Values(
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32, mdt::undef,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32,    mdt::s8,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    385,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s8,   mdt::f32,    mdt::s8,   mdt::s8,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,        no_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s4,
    sdpa_test_t,
                              //  mb,  hd_num,kv_hd_num,seq_len,qry_num, hd_size, kg_sz, vgrp_sz,        dt,       qdt,       kdt,       ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,     mskdt,                    qtype
    testing::Values(
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4, mdt::undef, mdt::undef,   mdt::s4, mdt::undef, mdt::undef,  mdt::f16, quantize_type::no_quantization,       no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4, mdt::undef, mdt::undef,   mdt::s4, mdt::undef, mdt::undef,  mdt::f16, quantize_type::no_quantization,       no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,   128,     128,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token,             with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16, mdt::undef,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f16,    mdt::s4,   mdt::s4,   mdt::f16,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32, mdt::undef,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32, mdt::undef,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    384,    384,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,       2,        2,    386,      1,     128,    64,      64,  mdt::f16,  mdt::f16,   mdt::s4,   mdt::f32,    mdt::s4,   mdt::s4,   mdt::f32,    mdt::s8,  mdt::f16, quantize_type::per_token_with_groups, with_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(GQA,
    sdpa_test_t,
                              //  mb,  hd_num,kv_hd_num,seq_len,qry_num, hd_size, kg_sz, vgrp_sz,        dt,       qdt,       kdt,       ksdt,       kzpdt,       vdt,       vsdt,      vzpdt,     mskdt,                    qtype
    testing::Values(
                    sdpa_dims_t{   1,       4,        2,    384,    384,       128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,       8,        2,    384,    384,       128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,       8,        4,    384,    384,       128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,      32,       16,    384,    384,       128,   128,     128,  mdt::f16,  mdt::f16,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef,  mdt::f16, quantize_type::per_token,             no_key_transposed, mask_type::no_mask }
    ), &print_to_string);

//llama-2-7b-chat shape: Q [1x32xSEQ_LENx128] KV [1x32xSEQ_LENx128]
//llama-3-8b shape: Q [1x32xSEQ_LENx128] KV [1x8xSEQ_LENx128]
//minicpm-1b-sft shape:  Q [1x24xSEQ_LENx64]  KV [1x8xSEQ_LENx64]
//qwen2-7b shape: Q [1x28xSEQ_LENx128] KV [1x4xSEQ_LENx128]
//phi3-mini-4k-instruct shape: Q [1x32xSEQ_LENx96] KV [1x32xSEQ_LENx96]

INSTANTIATE_TEST_SUITE_P(llama_2_7b_chat,
    sdpa_test_t,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       qdt,       kdt,        ksdt,      kzpdt,        vdt,       vsdt,      vzpdt,     mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    32,       32,    384,    384,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    385,      1,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    512,    512,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    513,      1,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1024,   1024,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1025,      1,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2048,   2048,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2049,      1,    128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(llama_3_8b,
    sdpa_test_t,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,      qdt,       kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    32,        8,    384,    384,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    386,    386,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    385,      1,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    512,    512,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    513,      1,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1024,   1024,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1025,      1,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2048,   2048,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2049,      1,    128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    384,    384,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    385,      1,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    512,    512,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    513,      1,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1024,   1024,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1025,      1,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2048,   2048,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2049,      1,    128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(llama_f32,
    sdpa_test_t,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,      qdt,       kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    2,        2,    384,    384,    32,   32,     32, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    385,      1,    32,   32,     32, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    64,   64,     64, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    385,      1,    64,   64,     64, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    128,   128,     128, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    385,      1,    128,   128,     128, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    256,   256,     256, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    385,      1,    256,   256,     256, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    512,   512,     512, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,      1,    512,   512,     512, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },

                    sdpa_dims_t{   1,    2,        2,   1024,   1024,     32,    32,      32, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,     32,    32,      32, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,     64,    64,      64, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,     64,    64,      64, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    128,   128,     128, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    128,   128,     128, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    256,   256,     256, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    256,   256,     256, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    512,   512,     512, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    512,   512,     512, mdt::f32, mdt::f32,  mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },


                    sdpa_dims_t{   1,    2,        2,    384,    384,     32,    32,      32, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,     32,    32,      32, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,     32,    32,      32, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,     64,    64,      64, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,     64,    64,      64, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,     64,    64,      64, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    128,   128,     128, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    128,   128,     128, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    128,   128,     128, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    256,   256,     256, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    256,   256,     256, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    256,   256,     256, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,    384,    384,    512,   512,     512, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1024,   1024,    512,   512,     512, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    2,        2,   1025,      1,    512,   512,     512, mdt::f32, mdt::f32,  mdt::s8,  mdt::f32, mdt::undef,  mdt::s8, mdt::f32, mdt::undef, mdt::f32, quantize_type::per_token_with_groups,        with_key_transposed, mask_type::twoD }
    ), &print_to_string);





INSTANTIATE_TEST_SUITE_P(minicpm_1b_st,
    sdpa_test_t,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       qdt,     kdt,      ksdt,      kzpdt,      vdt,     vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    24,        8,    384,    384,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    385,      1,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    512,    512,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    513,      1,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1024,   1024,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1025,      1,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2048,   2048,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2049,      1,     64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(qwen2_7b,
    sdpa_test_t,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,        qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,  vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    28,        4,    384,    384,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    385,      1,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    512,    512,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    513,      1,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1024,   1024,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1025,      1,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2048,   2048,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2049,      1,    128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


//INSTANTIATE_TEST_SUITE_P(phi3_mini_4k_instruct,
//    sdpa_test_t,
//                               // mb,  hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,        qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt,    mskdt, qtype
//    testing::Values(
//                    sdpa_dims_t{   1,      2,        2,    384,     384,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,    384,     384,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::oneD },
//                    sdpa_dims_t{   1,      2,        2,    384,     384,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::no_mask },
//                    sdpa_dims_t{   1,      2,        2,    385,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,    512,     512,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,    513,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,   1024,    1024,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,   1025,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,   2048,    2048,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
//                    sdpa_dims_t{   1,      2,        2,   2049,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
//    ), &print_to_string);

// clang-format on

memory as(dnnl::stream &strm, memory &mem, memory::data_type dt) {
    const memory::dims sz = mem.get_desc().get_dims();

    auto md = memory::desc(sz, dt, mem.get_desc().get_strides());
    auto out = memory(md, mem.get_engine());
    dnnl::reorder(mem, out).execute(strm, mem, out);
    return out;
}

memory reshape(dnnl::stream &strm, memory &mem, const memory::desc &md) {
    auto out = memory(md, mem.get_engine());

#ifdef DNNL_WITH_SYCL
    size_t size = md.get_size();
    auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
    if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
        auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
        auto src = buffer.get_host_access();
        uint8_t *src_ptr = src.get_pointer();
        if (!src_ptr) throw std::runtime_error("get_pointer returned nullptr.");

        auto out_buffer = dnnl::sycl_interop::get_buffer<uint8_t>(out);
        auto dst = out_buffer.get_host_access();
        uint8_t *dst_ptr = dst.get_pointer();
        if (!dst_ptr) throw std::runtime_error("get_pointer returned nullptr.");

        for (size_t i = 0; i < size; ++i)
            ((uint8_t *)dst_ptr)[i] = src_ptr[i];
    } else {
        assert(mkind == dnnl::sycl_interop::memory_kind::usm);
        uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
        uint8_t *dst_ptr = (uint8_t *)out.get_data_handle();
        if (!src_ptr)
            throw std::runtime_error("get_data_handle returned nullptr.");
        auto sycl_queue = dnnl::sycl_interop::get_queue(strm);
        sycl_queue.memcpy(dst_ptr, src_ptr, size).wait();
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    strm.wait();
    void *mem_ptr_ = (void *)mem.map_data();
    if (mem_ptr_ == nullptr)
        throw std::runtime_error("Failed to map mem in resize");
    void *out_ptr_ = (void *)out.map_data();
    if (out_ptr_ == nullptr)
        throw std::runtime_error("Failed to map out in resize");
    memcpy(out_ptr_, mem_ptr_, mem.get_desc().get_size());
    mem.unmap_data(mem_ptr_);
    out.unmap_data(out_ptr_);
#endif
    return out;
}

std::pair<dnnl::reorder, memory> dequantize_prim(const engine &eng, mdt dt,
        const memory::desc &desc, int mask, const memory::dims &groups, mdt sdt,
        mdt zpdt, dnnl::memory::format_tag tag = memory::format_tag::abcd) {
    auto dequantized_md = memory::desc(desc.get_dims(), dt, tag);
    primitive_attr dequantized_attr;

    if (sdt != mdt::undef) {
        dequantized_attr.set_scales(DNNL_ARG_FROM, mask, groups, sdt);
    }
    if (zpdt != mdt::undef) {
        dequantized_attr.set_zero_points(DNNL_ARG_SRC, mask, groups, zpdt);
    }

    auto dequantize_pd = dnnl::reorder::primitive_desc(
            eng, desc, eng, dequantized_md, dequantized_attr, false);

    memory dequantized_mem
            = memory({desc.get_dims(), dt, memory::format_tag::abcd}, eng);
    return std::make_pair(dnnl::reorder(dequantize_pd), dequantized_mem);
}

void prim_sdpa_quant(const sdpa_dims_t &p, const sdpa_tensors_t &t,
        dnnl::engine &eng, dnnl::stream &strm, dnnl::memory &query,
        dnnl::memory &key, dnnl::memory &key_scales, dnnl::memory &key_zp,
        dnnl::memory::data_type scale_dt, dnnl::memory &scale,
        dnnl::memory &mask, dnnl::memory &value, dnnl::memory &value_scales,
        dnnl::memory &value_zp, dnnl::memory &output, bool invert_scale,
        std::vector<dnnl_memory_t> &doubled_memory) {
    using namespace dnnl;
    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    post_ops bmm1_po;
    auto scale_f32 = as(strm, scale, mdt::f32);
    auto mask_f32 = as(strm, mask, mdt::f32);
    auto mask_sz = mask.get_desc().get_dims();

    scale_f32 = reshape(strm, scale_f32,
            {{1, 1, 1, 1, 1}, mdt::f32, memory::format_tag::abcde});
    mask_f32 = reshape(strm, mask_f32,
            {{mask_sz[0], 1, 1, mask_sz[2], mask_sz[3]}, mdt::f32,
                    memory::format_tag::abcde});
    if (scale_dt != mdt::undef) {
        if (invert_scale)
            bmm1_po.append_binary(algorithm::binary_div, scale_f32.get_desc());
        else
            bmm1_po.append_binary(algorithm::binary_mul, scale_f32.get_desc());
    }
    if (p.mask != mask_type::no_mask) {
        bmm1_po.append_binary(algorithm::binary_add, mask_f32.get_desc());
    }

    bmm1_attr.set_post_ops(bmm1_po);

    int head_kv_group_size = 0;
    int head_q_group_size = 0;
    int head_group_batches = 0;
    if (p.kv_head_num == p.head_num) {
        head_kv_group_size = p.kv_head_num;
        head_q_group_size = p.head_num;
        head_group_batches = 1;
    } else {
        head_kv_group_size = 1;
        head_q_group_size = p.head_num / p.kv_head_num;
        head_group_batches = p.kv_head_num;
    }

    auto original_k_sz = key.get_desc().get_dims();
    const memory::dims k_sz {p.mb, head_group_batches, head_kv_group_size,
            original_k_sz[2], original_k_sz[3]};
    const memory::dims v_sz {p.mb, head_group_batches, head_kv_group_size,
            p.seq_len, p.head_size};
    const memory::dims q_sz {p.mb, head_group_batches, head_q_group_size,
            p.query_num, p.head_size};
    const memory::dims m_sz {1, 1, 1, mask.get_desc().get_dims()[2],
            mask.get_desc().get_dims()[3]};
    memory::desc grouped_key_md(k_sz, p.dt, memory::format_tag::abcde);
    memory::desc grouped_value_md(v_sz, mdt::f32, memory::format_tag::abcde);
    memory::desc grouped_query_md(q_sz, p.qdt, memory::format_tag::abcde);
    memory::desc grouped_mask_md(
            m_sz, mask.get_desc().get_data_type(), memory::format_tag::abcde);

    memory key_dequantized;
    if ((key.get_desc().get_data_type() != mdt::f16
                && key.get_desc().get_data_type() != mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {

        dnnl::reorder key_dequantize_prim;
        std::tie(key_dequantize_prim, key_dequantized) = dequantize_prim(eng,
                p.dt, key.get_desc(), t.kq_mask, t.kq_groups, p.ksdt, p.kzpdt);

        std::unordered_map<int, memory> key_dequantize_args = {
                {DNNL_ARG_FROM, key},
                {DNNL_ARG_TO, key_dequantized},
        };
        if (p.ksdt != mdt::undef) {
            key_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = key_scales;
        }
        if (p.kzpdt != mdt::undef)
            key_dequantize_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM]
                    = key_zp;
        key_dequantize_prim.execute(strm, key_dequantize_args);
        key_dequantized = reshape(strm, key_dequantized, grouped_key_md);
    } else {
        auto keytmp = as(strm, key, p.dt);
        grouped_key_md = p.with_key_transposed
                ? memory::desc(k_sz, p.dt, memory::format_tag::abced)
                : memory::desc(k_sz, p.dt, memory::format_tag::abcde);

        key_dequantized = reshape(strm, keytmp, grouped_key_md);
    }

    memory value_dequantized;
    if (value.get_desc().get_data_type() != mdt::f16
            && value.get_desc().get_data_type() != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        dnnl::reorder value_dequantize_prim;
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f32, value.get_desc(), t.vs_mask,
                        t.vs_groups, p.vsdt, p.vzpdt);

        std::unordered_map<int, memory> value_dequantize_args = {
                {DNNL_ARG_FROM, value},
                {DNNL_ARG_TO, value_dequantized},
        };
        if (p.vsdt != mdt::undef) {
            value_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = value_scales;
        }
        if (p.vzpdt != mdt::undef)
            value_dequantize_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM]
                    = value_zp;
        value_dequantize_prim.execute(strm, value_dequantize_args);
        value_dequantized = reshape(strm, value_dequantized, grouped_value_md);
    } else {
        auto value32 = as(strm, value, mdt::f32);
        value_dequantized = reshape(strm, value32, grouped_value_md);
    }

    memory grouped_query = reshape(strm, query, grouped_query_md);

    const memory::dims score_sz = {p.mb, head_group_batches, head_q_group_size,
            p.query_num, p.seq_len};
    memory::desc score_md {score_sz, mdt::f32, memory::format_tag::abcde};

    auto score = memory(score_md, eng);
    auto score2 = memory(score_md, eng);
    auto bmm1_pd = matmul::primitive_desc(eng, grouped_query_md,
            key_dequantized.get_desc(), score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::library);
    auto softmax_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_inference,
            (algorithm)dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
            score.get_desc(), score.get_desc(), 4, softmax_attr);
    auto softmax_prim = softmax_forward(softmax_pd);

    // attention_output = attention_probs x value
    primitive_attr bmm2_attr;

    bmm2_attr.set_scratchpad_mode(scratchpad_mode::library);
    auto grouped_output
            = double_and_resize(grouped_query_md, eng, doubled_memory);
    auto bmm2_pd = matmul::primitive_desc(
            eng, score_md, grouped_value_md, grouped_query_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    std::unordered_map<int, memory> bmm1_args = {{DNNL_ARG_SRC, grouped_query},
            {DNNL_ARG_WEIGHTS, key_dequantized}, {DNNL_ARG_DST, score}};

    if (scale_dt != mdt::undef) {
        bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                = scale_f32;
        if (p.mask != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = mask_f32;
        }
    } else {
        if (p.mask != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                    = mask_f32;
        }
    }

    const auto loop = [&]() {
        bmm1_prim.execute(strm, bmm1_args);
        //strm.wait();
        //print_mem(score, "score");

        softmax_prim.execute(strm,
                {
                        {DNNL_ARG_SRC, score},
                        {DNNL_ARG_DST, score2},
                });
        //strm.wait();
        //print_mem(score2, "score2");

        bmm2_prim.execute(strm,
                {
                        {DNNL_ARG_SRC, score2},
                        {DNNL_ARG_WEIGHTS, value_dequantized},
                        {DNNL_ARG_DST, grouped_output},
                });
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

#ifdef DNNL_WITH_SYCL
    // copy data from grouped_output to output
    auto sycl_queue = dnnl::sycl_interop::get_queue(strm);
    sycl_queue.memcpy(output.get_data_handle(),
            grouped_output.get_data_handle(), grouped_query_md.get_size());
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    strm.wait();
    void *output_ptr_ = (void *)output.map_data();
    void *grouped_output_ptr_ = (void *)grouped_output.map_data();
    memcpy(output_ptr_, grouped_output_ptr_, grouped_query_md.get_size());
    grouped_output.unmap_data(grouped_output_ptr_);
    output.unmap_data(output_ptr_);
#endif

    strm.wait();
}

template <typename T>
void check_memory(memory &gold, memory &test, dnnl::stream &strm) {

    T *mapped_ptr_gold = nullptr;
    T *mapped_ptr_test = nullptr;
#ifdef DNNL_WITH_SYCL
    auto mkind = dnnl::sycl_interop::get_memory_kind(gold);

    size_t size = gold.get_desc().get_size();
    std::vector<uint8_t> gold_data(size);
    std::vector<uint8_t> test_data(size);
    mapped_ptr_gold = (T *)gold_data.data();
    mapped_ptr_test = (T *)test_data.data();

    if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
    } else {
        uint8_t *ptr_gold = (uint8_t *)gold.get_data_handle();
        uint8_t *ptr_test = (uint8_t *)test.get_data_handle();

        auto sycl_queue = dnnl::sycl_interop::get_queue(strm);
        sycl_queue.memcpy(gold_data.data(), ptr_gold, size);
        sycl_queue.memcpy(test_data.data(), ptr_test, size).wait();
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    mapped_ptr_gold = (T *)gold.map_data();
    mapped_ptr_test = (T *)test.map_data();
#endif
    strm.wait();

    auto dims = gold.get_desc().get_dims();
    auto strides = gold.get_desc().get_strides();

    int mismatches = 0;
    int total = 0;
    float fthreshold = 0.f;
    if (std::is_same<T, float16_t>::value) {
        fthreshold = 0.001466f;
    } else {
        fthreshold = 0.0079f;
    }

    float max_diff = std::numeric_limits<float>::min();
    std::map<int, std::map<int, int>> hist;
    bool verbose = false;
    for_(int l = 0; l < dims[0]; l++)
    for_(int k = 0; k < dims[1]; k++)
    for_(int j = 0; j < dims[2]; j++)
    for (int i = 0; i < dims[3]; i++) {
        auto offset = l * strides[0] + k * strides[1] + j * strides[2]
                + i * strides[3];
        auto o_gold = (float)mapped_ptr_gold[offset];
        auto o_test = (float)mapped_ptr_test[offset];
        total++;

        float abs_diff = abs(o_gold - o_test);
        bool is_nan = isnan(o_gold) || isnan(o_test);

        bool is_mismatch = is_nan
                || (abs(o_gold) > 1.f ? abs_diff > abs(o_gold * fthreshold)
                                      : abs_diff > fthreshold);
        if (max_diff < abs_diff) {
            if (verbose) {
                printf("new max: gold: %f vs test: %f diff: %f\n", o_gold,
                        o_test, abs_diff);
            }
            max_diff = abs_diff;
        }
        if (is_mismatch) {
            hist[0][l]++;
            hist[1][k]++;
            hist[2][j]++;
            hist[3][i]++;
        }
        if ((is_mismatch && mismatches++ < 32)) {
            if (verbose)
                fprintf(stderr,
                        "Mismatch at (%d,%d,%d,%d): test %f "
                        "vs. gold %f (diff: %f thresh: %f)\n",
                        l, k, j, i, o_test, o_gold, abs_diff,
                        (abs(o_gold) > 2.f ? abs(o_gold * fthreshold)
                                           : fthreshold));
        }
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    gold.unmap_data(mapped_ptr_gold);
    test.unmap_data(mapped_ptr_test);
#endif

    int threshold = total * 0.0006;

    ASSERT_LE(mismatches, threshold)
            << "max diff: " << max_diff << " out of: " << total;
}

int to_attn_mask_type(mask_type t) {
    using namespace dnnl::impl::attn_mask_type;
    auto attn_mask = buffer;
    switch (t) {
        case mask_type::causal_tl: attn_mask = top_left; break;
        case mask_type::causal_br: attn_mask = bottom_right; break;
        default:;
    }
    return static_cast<int>(attn_mask);
}

GPU_TEST_P(sdpa_test_t, compare) {
    memory::data_type scale_dt = t.m_query_test.get_desc().get_data_type();
    //memory::data_type scale_dt = memory::data_type::undef;
    bool invert_scale = true;

    using namespace dnnl::impl;
    auto mask = t.m_mask.get_desc();

    memory::desc *mask_ptr = nullptr;

    switch (p.mask) {
        case mask_type::no_mask:
        case mask_type::causal_tl:
        case mask_type::causal_br: mask_ptr = nullptr; break;
        case mask_type::oneD:
        case mask_type::twoD: mask_ptr = &mask; break;
    }

    sdpa::primitive_desc sdpa_quantized_pd;
    sdpa sdpa_quantized_p;
    try {
        sdpa_quantized_pd = sdpa::primitive_desc(eng, t.m_query_test.get_desc(),
                p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                      : t.m_key_quantized.get_desc(),
                t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
                t.m_output_quantized.get_desc(), invert_scale, p.kv_head_num,
                to_attn_mask_type(p.mask),
                dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                t.sdpa_attr_quantized, t.sdpa_kq_attr_quantized,
                t.sdpa_vs_attr_quantized);
        sdpa_quantized_p = sdpa(sdpa_quantized_pd);
    } catch (const dnnl::error &e) {
        if (e.status == dnnl_unimplemented)
            GTEST_SKIP() << "Unimplemented: " << e.what();
        else
            throw;
    }

    std::unordered_map<int, memory> s8_args = {{{DNNL_ARG_QUERIES, t.m_query},
            {DNNL_ARG_VALUES, t.m_value_quantized},
            {DNNL_ARG_DST, t.m_output_quantized}}};

    if (p.with_key_transposed) {
        s8_args[DNNL_ARG_KEYS] = t.m_key_t_quantized;
    } else {
        s8_args[DNNL_ARG_KEYS] = t.m_key_quantized;
    }
    if (scale_dt != mdt::undef) { s8_args[DNNL_ARG_SCALE] = t.m_scale; }

    bool k_is_16_bit_float = ((p.kdt == mdt::f16) || (p.kdt == mdt::bf16));
    bool v_is_16_bit_float = ((p.vdt == mdt::f16) || (p.vdt == mdt::bf16));
    if (!k_is_16_bit_float && p.qtype != quantize_type::no_quantization) {
        if (p.ksdt != mdt::undef)
            s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
        if (p.kzpdt != mdt::undef)
            s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
    }
    if (!v_is_16_bit_float && p.qtype != quantize_type::no_quantization) {
        if (p.vsdt != mdt::undef)
            s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = t.m_value_scales;
        if (p.vzpdt != mdt::undef)
            s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES] = t.m_value_zp;
    }
    if (mask_ptr) { s8_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    std::unordered_map<int, memory> f16_args
            = {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, t.m_key},
                    {DNNL_ARG_VALUES, t.m_value}, {DNNL_ARG_DST, t.m_output}};
    if (scale_dt != mdt::undef) { f16_args[DNNL_ARG_SCALE] = t.m_scale; }
    if (mask_ptr) { f16_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    auto loop_quantized = [&] { sdpa_quantized_p.execute(strm, s8_args); };
    loop_quantized();

    prim_sdpa_quant(p, t, eng, strm, t.m_query,
            p.with_key_transposed ? t.m_key_t_quantized : t.m_key_quantized,
            t.m_key_scales, t.m_key_zp, scale_dt, t.m_scale, t.m_mask,
            t.m_value_quantized, t.m_value_scales, t.m_value_zp, t.m_output,
            invert_scale, doubled_memory);

#if 0
    if (::getenv("SKIP_CHECK")) return;
#endif
    if (t.m_output.get_desc().get_data_type() == mdt::f16)
        check_memory<float16_t>(t.m_output, t.m_output_quantized, strm);
    else if (t.m_output.get_desc().get_data_type() == mdt::bf16)
        check_memory<bfloat16_t>(t.m_output, t.m_output_quantized, strm);

#if 0
    for (auto &kv : hist) {
        for (auto &kv2 : kv.second) {
            printf("hist[%d][%d] = %d\n", kv.first, kv2.first, kv2.second);
        }
    }
#endif
}
std::vector<std::chrono::microseconds> timeit(
        const std::function<void()> &func, dnnl::stream &str, int iterations) {
    using namespace std::chrono;
    func();
    func();
    std::vector<std::chrono::microseconds> times;
    for (int j = 0; j < 5; j++) {
        auto e = steady_clock::now();
        str.wait();
        auto s = steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        str.wait();
        e = steady_clock::now();
        times.push_back(std::chrono::duration_cast<microseconds>(e - s));
    }
    return times;
}

template <typename O, typename I>
O magnitude_cast(I input) {
    using ratio = std::ratio_divide<typename I::ratio, typename O::ratio>;
    return input.value * ratio::num / ratio::den;
}

template <class Unit = std::ratio<1>>
class byte_t {
public:
    using ratio = Unit;
    float value;
    byte_t(float v) : value(v) {}

    byte_t(memory::data_type dt)
        : value(dnnl_data_type_size((dnnl_data_type_t)dt)
                / ((dt == mdt::s4 || dt == mdt::u4) ? 2 : 1)) {}

    template <typename OR>
    byte_t(byte_t<OR> o) : value(magnitude_cast<Unit>(o).value) {}

    operator float() { return value; }
};

template <class Unit = std::ratio<1>>
class num_ops_t {
public:
    using ratio = Unit;
    float value;
    num_ops_t(float v) : value(v) {}

    template <typename OR>
    num_ops_t(num_ops_t<OR> o) : value(magnitude_cast<Unit>(o).value) {}

    operator float() { return value; }
};

using kilobyte = byte_t<std::ratio<1024>>;
using megabyte = byte_t<std::ratio<1024 * 1024, 1>>;
using gigabyte = byte_t<std::ratio<1024 * 1024 * 1024, 1>>;

using kiloops = num_ops_t<std::ratio<1000>>;
using megaops = num_ops_t<std::ratio<1000 * 1000, 1>>;
using gigaops = num_ops_t<std::ratio<1000 * 1000 * 1000, 1>>;

template <typename BYTES, typename TIME>
float bandwidth(BYTES bytes, TIME duration) {
    return (bytes.value
            / std::chrono::duration_cast<std::chrono::duration<float>>(duration)
                      .count());
}

template <typename OPS, typename TIME>
float compute(OPS ops, TIME duration) {
    return (ops.value
            / std::chrono::duration_cast<std::chrono::duration<float>>(duration)
                      .count());
}

static std::once_flag header_flag;

GPU_TEST_P(sdpa_test_t, perf) {
    memory::data_type scale_dt = t.m_query_test.get_desc().get_data_type();
    //memory::data_type scale_dt = memory::data_type::undef;
    bool invert_scale = true;

    using namespace dnnl::impl;
    auto mask = t.m_mask.get_desc();

    memory::desc *mask_ptr = nullptr;

    switch (p.mask) {
        case mask_type::no_mask:
        case mask_type::causal_tl:
        case mask_type::causal_br: mask_ptr = nullptr; break;
        case mask_type::oneD:
        case mask_type::twoD: mask_ptr = &mask; break;
    }

    sdpa::primitive_desc sdpa_quantized_pd;
    sdpa sdpa_quantized_p;
    try {
        sdpa_quantized_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
                p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                      : t.m_key_quantized.get_desc(),
                t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
                t.m_output_quantized.get_desc(), invert_scale, p.kv_head_num,
                to_attn_mask_type(p.mask),
                alg_kind::softmax_accurate_inf_as_zero, t.sdpa_attr_quantized,
                t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);
        sdpa_quantized_p = sdpa(sdpa_quantized_pd);
    } catch (const dnnl::error &e) {
        if (e.status == dnnl_unimplemented)
            GTEST_SKIP() << "Unimplemented: " << e.what();
        else
            throw;
    }

    auto sdpaf16_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
            p.with_key_transposed ? t.m_key_t.get_desc() : t.m_key.get_desc(),
            t.m_value.get_desc(), mask_ptr, scale_dt, t.m_output.get_desc(),
            invert_scale, p.kv_head_num, to_attn_mask_type(p.mask),
            alg_kind::softmax_accurate_inf_as_zero, t.sdpa_attr);
    auto sdpaf16_p = sdpa(sdpaf16_pd);

    std::unordered_map<int, memory> s8_args = {{{DNNL_ARG_QUERIES, t.m_query},
            {DNNL_ARG_VALUES, t.m_value_quantized},
            {DNNL_ARG_DST, t.m_output_quantized}}};

    if (p.with_key_transposed) {
        s8_args[DNNL_ARG_KEYS] = t.m_key_t_quantized;
    } else {
        s8_args[DNNL_ARG_KEYS] = t.m_key_quantized;
    }
    if (scale_dt != mdt::undef) { s8_args[DNNL_ARG_SCALE] = t.m_scale; }

    if (p.kdt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
    }
    if (p.vdt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = t.m_value_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES] = t.m_value_zp;
    }
    if (mask_ptr) { s8_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    auto loop_quantized = [&] { sdpa_quantized_p.execute(strm, s8_args); };

    /// Dequantize reorder for key
    memory key_dequantized;
    dnnl::reorder key_dequantize_prim;
    bool dequantize_k = !(p.kdt == mdt::f16 || p.kdt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization;
    if (dequantize_k) {
        std::tie(key_dequantize_prim, key_dequantized)
                = dequantize_prim(eng, mdt::f16,
                        p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                              : t.m_key_quantized.get_desc(),
                        t.kq_mask, t.kq_groups, p.ksdt, p.kzpdt,
                        (p.with_key_transposed ? memory::format_tag::abdc
                                               : memory::format_tag::abcd));
    } else {
        key_dequantized = p.with_key_transposed ? t.m_key_t_quantized
                                                : t.m_key_quantized;
    }

    /// Dequantize reorder for value
    memory value_dequantized;
    dnnl::reorder value_dequantize_prim;
    bool dequantize_v = p.vdt != mdt::f16 && p.vdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization;
    if (dequantize_v) {
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f16, t.m_value_quantized.get_desc(),
                        t.vs_mask, t.vs_groups, p.vsdt, p.vzpdt);
    } else {
        value_dequantized = t.m_value_quantized;
    }

    std::unordered_map<int, memory> f16_args
            = {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, key_dequantized},
                    {DNNL_ARG_VALUES, value_dequantized},
                    {DNNL_ARG_DST, t.m_output_quantized}};
    if (scale_dt != mdt::undef) { f16_args[DNNL_ARG_SCALE] = t.m_scale; }
    if (mask_ptr) { f16_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    //auto loop_sdpa_f16 = [&] { sdpaf16_p.execute(strm, f16_args); };

    int iterations = 20;
    auto quantized_time = timeit(loop_quantized, strm, iterations);
    //auto sdpa_f16_time = timeit(loop_sdpa_f16, strm, iterations);

    using namespace std::chrono;
    auto min_time = [](const std::vector<microseconds> &a) {
        return *std::min_element(a.begin(), a.end());
    };

    auto qtime = min_time(quantized_time) / iterations;

    // total number of bytes of all tensors
    byte_t<> total_bytes = t.m_query.get_desc().get_size()

            + key_dequantized.get_desc().get_size() / 2
            + t.m_key_scales.get_desc().get_size()
            + t.m_key_zp.get_desc().get_size()

            + value_dequantized.get_desc().get_size() / 2
            + t.m_value_scales.get_desc().get_size()
            + t.m_value_zp.get_desc().get_size()

            + t.m_output.get_desc().get_size()
            + (mask_ptr ? t.m_mask.get_desc().get_size() : 0);

    auto mask_slice_elements = 0;
    switch (p.mask) {
        case mask_type::twoD:
            mask_slice_elements = p.seq_len * p.query_num;
            break;
        case mask_type::oneD: mask_slice_elements = p.seq_len; break;
        default: mask_slice_elements = 0; break;
    }

    size_t kv_slice_tensor_elements = (p.head_size * p.seq_len);
    size_t batch_elements = p.mb * std::max(p.head_num, p.kv_head_num);

    // Total number of bytes read by the micro_sdpa kernel. This calculation
    // is different from total_bytes because it expands tensors like masks
    // to match the batches of kvq tensors. Typically this is bigger than
    // total bytes.
    byte_t<> total_bytes_effective
            = (batch_elements
                      * (byte_t<>(p.kdt) * kv_slice_tensor_elements
                              + byte_t<>(p.vdt) * kv_slice_tensor_elements
                              + byte_t<>(p.qdt)
                                      * (2 * p.head_size * p.query_num)
                              + (mask_ptr ? byte_t<>(p.mskdt)
                                                      * mask_slice_elements
                                          : 0)))
            + t.m_key_scales.get_desc().get_size()
            + t.m_key_zp.get_desc().get_size()
            + t.m_value_scales.get_desc().get_size()
            + t.m_value_zp.get_desc().get_size();

    // All flops even for causal mask cases
    num_ops_t<> total_flops = std::max<size_t>(p.kv_head_num, p.head_num) * p.mb
            * (2.f * (2.f * p.head_size * p.seq_len * p.query_num)
                    + (scale_dt != mdt::undef ? (p.seq_len * p.query_num) : 0)
                    + (p.mask != mask_type::no_mask ? (p.seq_len * p.query_num)
                                                    : 0)
                    + (5 * p.seq_len * p.query_num));

    // Ignores softmax/mask/scale and does not count masked out values in causal mask cases
    num_ops_t<> flash_flops
            = (4.f * p.mb * p.head_num * p.seq_len * p.query_num * p.head_size)
            / ((p.mask == mask_type::causal_tl
                       || p.mask == mask_type::causal_br)
                            ? 2.f
                            : 1.f);

    std::call_once(header_flag, print_table_header);
    std::cout << print_row(p) << "|" << qtime.count() << "|"
              << bandwidth(
                         magnitude_cast<gigabyte>(total_bytes_effective), qtime)
              << "/" << bandwidth(magnitude_cast<gigabyte>(total_bytes), qtime)
              << "|" << compute(magnitude_cast<gigaops>(flash_flops), qtime)
              << "/" << compute(magnitude_cast<gigaops>(total_flops), qtime)
              << "|" << std::endl;
}
