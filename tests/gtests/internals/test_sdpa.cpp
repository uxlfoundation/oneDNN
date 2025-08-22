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

std::ostream &operator<<(std::ostream &ss, const mask_type &p) {
    ss << "mask:";
    switch (p) {
        case mask_type::no_mask: ss << "no mask"; break;
        case mask_type::oneD: ss << "1D"; break;
        case mask_type::twoD: ss << "2D"; break;
        case mask_type::causal_br: ss << "causal bottom right"; break;
        case mask_type::causal_tl: ss << "causal top left"; break;
    }
    return ss;
}

struct tag_t {
    dnnl::memory::format_tag t;
    tag_t() : t(dnnl::memory::format_tag::undef) {}
    tag_t(dnnl::memory::format_tag t_) : t(t_) {}
    operator dnnl::memory::format_tag() const { return t; }
    bool operator==(const memory::format_tag &other) const {
        return t == other;
    }
};

std::ostream &operator<<(std::ostream &ss, const tag_t &p) {
    ss << "key tag:";
    switch (p.t) {
        case dnnl::memory::format_tag::abcd: ss << "abcd"; break;
        case dnnl::memory::format_tag::abdc: ss << "abdc"; break;
        default: ss << "undef"; break;
    }
    return ss;
}

bool is_quantized(mdt dt, mdt sdt, mdt zpdt, quantize_type qtype) {
    if (qtype == quantize_type::no_quantization) return false;

    if (dt == mdt::f16 || dt == mdt::bf16 || dt == mdt::f32
            || dt == mdt::f8_e4m3 || dt == mdt::f8_e5m2) {
        return false;
    }
    if (sdt == mdt::undef && zpdt == mdt::undef) return false;
    return true;
}

struct tensor_type_t {
    std::string name;
    mdt dt;
    mdt sdt; // scaled data type
    mdt zpdt; // zero point data type
    tensor_type_t() = default;
    tensor_type_t(const tensor_type_t &other) = default;
    tensor_type_t(tensor_type_t &&other) = default;
    tensor_type_t &operator=(const tensor_type_t &other) = default;
    tensor_type_t &operator=(tensor_type_t &&other) = default;
    tensor_type_t(std::string name_, memory::data_type t,
            memory::data_type st = mdt::undef,
            memory::data_type zpt = mdt::undef)
        : name(std::move(name_)), dt(t), sdt(st), zpdt(zpt) {}
    //operator memory::data_type() const { return dt; }
};

std::ostream &operator<<(std::ostream &ss, const tensor_type_t &p) {
    ss << p.name << ":" << p.dt;
    if (is_quantized(p.dt, p.sdt, p.zpdt, quantize_type::per_token)) {
        ss << "x" << p.sdt;
        ss << "x" << p.zpdt;
    }
    return ss;
}

struct head_group_size_t {
    memory::dim head_size;
    int kgroup_size;
    int vgroup_size;
};

std::ostream &operator<<(
        std::ostream &ss, const head_group_size_t &head_group) {
    ss << "Head Size(D)=" << head_group.head_size;
    if (head_group.head_size != head_group.kgroup_size
            || head_group.head_size != head_group.vgroup_size) {
        ss << " Group Size=";
        if (head_group.kgroup_size == head_group.vgroup_size) {
            ss << head_group.kgroup_size;
        } else {
            ss << "(" << head_group.kgroup_size << "x" << head_group.vgroup_size
               << ")";
        }
    }
    return ss;
}

struct seq_len_size_t {
    memory::dim q;
    memory::dim kv;
};

std::ostream &operator<<(std::ostream &ss, const seq_len_size_t &seq_len) {
    ss << "Sequence Length";
    if (seq_len.q == seq_len.kv) {
        ss << "(K/Q)=" << seq_len.q;
    } else {
        ss << "Q:" << seq_len.q << " K/V:" << seq_len.kv;
    }
    return ss;
}

struct num_heads_t {
    memory::dim q;
    memory::dim kv;
};

std::ostream &operator<<(std::ostream &ss, const num_heads_t &heads) {
    ss << "Number of Heads(N)=";
    if (heads.q == heads.kv) {
        ss << heads.q;
    } else {
        ss << "Q:" << heads.q << " K/V:" << heads.kv;
    }
    return ss;
}

struct mask_config_t {
    mask_type type;
    memory::data_type dt;
};

std::ostream &operator<<(std::ostream &ss, const mask_config_t &m) {
    ss << "Mask =";
    switch (m.type) {
        case mask_type::no_mask: ss << "no mask"; break;
        case mask_type::oneD: ss << "1D:" << m.dt; break;
        case mask_type::twoD: ss << "2D:" << m.dt; break;
        case mask_type::causal_br: ss << "causalbr"; break;
        case mask_type::causal_tl: ss << "causaltl"; break;
    }
    return ss;
}

using sdpa_dims_t_tuple = std::tuple<int, num_heads_t, seq_len_size_t,
        head_group_size_t, tensor_type_t, tensor_type_t, tensor_type_t,
        quantize_type, tag_t, mask_config_t>;

struct sdpa_dims_t {
    memory::dim mb;
    num_heads_t heads;
    seq_len_size_t seq_len;

    head_group_size_t head_group;

    tensor_type_t dt;
    tensor_type_t key;
    tensor_type_t value;

    quantize_type qtype;
    memory::format_tag key_format_tag;
    mask_config_t mask;

    sdpa_dims_t() = default;
    sdpa_dims_t(memory::dim mb_, memory::dim head_num_,
            memory::dim kv_head_num_, memory::dim seq_len_,
            memory::dim query_num_, memory::dim head_size_, int kgroup_size_,
            int vgroup_size_, memory::data_type dt_, memory::data_type kdt_,
            memory::data_type ksdt_, memory::data_type kzpdt_,
            memory::data_type vdt_, memory::data_type vsdt_,
            memory::data_type vzpdt_, memory::data_type mskdt_,
            quantize_type qtype_ = quantize_type::no_quantization,
            dnnl::memory::format_tag key_format_tag_
            = dnnl::memory::format_tag::abcd,
            mask_type mask_ = mask_type::no_mask)
        : mb(mb_)
        , heads {head_num_, kv_head_num_}
        , seq_len {seq_len_, query_num_}
        , head_group {head_size_, kgroup_size_, vgroup_size_}
        , dt("Q", dt_)
        , key("K", kdt_, ksdt_, kzpdt_)
        , value("V", vdt_, vsdt_, vzpdt_)
        , qtype(qtype_)
        , key_format_tag(key_format_tag_)
        , mask {mask_, mskdt_} {}

    sdpa_dims_t(const sdpa_dims_t_tuple &dims)
        : mb(std::get<0>(dims))
        , heads(std::get<1>(dims))
        , seq_len(std::get<2>(dims))
        , head_group(std::get<3>(dims))
        , dt(std::get<4>(dims))
        , key(std::get<5>(dims))
        , value(std::get<6>(dims))
        , qtype(std::get<7>(dims))
        , key_format_tag(std::get<8>(dims))
        , mask(std::get<9>(dims)) {}
};

struct sdpa_tensors_t {
    memory m_query, m_scale, m_mask, m_output;
    memory m_key_quantized, m_value_quantized, m_output_quantized;

    memory m_key_scales, m_key_zp, m_value_scales, m_value_zp;
    dnnl::primitive_attr sdpa_attr_quantized, sdpa_kq_attr_quantized,
            sdpa_vs_attr_quantized;

    int kq_mask, vs_mask;
    memory::dims kq_groups, vs_groups;
};

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t &p) {
    ss << "mb" << p.mb;
    if (p.heads.kv != p.heads.q) { ss << "_KVN" << p.heads.kv; }
    ss << "_N" << p.heads.q;
    ss << "_D" << p.head_group.head_size;
    if (p.head_group.head_size != p.head_group.kgroup_size
            || p.head_group.head_size != p.head_group.vgroup_size) {
        ss << "_" << p.head_group.kgroup_size << "x";
        ss << p.head_group.vgroup_size;
    }
    if (p.key_format_tag == memory::format_tag::abdc)
        ss << "_T";
    else
        ss << "_";
    ss << "K" << p.seq_len.kv;
    ss << "_Q" << p.seq_len.q;
    ss << "_Qdt_" << p.dt.dt;
    ss << "_Kdt_" << p.key.dt;
    if (is_quantized(p.key.dt, p.key.sdt, p.key.zpdt, p.qtype)) {
        ss << "x" << p.key.sdt;
        ss << "x" << p.key.zpdt;
    }
    ss << "_Vdt_" << p.value.dt;
    if (is_quantized(p.value.dt, p.value.sdt, p.value.zpdt, p.qtype)) {
        ss << "x" << p.value.sdt;
        ss << "x" << p.value.zpdt;
    }
    switch (p.mask.type) {
        case mask_type::no_mask: ss << "_no_mask"; break;
        case mask_type::oneD: ss << "_mask1D_" << p.mask.dt; break;
        case mask_type::twoD: ss << "_mask2D_" << p.mask.dt; break;
        case mask_type::causal_br: ss << "_maskcausalbr"; break;
        case mask_type::causal_tl: ss << "_maskcausaltl"; break;
    }
    switch (p.qtype) {
        case quantize_type::no_quantization: ss << "_noq"; break;
        case quantize_type::per_token_with_groups: ss << "_ptwg"; break;
        case quantize_type::per_token: ss << "_pt"; break;
        case quantize_type::per_tensor: ss << "_ptensor"; break;
        case quantize_type::per_tensor1: ss << "_ptensor1"; break;
        case quantize_type::per_tensor3: ss << "_ptensor3"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t_tuple &p) {
    ss << sdpa_dims_t(p);
    return ss;
}

std::string print_to_string(const ::testing::TestParamInfo<sdpa_dims_t> &info) {
    dnnl::impl::stringstream_t ss;
    ss << info.param;
    return ss.str();
}

std::string print_to_string2(
        const ::testing::TestParamInfo<sdpa_dims_t_tuple> &info) {
    dnnl::impl::stringstream_t ss;
    ss << sdpa_dims_t(info.param);
    return ss.str();
}

void print_table_header() {
    std::cout << "| mb | Q Heads | KV Heads |   D |    K  |    Q | Kdt | "
                 "Vdt | "
                 "mask | quant |  time (ns) | BW eff/actual (Gbps) | "
                 "gemm/total FLOPs (GFLOPs) |\n";
}

std::string print_row(const sdpa_dims_t &p) {
    dnnl::impl::stringstream_t ss;

    ss << "|" << p.mb;
    ss << "|" << p.heads.q;
    ss << "|" << p.heads.kv;
    ss << "|" << p.head_group.head_size;
    ss << "|" << p.seq_len.kv;
    ss << "|" << p.seq_len.q;
    ss << "|" << p.key.dt;
    if (!(p.key.dt == mdt::f16 || p.value.dt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.key.sdt;
        ss << "/" << p.key.zpdt;
    }
    ss << "|" << p.value.dt;
    if (!(p.value.dt == mdt::f16 || p.value.dt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.value.sdt;
        ss << "/" << p.value.zpdt;
    }
    ss << "|";
    switch (p.mask.type) {
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
    const auto &dims = desc.get_dims();
    if (dims.empty()) return;
    size_t seq_len = dims[3];
    size_t query_num = dims[2];
    size_t batches = dims[1] * dims[0];
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
    const auto &dims = desc.get_dims();
    if (dims.empty()) return;
    int64_t seq_len = dims[3];
    int64_t query_num = dims[2];
    int64_t batches = dims[1] * dims[0];
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
    if (!ret.empty()) ret[0] *= 2;
    return ret;
}

/// This function creates a large tensor double the size requested by /p desc and
/// fills it with NaN values. It then creates a new memory object backed by
/// the first memory handle but with the size of the original memory descriptor.
///
/// This function allows us to identify situations where the SDPA kernel is
/// accessing data out-of-bounds
memory double_and_resize(const memory::desc &desc, dnnl::engine &eng,
        dnnl::stream &strm, std::vector<dnnl_memory_t> &doubled_memory) {
    memory::dims dims2 = double_mb(desc.get_dims());
    auto desc2 = memory::desc(dims2, desc.get_data_type(), desc.get_strides());

    dnnl_memory_t mem2;
    CHECK(dnnl_memory_create(
            &mem2, desc2.get(), eng.get(), DNNL_MEMORY_ALLOCATE));
    doubled_memory.push_back(mem2);

    void *handle;
    CHECK(dnnl_memory_get_data_handle(mem2, &handle));
    if (desc2.get_size()) {
        void *mapped_ptr = nullptr;
        strm.wait();
        CHECK(dnnl_memory_map_data(mem2, &mapped_ptr));
        memset(mapped_ptr, 0xFF, desc2.get_size());
        CHECK(dnnl_memory_unmap_data(mem2, mapped_ptr));
        strm.wait();
    }

    auto out = memory(desc, eng, handle);
    return out;
}

sdpa_tensors_t get_descriptors(dnnl::engine &eng, dnnl::stream &strm,
        const sdpa_dims_t &p, std::vector<dnnl_memory_t> &doubled_memory) {
    sdpa_tensors_t out;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz
            = {p.mb, p.heads.q, p.seq_len.q, p.head_group.head_size};
    const memory::dims k_sz
            = {p.mb, p.heads.kv, p.head_group.head_size, p.seq_len.kv};
    const memory::dims k_stride
            = {p.mb, p.heads.kv, p.head_group.head_size, p.seq_len.kv * 2};
    const memory::dims k_t_stride
            = {p.mb, p.heads.kv, p.seq_len.kv * 2, p.head_group.head_size};
    const memory::dims v_sz
            = {p.mb, p.heads.kv, p.seq_len.kv, p.head_group.head_size};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims key_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {k_sz[0], k_sz[1],
                        k_sz[2] / p.head_group.kgroup_size, k_sz[3]};
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
                return memory::dims {v_sz[0], v_sz[1], v_sz[2],
                        v_sz[3] / p.head_group.vgroup_size};
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
    switch (p.mask.type) {
        case mask_type::no_mask: mask_sz = {};
        case mask_type::oneD: mask_sz = {1, 1, 1, p.seq_len.kv}; break;
        case mask_type::causal_br:
        case mask_type::causal_tl:
        case mask_type::twoD:
            mask_sz = {1, 1, p.seq_len.q, p.seq_len.kv};
            break;
    }

    auto ksdt = p.key.sdt == mdt::undef ? p.key.dt : p.key.sdt;
    auto kzpdt = p.key.zpdt == mdt::undef ? mdt::s8 : p.key.zpdt;
    auto vsdt = p.value.sdt == mdt::undef ? p.value.dt : p.value.sdt;
    auto vzpdt = p.value.zpdt == mdt::undef ? mdt::s8 : p.value.zpdt;

    dnnl::memory::format_tag abcd = dnnl::memory::format_tag::abcd;
    dnnl::memory::format_tag abdc = dnnl::memory::format_tag::abdc;
    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    // clang-format off
    auto query_md            = memory::desc(q_sz,          p.dt.dt,      abcd);
    auto key_md              = memory::desc(k_sz,          p.dt.dt,       abcd);
    auto value_md            = memory::desc(v_sz,          p.dt.dt,       abcd);
    auto scale_md            = memory::desc(scale_sz,      p.dt.dt,      abcd);

    auto query_test_md       = memory::desc(q_sz,          p.dt.dt,      abcd);

    auto key_quantized_md    = memory::desc(k_sz,          p.key.dt,   p.key_format_tag);
    auto key_scales_md       = memory::desc(key_scales_sz, ksdt,       abcd);
    auto key_zp_md           = memory::desc(key_scales_sz, kzpdt,      abcd);

    auto val_quantized_md    = memory::desc(v_sz,          p.value.dt, abcd);
    auto val_t_quantized_md  = memory::desc(v_sz,          p.value.dt, abdc);
    auto val_scales_md       = memory::desc(val_scales_sz, vsdt,      abcd);
    auto val_zp_md           = memory::desc(val_scales_sz, vzpdt,     abcd);


    auto mask_md             = memory::desc(mask_sz,       p.mask.dt != mdt::undef ? p.mask.dt : p.dt.dt,   abcd);
    auto output_md           = memory::desc(q_sz,          p.dt.dt,     abcd);
    auto output_quantized_md = memory::desc(q_sz,          p.dt.dt,     abcd);
    // clang-format on

    // Create memory objects
    out.m_query = double_and_resize(query_md, eng, strm, doubled_memory);
    out.m_scale = double_and_resize(scale_md, eng, strm, doubled_memory);
    out.m_key_quantized
            = double_and_resize(key_quantized_md, eng, strm, doubled_memory);
    out.m_key_scales
            = double_and_resize(key_scales_md, eng, strm, doubled_memory);
    out.m_key_zp = double_and_resize(key_zp_md, eng, strm, doubled_memory);
    out.m_value_quantized
            = double_and_resize(val_quantized_md, eng, strm, doubled_memory);
    out.m_value_scales
            = double_and_resize(val_scales_md, eng, strm, doubled_memory);
    out.m_value_zp = double_and_resize(val_zp_md, eng, strm, doubled_memory);
    out.m_mask = double_and_resize(mask_md, eng, strm, doubled_memory);
    out.m_output = double_and_resize(output_md, eng, strm, doubled_memory);
    out.m_output_quantized
            = double_and_resize(output_quantized_md, eng, strm, doubled_memory);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz), 0.f);
    std::vector<float> scale_data(
            product(scale_sz), std::sqrt(p.head_group.head_size));
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

    out.sdpa_attr_quantized.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    out.kq_mask = 0;
    out.vs_mask = 0;
    out.kq_groups = {};
    out.vs_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            out.kq_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.kq_groups = {p.head_group.kgroup_size, 1};
            out.vs_groups = {1, p.head_group.vgroup_size};
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
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.sdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.key.sdt);
        }

        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.sdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.value.sdt);
        }

        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.key.zpdt);
        }

        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.value.zpdt);
        }
    }

    fill_random(query_data, query_md);
    fill_random_quantized(key_quantized_data, key_quantized_md,
            (p.key.dt == mdt::u4 || p.key.dt == mdt::u8));
    fill_random_quantized(val_quantized_data, val_quantized_md,
            (p.value.dt == mdt::u4 || p.value.dt == mdt::u8));
    if (p.qtype != quantize_type::no_quantization) {
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.sdt != mdt::undef) {
            fill_random_scales(key_scale_data, key_scales_md);
        } else {
            fill_value(key_scale_data, key_scales_md, 1.f);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.sdt != mdt::undef) {
            fill_random_scales(val_scale_data, val_scales_md);
        } else {
            fill_value(val_scale_data, val_scales_md, 1.f);
        }
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_signed, key_zp_md);
        } else {
            fill_value(key_zp_data_signed, key_zp_md, 0);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_signed, val_zp_md);
        } else {
            fill_value(val_zp_data_signed, val_zp_md, 0);
        }
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_unsigned, key_zp_md);
        } else {
            fill_value(key_zp_data_unsigned, key_zp_md, 0U);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_unsigned, val_zp_md);
        } else {
            fill_value(val_zp_data_unsigned, val_zp_md, 0U);
        }
    }

    if (p.mask.type == mask_type::causal_br
            || p.mask.type == mask_type::causal_tl) {
        fill_causal_mask(mask_data, mask_md, p.mask.type);
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
    auto d = p.head_group.head_size;
    auto k = p.seq_len.kv;
    auto q = p.seq_len.q;

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
        mask_data[mr * p.seq_len.kv + mc] = -999;
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

    int group_size = p.head_group.kgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = k_sz[0] * k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = k_sz[2] * k_sz[3];
    }

    std::vector<float> key_data;
    if (p.key.zpdt == mdt::s4 || p.key.zpdt == mdt::s8) {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_signed, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    } else {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_unsigned, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    }

    group_size = p.head_group.vgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = v_sz[0] * v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = v_sz[2] * v_sz[3];
    }
    std::vector<float> value_data;
    if (p.value.zpdt == mdt::s4 || p.value.zpdt == mdt::s8) {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_signed, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    } else {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_unsigned, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    }

    if (p.mask.type != mask_type::no_mask)
        write_to_dnnl_memory(mask_data.data(), out.m_mask, eng, strm);
    write_to_dnnl_memory(scale_data.data(), out.m_scale, eng, strm);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), out.m_query, eng, strm);

    write_to_dnnl_memory(
            key_quantized_data.data(), out.m_key_quantized, eng, strm);

    write_to_dnnl_memory(
            val_quantized_data.data(), out.m_value_quantized, eng, strm);
    if (p.key.zpdt == mdt::s4 || p.key.zpdt == mdt::s8) {
        write_to_dnnl_memory(
                key_zp_data_signed.data(), out.m_key_zp, eng, strm);
    } else {
        write_to_dnnl_memory(
                key_zp_data_unsigned.data(), out.m_key_zp, eng, strm);
    }
    if (p.value.zpdt == mdt::s4 || p.value.zpdt == mdt::s8) {
        write_to_dnnl_memory(
                val_zp_data_signed.data(), out.m_value_zp, eng, strm);
    } else {
        write_to_dnnl_memory(
                val_zp_data_unsigned.data(), out.m_value_zp, eng, strm);
    }
    write_to_dnnl_memory(key_scale_data.data(), out.m_key_scales, eng, strm);
    write_to_dnnl_memory(val_scale_data.data(), out.m_value_scales, eng, strm);
    write_to_dnnl_memory(output_data.data(), out.m_output, eng, strm);
    write_to_dnnl_memory(output_data.data(), out.m_output_quantized, eng, strm);

    return out;
}

static std::unique_ptr<dnnl::engine> sdpa_eng;

dnnl::engine get_sdpa_test_engine() {
    return *sdpa_eng;
}

memory as(dnnl::stream &strm, memory &mem, memory::data_type dt) {
    const memory::dims sz = mem.get_desc().get_dims();

    auto md = memory::desc(sz, dt, mem.get_desc().get_strides());
    auto out = memory(md, mem.get_engine());
    dnnl::reorder(mem, out).execute(strm, mem, out);
    return out;
}

memory reshape(dnnl::stream &strm, memory &mem, const memory::desc &md) {
    auto out = memory(md, mem.get_engine());
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
    return out;
}

std::pair<dnnl::reorder, memory> dequantize_prim(const engine &eng, mdt dt,
        const memory::desc &desc, int mask, const memory::dims &groups, mdt sdt,
        mdt zpdt,
        dnnl::memory::format_tag tag = dnnl::memory::format_tag::abcd) {
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

    memory dequantized_mem = memory(
            {desc.get_dims(), dt, dnnl::memory::format_tag::abcd}, eng);
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

    if (scale_dt != mdt::undef) {
        scale_f32 = reshape(strm, scale_f32,
                {{1, 1, 1, 1, 1}, mdt::f32, memory::format_tag::abcde});
        if (invert_scale)
            bmm1_po.append_binary(algorithm::binary_div, scale_f32.get_desc());
        else
            bmm1_po.append_binary(algorithm::binary_mul, scale_f32.get_desc());
    }
    if (p.mask.type != mask_type::no_mask) {
        mask_f32 = reshape(strm, mask_f32,
                {{mask_sz[0], 1, 1, mask_sz[2], mask_sz[3]}, mdt::f32,
                        memory::format_tag::abcde});
        bmm1_po.append_binary(algorithm::binary_add, mask_f32.get_desc());
    }

    bmm1_attr.set_post_ops(bmm1_po);

    int head_kv_group_size = 0;
    int head_q_group_size = 0;
    int head_group_batches = 0;
    if (p.heads.kv == p.heads.q) {
        head_kv_group_size = p.heads.kv;
        head_q_group_size = p.heads.q;
        head_group_batches = 1;
    } else {
        head_kv_group_size = 1;
        head_q_group_size = p.heads.q / p.heads.kv;
        head_group_batches = p.heads.kv;
    }

    auto original_k_sz = key.get_desc().get_dims();
    const memory::dims k_sz {p.mb, head_group_batches, head_kv_group_size,
            original_k_sz[2], original_k_sz[3]};
    const memory::dims v_sz {p.mb, head_group_batches, head_kv_group_size,
            p.seq_len.kv, p.head_group.head_size};
    const memory::dims q_sz {p.mb, head_group_batches, head_q_group_size,
            p.seq_len.q, p.head_group.head_size};
    memory::desc grouped_key_md(k_sz, p.dt.dt, memory::format_tag::abcde);
    memory::desc grouped_value_md(v_sz, mdt::f32, memory::format_tag::abcde);
    memory::desc grouped_query_md(q_sz, p.dt.dt, memory::format_tag::abcde);

    memory key_dequantized;
    if ((key.get_desc().get_data_type() != mdt::f16
                && key.get_desc().get_data_type() != mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {

        dnnl::reorder key_dequantize_prim;
        std::tie(key_dequantize_prim, key_dequantized)
                = dequantize_prim(eng, p.dt.dt, key.get_desc(), t.kq_mask,
                        t.kq_groups, p.key.sdt, p.key.zpdt);

        std::unordered_map<int, memory> key_dequantize_args = {
                {DNNL_ARG_FROM, key},
                {DNNL_ARG_TO, key_dequantized},
        };
        if (p.key.sdt != mdt::undef) {
            key_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = key_scales;
        }
        if (p.key.zpdt != mdt::undef)
            key_dequantize_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM]
                    = key_zp;
        key_dequantize_prim.execute(strm, key_dequantize_args);
        key_dequantized = reshape(strm, key_dequantized, grouped_key_md);
    } else {
        auto keytmp = as(strm, key, p.dt.dt);
        grouped_key_md = p.key_format_tag == memory::format_tag::abcd
                ? memory::desc(k_sz, p.dt.dt, memory::format_tag::abcde)
                : memory::desc(k_sz, p.dt.dt, memory::format_tag::abced);

        key_dequantized = reshape(strm, keytmp, grouped_key_md);
    }

    memory value_dequantized;
    if (value.get_desc().get_data_type() != mdt::f16
            && value.get_desc().get_data_type() != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        dnnl::reorder value_dequantize_prim;
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f32, value.get_desc(), t.vs_mask,
                        t.vs_groups, p.value.sdt, p.value.zpdt);

        std::unordered_map<int, memory> value_dequantize_args = {
                {DNNL_ARG_FROM, value},
                {DNNL_ARG_TO, value_dequantized},
        };
        if (p.value.sdt != mdt::undef) {
            value_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = value_scales;
        }
        if (p.value.zpdt != mdt::undef)
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
            p.seq_len.q, p.seq_len.kv};
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
            = double_and_resize(grouped_query_md, eng, strm, doubled_memory);
    auto bmm2_pd = matmul::primitive_desc(
            eng, score_md, grouped_value_md, grouped_query_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    std::unordered_map<int, memory> bmm1_args = {{DNNL_ARG_SRC, grouped_query},
            {DNNL_ARG_WEIGHTS, key_dequantized}, {DNNL_ARG_DST, score}};

    if (scale_dt != mdt::undef) {
        bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                = std::move(scale_f32);
        if (p.mask.type != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = std::move(mask_f32);
        }
    } else {
        if (p.mask.type != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                    = std::move(mask_f32);
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

    strm.wait();
    void *output_ptr_ = (void *)output.map_data();
    void *grouped_output_ptr_ = (void *)grouped_output.map_data();
    memcpy(output_ptr_, grouped_output_ptr_, grouped_query_md.get_size());
    grouped_output.unmap_data(grouped_output_ptr_);
    output.unmap_data(output_ptr_);
    strm.wait();
}

template <typename T>
void check_memory(dnnl::stream &strm, memory &gold, memory &test,
        float max_diff_threshold = 0.03f, float fthreshold = 0.001466) {
    T *mapped_ptr_gold = nullptr;
    T *mapped_ptr_test = nullptr;
    mapped_ptr_gold = (T *)gold.map_data();
    mapped_ptr_test = (T *)test.map_data();
    strm.wait();

    auto dims = gold.get_desc().get_dims();
    auto strides = gold.get_desc().get_strides();

    int mismatches = 0;
    int total = 0;

    float max_diff = std::numeric_limits<float>::min();
    std::map<int, std::map<int, int>> hist;
    const bool verbose = false;
    for_(int l = 0; l < dims[0]; l++)
    for_(int k = 0; k < dims[1]; k++)
    for_(int j = 0; j < dims[2]; j++)
    for (int i = 0; i < dims[3]; i++) {
        auto offset = l * strides[0] + k * strides[1] + j * strides[2]
                + i * strides[3];
        auto o_gold = (float)mapped_ptr_gold[offset];
        auto o_test = (float)mapped_ptr_test[offset];
        total++;

        auto min_val = fmin(o_gold, o_test);
        auto max_val = fmax(o_gold, o_test);
        float abs_diff = abs(max_val - min_val);
        bool is_nan = isnan(o_gold) || isnan(o_test);

        float large_threshold = abs(o_gold) * fthreshold;
        bool is_mismatch = is_nan
                || (abs(o_gold) > 1.f ? abs_diff > large_threshold
                                      : abs_diff > fthreshold);
        if (max_diff < abs_diff) {
            if (verbose) {
                printf("new max(%d,%d,%d,%d): test: %f vs gold: %f diff: "
                       "%f\n",
                        l, k, j, i, o_test, o_gold, abs_diff);
            }
            max_diff = abs_diff;
        }
        if (is_mismatch) {
            hist[0][l]++;
            hist[1][k]++;
            hist[2][j]++;
            hist[3][i]++;
        }
        if (is_mismatch && mismatches++ < 32) {
            if (verbose)
                printf("Mismatch at (%d,%d,%d,%d): test %f "
                       "vs. gold %f (diff: %f thresh: %f)\n",
                        l, k, j, i, o_test, o_gold, abs_diff,
                        (abs(o_gold) > 1.f ? large_threshold : fthreshold));
        }
    }

    gold.unmap_data(mapped_ptr_gold);
    test.unmap_data(mapped_ptr_test);

    int threshold = total * 0.0006;

    ASSERT_LE(mismatches, threshold) << mismatches << " out of: " << total;
    ASSERT_LE(max_diff, max_diff_threshold);
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

std::vector<std::chrono::nanoseconds> timeit(
        const std::function<void()> &func, dnnl::stream &str, int iterations) {
    using namespace std::chrono;
    func();
    func();
    std::vector<std::chrono::nanoseconds> times;
    for (int j = 0; j < 5; j++) {
        auto e = steady_clock::now();
        str.wait();
        auto s = steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        str.wait();
        e = steady_clock::now();
        times.push_back(std::chrono::duration_cast<nanoseconds>(e - s));
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
                / ((dt == mdt::s4 || dt == mdt::u4) ? 2.f : 1.f)) {}

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

template <typename T>
class sdpa_test_t : public ::testing::TestWithParam<T> {
public:
    // Testing reusable functionality requires shared engine between tests.
    static void SetUpTestSuite() {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "SDPA primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "SDPA primitive tests do not support HIP";
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
        p = sdpa_test_t<T>::GetParam();
        doubled_memory.reserve(30);
        t = get_descriptors(eng, strm, p, doubled_memory);
        scale_dt = t.m_query.get_desc().get_data_type();
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

    void compare() {
        using namespace dnnl::impl;
        auto mask = t.m_mask.get_desc();

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
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
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
                    t.m_output_quantized.get_desc(), invert_scale, p.heads.kv,
                    to_attn_mask_type(p.mask.type),
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

        std::unordered_map<int, memory> sdpa_args
                = {{{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized}}};

        sdpa_args[DNNL_ARG_KEYS] = t.m_key_quantized;

        if (scale_dt != mdt::undef) { sdpa_args[DNNL_ARG_SCALE] = t.m_scale; }

        bool k_is_float = ((p.key.dt == mdt::f16) || (p.key.dt == mdt::bf16));
        bool v_is_float
                = ((p.value.dt == mdt::f16) || (p.value.dt == mdt::bf16));
        if (!k_is_float && p.qtype != quantize_type::no_quantization) {
            if (p.key.sdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS]
                        = t.m_key_scales;
            if (p.key.zpdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS]
                        = t.m_key_zp;
        }
        if (!v_is_float && p.qtype != quantize_type::no_quantization) {
            if (p.value.sdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES]
                        = t.m_value_scales;
            if (p.value.zpdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
                        = t.m_value_zp;
        }
        if (mask_ptr) { sdpa_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

        sdpa_quantized_p.execute(strm, sdpa_args);

        prim_sdpa_quant(p, t, eng, strm, t.m_query, t.m_key_quantized,
                t.m_key_scales, t.m_key_zp, scale_dt, t.m_scale, t.m_mask,
                t.m_value_quantized, t.m_value_scales, t.m_value_zp, t.m_output,
                invert_scale, doubled_memory);

#if 0
    if (::getenv("SKIP_CHECK")) return;
#endif
        float max_diff_threshold = 0.03f;
        float fthreshold = 0.f;
        if (p.dt.dt == mdt::bf16) {
            if (p.key.dt == mdt::s4 || p.value.dt == mdt::s4) {
                fthreshold = 0.0157f;
            } else {
                fthreshold = 0.0079f;
            }
        } else {
            fthreshold = 0.001466f;
        }
        if (p.key.dt == mdt::s4 || p.value.dt == mdt::s4) {
            max_diff_threshold = 0.063f;
        }
        if (t.m_output.get_desc().get_data_type() == mdt::f16)
            check_memory<float16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold);
        else if (t.m_output.get_desc().get_data_type() == mdt::bf16)
            check_memory<bfloat16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold);
        else if (t.m_output.get_desc().get_data_type() == mdt::f32)
            check_memory<float_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold);

#if 0
    for (auto &kv : hist) {
        for (auto &kv2 : kv.second) {
            printf("hist[%d][%d] = %d\n", kv.first, kv2.first, kv2.second);
        }
    }
#endif
    }

    void perf() {
        using namespace dnnl::impl;
        auto mask = t.m_mask.get_desc();

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
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
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
                    t.m_output_quantized.get_desc(), invert_scale, p.heads.kv,
                    to_attn_mask_type(p.mask.type),
                    alg_kind::softmax_accurate_inf_as_zero,
                    t.sdpa_attr_quantized, t.sdpa_kq_attr_quantized,
                    t.sdpa_vs_attr_quantized);
            sdpa_quantized_p = sdpa(sdpa_quantized_pd);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented)
                GTEST_SKIP() << "Unimplemented: " << e.what();
            else
                throw;
        }

        std::unordered_map<int, memory> sdpa_args
                = {{{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized}}};

        sdpa_args[DNNL_ARG_KEYS] = t.m_key_quantized;
        if (scale_dt != mdt::undef) { sdpa_args[DNNL_ARG_SCALE] = t.m_scale; }

        if (p.key.dt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
            sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
            sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
        }
        if (p.value.dt != mdt::f16
                && p.qtype != quantize_type::no_quantization) {
            sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES]
                    = t.m_value_scales;
            sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
                    = t.m_value_zp;
        }
        if (mask_ptr) { sdpa_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

        auto loop_quantized
                = [&] { sdpa_quantized_p.execute(strm, sdpa_args); };

        int iterations = 20;
        auto quantized_time = timeit(loop_quantized, strm, iterations);

        using namespace std::chrono;
        auto min_time = [](const std::vector<nanoseconds> &a) {
            return *std::min_element(a.begin(), a.end());
        };

        auto qtime = min_time(quantized_time) / iterations;

        // total number of bytes of all tensors
        byte_t<> total_bytes = t.m_query.get_desc().get_size()

                + t.m_key_quantized.get_desc().get_size() / 2
                + t.m_key_scales.get_desc().get_size()
                + t.m_key_zp.get_desc().get_size()

                + t.m_value_quantized.get_desc().get_size() / 2
                + t.m_value_scales.get_desc().get_size()
                + t.m_value_zp.get_desc().get_size()

                + t.m_output.get_desc().get_size()
                + (mask_ptr ? t.m_mask.get_desc().get_size() : 0);

        auto mask_slice_elements = 0;
        switch (p.mask.type) {
            case mask_type::twoD:
                mask_slice_elements = p.seq_len.kv * p.seq_len.q;
                break;
            case mask_type::oneD: mask_slice_elements = p.seq_len.kv; break;
            default: mask_slice_elements = 0; break;
        }

        size_t kv_slice_tensor_elements
                = (p.head_group.head_size * p.seq_len.kv);
        size_t batch_elements = p.mb * std::max(p.heads.q, p.heads.kv);

        // Total number of bytes read by the micro_sdpa kernel. This calculation
        // is different from total_bytes because it expands tensors like masks
        // to match the batches of kvq tensors. Typically this is bigger than
        // total bytes.
        byte_t<> total_bytes_effective
                = (batch_elements
                          * (byte_t<>(p.key.dt) * kv_slice_tensor_elements
                                  + byte_t<>(p.value.dt)
                                          * kv_slice_tensor_elements
                                  + byte_t<>(p.dt.dt)
                                          * (2 * p.head_group.head_size
                                                  * p.seq_len.q)
                                  + (mask_ptr ? byte_t<>(p.mask.dt)
                                                          * mask_slice_elements
                                              : 0)))
                + t.m_key_scales.get_desc().get_size()
                + t.m_key_zp.get_desc().get_size()
                + t.m_value_scales.get_desc().get_size()
                + t.m_value_zp.get_desc().get_size();

        // All flops even for causal mask cases
        num_ops_t<> total_flops = std::max<size_t>(p.heads.kv, p.heads.q) * p.mb
                * (2.f
                                * (2.f * p.head_group.head_size * p.seq_len.kv
                                        * p.seq_len.q)
                        + (scale_dt != mdt::undef ? (p.seq_len.kv * p.seq_len.q)
                                                  : 0)
                        + (p.mask.type != mask_type::no_mask
                                        ? (p.seq_len.kv * p.seq_len.q)
                                        : 0)
                        + (5 * p.seq_len.kv * p.seq_len.q));

        // Ignores softmax/mask/scale and does not count masked out values in causal mask cases
        num_ops_t<> flash_flops
                = (4.f * p.mb * p.heads.q * p.seq_len.kv * p.seq_len.q
                          * p.head_group.head_size)
                / ((p.mask.type == mask_type::causal_tl
                           || p.mask.type == mask_type::causal_br)
                                ? 2.f
                                : 1.f);

        std::call_once(header_flag, print_table_header);
        std::cout << print_row(p) << "|" << qtime.count() << "|"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes_effective),
                             qtime)
                  << "/"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes), qtime)
                  << "|" << compute(magnitude_cast<gigaops>(flash_flops), qtime)
                  << "/" << compute(magnitude_cast<gigaops>(total_flops), qtime)
                  << "|" << std::endl;
    }

protected:
    dnnl::engine eng;
    dnnl::stream strm;
    sdpa_dims_t p;
    sdpa_tensors_t t;
    std::vector<dnnl_memory_t> doubled_memory;

    bool invert_scale = true;
    memory::data_type scale_dt;
};

memory::format_tag with_key_transposed = memory::format_tag::abdc;
memory::format_tag no_key_transposed = memory::format_tag::abcd;

using sdpa_test = sdpa_test_t<sdpa_dims_t>;
using sdpa_test_datatypes = sdpa_test_t<sdpa_dims_t_tuple>;

// clang-format off

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s8, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                testing::Values(tensor_type_t("K", mdt::f16), tensor_type_t("K", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8) /*, tensor_type_t("K", mdt::s8, mdt::undef, mdt::s8)*/), // kdt
                testing::Values(tensor_type_t("V", mdt::f16), tensor_type_t("V", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8) /*, tensor_type_t("V", mdt::s8, mdt::undef, mdt::s8) */), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s4, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                testing::Values(tensor_type_t("K", mdt::f16), tensor_type_t("K", mdt::s4, mdt::f16, mdt::undef), tensor_type_t("K", mdt::s4, mdt::f16, mdt::s8)), // kdt
                testing::Values(tensor_type_t("V", mdt::f16), tensor_type_t("V", mdt::s4, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::f16, mdt::s8)), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_bf16_s8, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::bf16)), // dt
                testing::Values(tensor_type_t("K", mdt::bf16) /*, tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8), tensor_type_t("K", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("K", mdt::s8, mdt::undef, mdt::s8)*/), // kdt
                testing::Values(tensor_type_t("V", mdt::bf16) /*, tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8), tensor_type_t("V", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s8, mdt::undef, mdt::s8)*/), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::oneD, mdt::bf16}, mask_config_t {mask_type::twoD, mdt::bf16}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_bf16_s4, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::bf16)), // dt
                testing::Values(tensor_type_t("K", mdt::bf16), tensor_type_t("K", mdt::s4, mdt::bf16, mdt::undef), tensor_type_t("K", mdt::s4, mdt::bf16, mdt::s8)), // kdt
                testing::Values(tensor_type_t("V", mdt::bf16), tensor_type_t("V", mdt::s4, mdt::bf16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::bf16, mdt::s8)), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::oneD, mdt::bf16}, mask_config_t {mask_type::twoD, mdt::bf16}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f32, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}, seq_len_size_t {1024, 1024}, seq_len_size_t {1, 1025}), // seq_len
                testing::Values(head_group_size_t {32, 32, 32}, head_group_size_t {64, 64, 64}, head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::f32)), // dt
                testing::Values(tensor_type_t("K", mdt::f32), tensor_type_t("K", mdt::s8, mdt::f32)), // kdt
                testing::Values(tensor_type_t("V", mdt::f32), tensor_type_t("V", mdt::s8, mdt::f32)), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::twoD, mdt::f32}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(AllMaskTypes, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                testing::Values(head_group_size_t {64, 64, 64}, head_group_size_t {128, 128, 128}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                testing::Values(tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8)), // kdt
                testing::Values(tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8)), // vdt
                testing::Values(quantize_type::per_token), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::no_mask}, mask_config_t {mask_type::causal_tl}, mask_config_t {mask_type::causal_br}, mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f16}) // mask_type
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(GQA, sdpa_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {4, 2}, num_heads_t {8, 2}, num_heads_t {32, 2}, num_heads_t {64, 2}), // hd_num
                testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                testing::Values(tensor_type_t("V", mdt::f16)), // vdt
                testing::Values(quantize_type::no_quantization), // qtype
                testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                testing::Values(mask_config_t {mask_type::no_mask}) // mask_type
                ),
        &print_to_string2);

////llama-2-7b-chat shape: Q [1x32xSEQ_LENx128] KV [1x32xSEQ_LENx128]
////llama-3-8b shape: Q [1x32xSEQ_LENx128] KV [1x8xSEQ_LENx128]
////minicpm-1b-sft shape:  Q [1x24xSEQ_LENx64]  KV [1x8xSEQ_LENx64]
////qwen2-7b shape: Q [1x28xSEQ_LENx128] KV [1x4xSEQ_LENx128]
////phi3-mini-4k-instruct shape: Q [1x32xSEQ_LENx96] KV [1x32xSEQ_LENx96]

INSTANTIATE_TEST_SUITE_P(llama_2_7b_chat,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       kdt,        ksdt,      kzpdt,        vdt,       vsdt,      vzpdt,     mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    32,       32,    384,    384,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    385,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    512,    512,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    513,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1024,   1024,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1025,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2048,   2048,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2049,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(llama_3_8b,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    32,        8,    384,    384,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    386,    386,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    385,      1,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    512,    512,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    513,      1,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1024,   1024,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1025,      1,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2048,   2048,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2049,      1,    128,   128,     128, mdt::f16,  mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,        with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    384,    384,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    385,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    512,    512,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    513,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1024,   1024,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1025,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2048,   2048,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2049,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(minicpm_1b_st,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,     kdt,      ksdt,      kzpdt,      vdt,     vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    24,        8,    384,    384,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    385,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    512,    512,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    513,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1024,   1024,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1025,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2048,   2048,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2049,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(qwen2_7b,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,  vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,    28,        4,    384,    384,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    385,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    512,    512,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    513,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1024,   1024,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1025,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2048,   2048,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2049,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(phi3_mini_4k_instruct,
    sdpa_test,
                               // mb, hd_num,kv_grp_sz,seq_len, qry_num,hd_size,  kg_sz, vgrp_sz,       dt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::oneD },
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,     32,       32,    385,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    512,     512,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    513,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   1024,    1024,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   1025,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   2048,    2048,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   2049,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);
// clang-format on

GPU_TEST_P(sdpa_test, compare) {
    compare();
}

GPU_TEST_P(sdpa_test_datatypes, compare) {
    compare();
}

GPU_TEST_P(sdpa_test, perf) {
    perf();
}

/*
GPU_TEST_P(sdpa_test_datatypes, perf) {
    perf();
}
*/
