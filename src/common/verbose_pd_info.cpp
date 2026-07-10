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

#include "common/verbose_pd_info.hpp"

#include "common/c_types_map.hpp"

#include "common/batch_normalization_pd.hpp"
#include "common/binary_pd.hpp"
#include "common/concat_pd.hpp"
#include "common/convolution_pd.hpp"
#include "common/deconvolution_pd.hpp"
#include "common/eltwise_pd.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/gemm_pd.hpp"
#include "common/group_normalization_pd.hpp"
#include "common/inner_product_pd.hpp"
#include "common/layer_normalization_pd.hpp"
#include "common/lrn_pd.hpp"
#include "common/matmul_pd.hpp"
#include "common/pooling_pd.hpp"
#include "common/prelu_pd.hpp"
#include "common/reduction_pd.hpp"
#include "common/reorder_pd.hpp"
#include "common/resampling_pd.hpp"
#include "common/rnn_pd.hpp"
#include "common/sdpa_pd.hpp"
#include "common/shuffle_pd.hpp"
#include "common/softmax_pd.hpp"
#include "common/sum_pd.hpp"

namespace dnnl {
namespace impl {

std::ostream &operator<<(std::ostream &ss, engine_kind_t eng_kind) {
    ss << dnnl_engine_kind2str(eng_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const engine_t *engine) {
    ss << dnnl_engine_kind2str(engine->kind());
    if (dnnl_engine_get_count(engine->kind()) > 1)
        ss << ":" + std::to_string(engine->index());
    return ss;
}

const char *prim_kind2str(primitive_kind_t prim_kind) {
    switch ((int)prim_kind) {
        case primitive_kind::zero_pad: return "zero_pad";
        default: return dnnl_prim_kind2str(prim_kind);
    }
}

std::ostream &operator<<(std::ostream &ss, primitive_kind_t prim_kind) {
    ss << prim_kind2str(prim_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, prop_kind_t prop_kind) {
    ss << dnnl_prop_kind2str(prop_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, data_type_t data_type) {
    ss << dnnl_dt2str(data_type);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, alg_kind_t alg) {
    ss << dnnl_alg_kind2str(alg);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, format_kind_t format_kind) {
    ss << dnnl_fmt_kind2str(format_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, sparse_encoding_t encoding) {
    ss << dnnl_sparse_encoding2str(encoding);
    return ss;
}

std::string normalization_flags2str(unsigned flags) {
    std::string s;
    if (flags & normalization_flags::use_global_stats) s += "G";
    if (flags & normalization_flags::use_scale) s += "C";
    if (flags & normalization_flags::use_shift) s += "H";
    if (flags & normalization_flags::fuse_norm_relu) s += "R";
    if (flags & normalization_flags::fuse_norm_add_relu) s += "A";
    if (flags & normalization_flags::rms_norm) s += "M";
    return s;
}

std::string rnn_flags2str(unsigned flags) {
    std::string s;
    if (flags & rnn_flags::diff_weights_overwrite) s += "O";
    return s;
}

std::string cublasltfmt2str(const memory_desc_t *md) {
    if (md->format_desc.cublaslt_blocked_desc.cublaslt_format
            == cublaslt_memory_format_t::col32_2r_4r4) {
        return ":col32_2r_4r4";
    }
    return "";
}

std::ostream &operator<<(std::ostream &ss, const memory_extra_desc_t &extra) {
    using namespace memory_extra_flags;

    ss << ":f" << extra.flags;
    if (extra.flags & compensation_conv_s8s8)
        ss << ":s8m" << extra.compensation_mask;
    if (extra.flags & compensation_conv_asymmetric_src)
        ss << ":zpm" << extra.asymm_compensation_mask;
    if (extra.flags & compensation_gpu_conv_asymmetric_src) {
        ss << ":zid" << extra.idhw[0];
        ss << ":zih" << extra.idhw[1];
        ss << ":ziw" << extra.idhw[2];
        ss << ":zod" << extra.odhw[0];
        ss << ":zoh" << extra.odhw[1];
        ss << ":zow" << extra.odhw[2];
        ss << ":zpd" << extra.pdhw[0];
        ss << ":zph" << extra.pdhw[1];
        ss << ":zpw" << extra.pdhw[2];
        ss << ":zdd" << extra.ddhw[0];
        ss << ":zdh" << extra.ddhw[1];
        ss << ":zdw" << extra.ddhw[2];
        ss << ":zs" << extra.dst_size;
    }
    if (extra.flags & scale_adjust && extra.scale_adjust != 1.f)
        ss << ":sa" << extra.scale_adjust;
    return ss;
}

std::string md2fmt_tag_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);

    // Can't report meaningful tag for runtime dimensions.
    if (mdw.has_runtime_strides()) return "*";

    struct sort_key_t {
        uint64_t stride_order;
        dim_t outer_block;
        int idx;
        char dim_char;
    };

    dims_t blocks = {0};
    mdw.compute_blocks(blocks);

    std::vector<sort_key_t> sort_keys(mdw.ndims());
    const auto &pdims = mdw.padded_dims();
    const auto &blk = mdw.blocking_desc();
    for (int i = 0; i < mdw.ndims(); ++i)
        // Assume that any dimension with stride 0 is outer relative to other
        // dimensions. Use (uint64_t)(stride - 1) to sort a stride of 0 highest.
        // Multiple dimensions with stride 0 is ambiguous.
        sort_keys[i] = {(uint64_t)(blk.strides[i] - 1), pdims[i] / blocks[i], i,
                (char)((blocks[i] == 1 ? 'a' : 'A') + i)};

    // Old approach: utils::simultaneous_sort(strides, outer_blocks, dim_chars)
    //   input tag: acdb
    //   dims: 5x8x0x2
    //   strides: 0x1x16x8
    //   output tag: cdba
    //
    // New approach with std::sort and sort keys:
    //   input tag: acdb
    //   dims: 5x8x0x2
    //   "stride orders": (BIG NUMBER)x0x15x7
    //   output tag: acdb
    std::sort(sort_keys.begin(), sort_keys.end(),
            [](const sort_key_t &left, const sort_key_t &right) {
        if (left.stride_order < right.stride_order) return false;
        if (left.stride_order == right.stride_order) {
            // WLOG, we can assume a dimension of size 1 has the same
            // stride as the next outermost dimension. Sort the one with
            // the non-unit outer block as the outer dimension. Multiple
            // dimensions of size 1 with the same stride is ambiguous.
            if (left.outer_block < right.outer_block) return false;
            if (left.outer_block == right.outer_block)
                // Sort 1x1x... outer blocks to (arbitrarily) list them
                // in alphabetical order.
                return left.idx < right.idx;
        }
        return true;
    });

    char dim_chars[DNNL_MAX_NDIMS + 1];
    for (int i = 0; i < mdw.ndims(); ++i)
        dim_chars[i] = sort_keys[i].dim_char;
    dim_chars[mdw.ndims()] = '\0';

    std::string s(dim_chars);

    if (!mdw.is_plain()) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            char c = ('a' + (char)blk.inner_idxs[iblk]);
            s += (std::to_string(blk.inner_blks[iblk]) + c);
        }
    }

    return s;
}

std::string md2fmt_strides_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);
    std::string s;

    // Print strides if non-dense descriptor with defined dims/strides was
    // provided.
    if (mdw.has_runtime_dims_or_strides() || mdw.is_dense(true)) return s;

    // Note: there's no API to create a memory desc with strides and blocks
    // together, thus, this non-plain md will dump strides for info purpose.
    s += md2dim_str(md, dims_type_t::strides);
    return s;
}

// Forms a format string for a given memory descriptor.
//
// There are two formats:
// - dense: defined as: 'dt:[a|p|o|0]:fmt_kind:fmt:strides:extra'.
// - sparse: defined as: 'dt:[a|p|o|0]:fmt_kind:encoding:extra'.
// Here:
//  - dt       -- data type
//  - a        -- indicates memory desc was created with fmt_kind `any`.
//  - p        -- indicates there is non-trivial padding
//  - o        -- indicates there is non-trivial padding offset
//  - 0        -- indicates there is non-trivial offset0
//  - fmt_kind -- format kind (blocked, wino, etc...)
//  - encoding -- [sparse_desc only] sparse encoding (csr, etc...)
//  - fmt      -- [blocking_desc only] extended format string
//  - strides  -- [blocking_desc only] non-dense strides string (dims style)
//  - extra    -- shows extra fields (underspecified)
//
// Note: `user_format` is an information that is not available in memory descs
// from `pd` since those are initialized by implementations. The knowledge about
// original user format specified is kept in pd->desc()->xxx_desc and this is
// the info provided to this call.
// On the other hand, just a user memory descriptor can't be passed because it
// is not initialized b the library, and format information will be missed.
std::string md2fmt_str(
        const char *name, const memory_desc_t *md, format_kind_t user_format) {
    stringstream_t ss;
    ss << name << ":";
    if (!md || types::is_zero_md(md)) {
        ss << data_type::undef << "::" << format_kind::undef << ":::";
        return ss.str();
    }

    memory_desc_wrapper mdw(md);
    ss << mdw.data_type() << ":";

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < mdw.ndims(); ++d) {
        if (mdw.dims()[d] != mdw.padded_dims()[d]) padded_dims = true;
        if (mdw.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = mdw.offset0();
    ss << (user_format == format_kind::any ? "a" : "");
    ss << (padded_dims ? "p" : "");
    ss << (padded_offsets ? "o" : "");
    ss << (offset0 ? "0" : "");
    ss << ":" << mdw.format_kind();

    // Cast is required to pass through compiler error:
    // error: case value ‘256’ not in enumerated type
    // ‘dnnl::impl::format_kind_t’ {aka ‘dnnl_format_kind_t’}
    switch (static_cast<int>(mdw.format_kind())) {
        case format_kind::blocked:
            ss << ":" << md2fmt_tag_str(md) << ":" << md2fmt_strides_str(md);
            break;
        case format_kind::cublaslt_blocked: ss << cublasltfmt2str(md); break;
        case format_kind::wino:
        case format_kind::rnn_packed:
        case format_kind::opaque: ss << "::"; break;
        case format_kind::sparse: ss << ":" << mdw.encoding() << ":"; break;
        case format_kind::any: ss << ":any:"; break;
        default:
            assert(!"unsupported format_kind");
            ss << "::";
            break;
    }

    ss << mdw.extra();

    return ss.str();
}

// Puts memory_desc information into stream without dimensions
std::ostream &operator<<(std::ostream &ss, const memory_desc_t *md) {
    assert(!"unexpected call to the operator<<");
    return ss;
}

template <typename T>
static std::string get_val_str(T val) {
    static_assert(
            std::is_arithmetic<T>::value, "T must be an arithmetic type.");
    if (is_runtime_value(val)) return std::string("*");
    return std::to_string(val);
}

// Returns string with dimensions from a given memory descriptor.
// The format is defined as: dim0xdim1x...xdimN, with RT values signed as `*`.
std::string md2dim_str(const memory_desc_t *md, dims_type_t dims_type) {
    if (md == nullptr || md->ndims == 0) return "";

    memory_desc_wrapper mdw(md);
    std::string s;

    assert(dims_type == dims_type_t::dims || dims_type == dims_type_t::strides);
    const auto &dims_obj
            = dims_type == dims_type_t::dims ? mdw.dims() : mdw.strides();

    s += get_val_str(dims_obj[0]);
    for (int d = 1; d < mdw.ndims(); ++d)
        s += ("x" + get_val_str(dims_obj[d]));

    return s;
}

// Returns string with descriptor style from memory_desc.
std::string md2desc_str(const memory_desc_t *md) {
    const auto dims = md->dims;
    std::string s;
    if (md->ndims >= 6) return md2dim_str(md);

    if (md->ndims == 1) {
        s += "x" + std::to_string(dims[0]);
        return s;
    }

    s += "mb" + std::to_string(dims[0]) + "ic" + std::to_string(dims[1]);
    if (md->ndims >= 5) s += "id" + std::to_string(dims[md->ndims - 3]);
    if (md->ndims >= 4) s += "ih" + std::to_string(dims[md->ndims - 2]);
    if (md->ndims >= 3) s += "iw" + std::to_string(dims[md->ndims - 1]);
    return s;
}

std::ostream &operator<<(
        std::ostream &ss, const rnn_create_time_scales_t &rnn_scales) {
    ss << rnn_scales.mask_;
    const float val = rnn_scales.scales_[0];
    // Can't use scientific flags since it breaks parsing on converter and
    // benchdnn side.
    if (rnn_scales.mask_ == 0 || is_runtime_value(val))
        ss << ":" << get_val_str(val);
    return ss;
}

namespace {
int get_runtime_mask(const memory_desc_t *md) {
    int mask = 0;
    for (int d = md->ndims - 1; d >= 0; --d) {
        mask += is_runtime_value(md->dims[d]) ? 1 << d : 0;
    }
    return mask;
}

int get_arg_index(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return arg - DNNL_ARG_MULTIPLE_SRC;
    switch (arg) {
        case DNNL_ARG_SRC_0: return 0;
        case DNNL_ARG_SRC_1: return 1;
        case DNNL_ARG_SRC_2: return 2;
        default: return -1;
    }
    return -1;
}

std::string get_arg(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return "msrc";

    std::string s;
    switch (arg) {
        case DNNL_ARG_SRC: // DNNL_ARG_SRC_0
        case DNNL_ARG_SRC_1:
        case DNNL_ARG_SRC_2: s = "src"; break;
        case DNNL_ARG_DST: s = "dst"; break;
        case DNNL_ARG_WEIGHTS: // DNNL_ARG_WEIGHTS_0
        case DNNL_ARG_WEIGHTS_1:
        case DNNL_ARG_WEIGHTS_2: s = "wei"; break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST:
            s = "attr_post_op_dw_dst";
            break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
            s = "attr_post_op_dw_wei";
            break;
        default: assert(!"unsupported arg"); s = "unsupported arg";
    }
    return s;
}
} // namespace

std::string arg2str(int arg) {
    std::string s = get_arg(arg);
    const int idx = get_arg_index(arg);
    if (idx != -1) s += std::to_string(idx);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const primitive_attr_t *attr) {
    struct {
        const char *operator()() {
            current[0] = next;
            next = ' ';
            return current;
        }

    private:
        char current[2] = {};
        char next = 0;
    } field_delim;

    std::string empty_delim, attr_delim = "+";

    // scratchpad and fpmath mode are not a part of
    // has_default_values(). Check them first.
    const scratchpad_mode_t &spm = attr->scratchpad_mode_;
    if (spm != scratchpad_mode_t::dnnl_scratchpad_mode_library) {
        ss << field_delim()
           << "attr-scratchpad:" << dnnl_scratchpad_mode2str(spm);
    }
    const fpmath_t &fpm = attr->fpmath_;
    if (fpm.mode_ != fpmath_mode_t::dnnl_fpmath_mode_strict
            || fpm.apply_to_int_) {
        ss << field_delim()
           << "attr-fpmath:" << dnnl_fpmath_mode2str(fpm.mode_);
        if (fpm.apply_to_int_) ss << ":true";
    }

    const accumulation_mode_t &am = attr->acc_mode_;
    if (am != accumulation_mode::strict) {
        ss << field_delim()
           << "attr-acc-mode:" << dnnl_accumulation_mode2str(am);
    }

    const auto &rm = attr->rounding_mode_;
    if (!rm.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-rounding-mode:";
        for (const auto &e : rm.rounding_modes_map_) {
            // TODO: add support for diff tensors in arg2str when
            // support is added
            if (!rm.has_default_values(e.first))
                ss << delim << arg2str(e.first) << ":"
                   << dnnl_rounding_mode2str(e.second);
            delim = attr_delim;
        }
    }

    const bool deterministic = attr->deterministic_;
    if (deterministic) {
        ss << field_delim() << "attr-deterministic:" << deterministic;
    }

    // Fast exit if rest attributes were not specified.
    if (attr->has_default_values()) return ss;

    const scales_t &scales = attr->scales_;
    if (!scales.has_default_values()) {
        ss << field_delim() << "attr-scales:" << scales.get_verbose();
    }

    const zero_points_t &zero_points = attr->zero_points_;
    if (!zero_points.has_default_values()) {
        ss << field_delim() << "attr-zero-points:" << zero_points.get_verbose();
    }

    const precomputed_reductions_t &pr = attr->precomputed_reductions_;
    if (!pr.has_default_values()) {
        ss << field_delim()
           << "attr-precomputed-reductions:" << pr.get_verbose();
    }

    const post_ops_t &po = attr->post_ops_;
    if (!po.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-post-ops:";
        for (int i = 0; i < po.len(); ++i) {
            const post_ops_t::entry_t &e = po.entry_[i];
            switch (e.kind) {
                case primitive_kind::sum: {
                    const auto &s = e.sum;
                    ss << delim << "sum";
                    if (s.scale != 1.f || s.zero_point != 0
                            || s.dt != data_type::undef)
                        ss << ":" << s.scale;
                    if (s.zero_point != 0 || s.dt != data_type::undef)
                        ss << ":" << s.zero_point;
                    if (s.dt != data_type::undef) ss << ":" << s.dt;
                } break;
                case primitive_kind::convolution: {
                    using namespace data_type;
                    const auto &c = e.depthwise_conv;
                    ss << delim << "dw:k" << c.kernel << "s" << c.stride << "p"
                       << c.padding;
                    if (c.wei_dt == s8 || c.dst_dt != f32)
                        ss << ":" << c.dst_dt;
                } break;
                case primitive_kind::eltwise: {
                    const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                    ss << delim << ew.alg;
                    if (ew.alpha != 0.f || ew.beta != 0.f || ew.scale != 1.f)
                        ss << ":" << ew.alpha;
                    if (ew.beta != 0.f || ew.scale != 1.f) ss << ":" << ew.beta;
                    if (ew.scale != 1.f) ss << ":" << ew.scale;
                } break;
                case primitive_kind::binary: {
                    const post_ops_t::entry_t::binary_t &eb = e.binary;
                    const auto &md = eb.user_src1_desc;
                    int mask = 0;
                    for (int d = 0; d < md.ndims; ++d)
                        mask += md.dims[d] != 1 ? (1 << d) : 0;
                    ss << delim << eb.alg << ":" << md.data_type << ":" << mask;
                    const memory_desc_wrapper mdw(md);
                    switch (mdw.format_kind()) {
                        case format_kind::blocked:
                            if (!mdw.count_non_unit_dims(1)) {
                                ss << ":" << md2fmt_tag_str(&eb.src1_desc);
                                const auto &strides_str
                                        = md2fmt_strides_str(&eb.src1_desc);
                                if (!strides_str.empty())
                                    ss << ":" << strides_str;
                            }
                            break;
                        case format_kind::sparse:
                            if (mdw.is_grouped_desc()) {
                                ss << ":grouped";
                                break;
                            }
                            assert(!"unsupported sparse encoding");
                            break;
                        case format_kind::any: ss << ":any"; break;
                        default: assert(!"unsupported format_kind");
                    }
                } break;
                case primitive_kind::prelu: {
                    const auto &ep = e.prelu;
                    ss << delim << "prelu" << ":" << ep.mask;
                } break;
                default: assert(!"unsupported post op primitive kind!"); break;
            }
            delim = attr_delim;
        }
    }

    const rnn_data_qparams_t &rnn_qp = attr->rnn_data_qparams_;
    if (!rnn_qp.has_default_values()) {
        ss << field_delim() << "rnn_data_qparams:" << rnn_qp.scale_ << ":"
           << rnn_qp.shift_ << ";";
    }

    if (!attr->dropout_.has_default_values()) {
        ss << field_delim() << "attr-dropout";
        const memory_desc_wrapper mdw(attr->dropout_.user_dropout_desc_);
        switch (mdw.format_kind()) {
            case format_kind::blocked:
                if (!mdw.count_non_unit_dims(1))
                    ss << ":" << md2fmt_tag_str(&attr->dropout_.dropout_desc_);
                break;
            case format_kind::any: ss << ":any"; break;
            case format_kind::undef: ss << ":undef"; break;
            default: assert(!"unsupported format_kind");
        }
        ss << ":" << dnnl_dt2str(attr->dropout_.seed_dt_);
        ss << ":" << attr->dropout_.use_offset_;
        ss << ":" << attr->dropout_.use_host_scalars_;
    }
    return ss;
}

std::string attr2str(const primitive_attr_t *attr) {
    stringstream_t ss;
    ss << attr;
    return ss.str();
}

/* init_info section */
namespace {

template <typename pd_t>
std::string init_info_batch_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();

    ss << md2fmt_str("data", src_md, pd->src_md(0, true)->format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_binary(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src0_md = pd->invariant_src_md(0);
    auto src1_md = pd->invariant_src_md(1);
    auto src2_md = pd->invariant_src_md(2);
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src0_md, pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str("src", src1_md, pd->invariant_src_user_format_kind(1))
       << " ";
    if (pd->desc()->alg_kind == alg_kind_t::dnnl_binary_select) {
        ss << md2fmt_str("src", src2_md, pd->invariant_src_user_format_kind(2))
           << " ";
    }

    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2dim_str(src0_md) << ":" << md2dim_str(src1_md);
    if (pd->desc()->alg_kind == alg_kind_t::dnnl_binary_select)
        ss << ":" << md2dim_str(src2_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_concat(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << md2fmt_str("src", src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->desc()->concat_dimension << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << md2dim_str(src_i_md);
        if (i < pd->n_inputs() - 1) ss << ":";
    }

    return ss.str();
}

template <typename pd_t>
std::string init_info_convolution(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("wei", wei_md, pd->invariant_wei_user_format_kind())
       << " ";
    ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind())
       << " ";

    // `has_fused_dw` modifies the convolution output in the following way:
    // * It provides additional src, wei and bia md to show their presence and
    //   wei format (for convenience, it can't be used directly).
    // * It makes output spatial dimensions same as input since first conv is
    //   always 1x1 for such fusion. This is to make a problem benchdnn
    //   compatible.
    // * Note: Queried `dst_md` with final dimensions after fusion will reside
    //   in fused conv pd. The op_desc for it is created on the library side and
    //   filled with already blocked formats compatible with precedeing 1x1
    //   convolution. It means that it can't identify if original dst_md was
    //   created with `format_kind::any` or not. For purposes of re-construction
    //   of benchdnn line, intermediate "src_fused" is required - it gives an
    //   info about data type to pass into benchdnn and also whether 1x1 conv
    //   dst was created with `format_kind::any`.
    // * Note: DW-post op is the only reason why `arg_md` got `user_input`
    //   argument and also takes argument. This is due to dw post-op mds can
    //   be queried only through `arg_md` interface.
    const bool has_fused_dw
            = pd->attr()->post_ops_.find(primitive_kind::convolution) >= 0;
    if (has_fused_dw) {
        auto src_fused_md = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC);
        auto wei_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        auto bia_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        // User-provided dst memory descriptor.
        ss << md2fmt_str("src_fused", src_fused_md,
                pd->invariant_dst_user_format_kind(
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC))
           << " ";
        // Not user-provided memory descriptors.
        ss << md2fmt_str("wei_fused", wei_fused_md, format_kind::undef) << " ";
        ss << md2fmt_str("bia_fused", bia_fused_md, format_kind::undef) << " ";
        ss << md2fmt_str("dst", dst_md, format_kind::undef);
    } else {
        ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    if (pd->with_groups()) ss << "g" << pd->G();
    ss << "mb" << pd->MB() << "_" << "ic" << pd->IC() << "oc" << pd->OC()
       << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << (has_fused_dw ? pd->ID() : pd->OD())
           << "kd" << pd->KD() << "sd" << pd->KSD() << "dd" << pd->KDD() << "pd"
           << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << (has_fused_dw ? pd->IH() : pd->OH())
           << "kh" << pd->KH() << "sh" << pd->KSH() << "dh" << pd->KDH() << "ph"
           << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << (has_fused_dw ? pd->IW() : pd->OW())
       << "kw" << pd->KW() << "sw" << pd->KSW() << "dw" << pd->KDW() << "pw"
       << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_deconvolution(const engine_t *e, const pd_t *pd) {
    return init_info_convolution(e, pd);
}

template <typename pd_t>
std::string init_info_eltwise(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->use_dst() ? pd->dst_md(0) : pd->src_md(0);
    auto user_data_format_kind = pd->use_dst()
            ? pd->dst_md(0, true)->format_kind
            : pd->src_md(0, true)->format_kind;
    auto diff_src_md = pd->diff_src_md();

    ss << md2fmt_str("data", data_md, user_data_format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", diff_src_md,
                      pd->invariant_src_user_format_kind(0));
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " alpha:" << pd->desc()->alpha
       << " beta:" << pd->desc()->beta << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_gated_mlp(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    ss << md2fmt_str("src", pd->arg_md(DNNL_ARG_SRC), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_gate", pd->arg_md(DNNL_ARG_WEIGHTS_GATE), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_up", pd->arg_md(DNNL_ARG_WEIGHTS_UP), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_down", pd->arg_md(DNNL_ARG_WEIGHTS_DOWN), format_kind::undef)
       << " ";
    ss << md2fmt_str("dst", pd->arg_md(DNNL_ARG_DST), format_kind::undef);

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->activation() << ",";
    ss << "mb" << pd->MB() << "ic" << pd->IC() << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_gemm(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_a_md = pd->invariant_src_md(0);
    auto src_b_md = pd->invariant_src_md(1);
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    ss << md2fmt_str("src_a", src_a_md, pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str("src_b", src_b_md, pd->invariant_src_user_format_kind(1))
       << " ";
    if (pd->with_bias()) {
        ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind());
        ss << "_mask" << get_bia_mask();
        ss << " ";
    }
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2dim_str(src_a_md) << ":" << md2dim_str(src_b_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_group_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("src", src_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff_src", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << "g" << pd->desc()->groups << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_inner_product(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("wei", wei_md, pd->invariant_wei_user_format_kind())
       << " ";
    ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2desc_str(src_md);
    ss << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_layer_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    auto stats_md = pd->is_fwd() && !pd->stats_are_src() ? pd->dst_md(1)
                                                         : pd->src_md(1);
    auto user_stats_format_kind = pd->is_fwd() && !pd->stats_are_src()
            ? pd->dst_md(1, true)->format_kind
            : pd->src_md(1, true)->format_kind;
    auto scaleshift_md = pd->weights_md(0);
    auto diff_scaleshift_md = pd->diff_weights_md(0);

    ss << md2fmt_str("src", src_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind())
       << " ";
    ss << md2fmt_str("stats", stats_md, user_stats_format_kind);
    if (pd->use_scale()) {
        ss << " "
           << md2fmt_str("scale", scaleshift_md,
                      pd->weights_md(0, true)->format_kind);
    }
    if (pd->use_shift()) {
        ss << " "
           << md2fmt_str("shift", scaleshift_md,
                      pd->weights_md(0, true)->format_kind);
    }
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff_src", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }
    if (!pd->is_fwd() && pd->use_scale()) {
        ss << " "
           << md2fmt_str("diff_scale", diff_scaleshift_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }
    if (!pd->is_fwd() && pd->use_shift()) {
        ss << " "
           << md2fmt_str("diff_shift", diff_scaleshift_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_lrn(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();

    ss << md2fmt_str("data", data_md, pd->src_md(0, true)->format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", diff_src_md,
                      pd->invariant_src_user_format_kind(0));
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2desc_str(data_md);
    ss << "ls" << pd->desc()->local_size << "beta" << pd->desc()->lrn_beta;

    return ss.str();
}

std::string mds2str_matmul(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *wei_md,
        format_kind_t wei_user_format_kind, const memory_desc_t *bia_md,
        format_kind_t bia_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    stringstream_t ss;

    ss << md2fmt_str("src", src_md, src_user_format_kind) << " ";
    ss << md2fmt_str("wei", wei_md, wei_user_format_kind) << " ";
    if (!memory_desc_wrapper(bia_md).is_zero()) {
        ss << md2fmt_str("bia", bia_md, bia_user_format_kind);
        ss << "_mask" << get_bia_mask();
        ss << " ";
    }
    ss << md2fmt_str("dst", dst_md, dst_user_format_kind);

    std::string s = ss.str();
    return s;
}

std::string dims2fmt_str_matmul(
        const memory_desc_t *src_md, const memory_desc_t *wei_md) {
    return md2dim_str(src_md) + ":" + md2dim_str(wei_md);
}

template <typename pd_t>
std::string init_info_matmul(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_matmul(src_md, pd->invariant_src_user_format_kind(), wei_md,
            pd->invariant_wei_user_format_kind(), bia_md,
            pd->invariant_bia_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime_dims_masks:" << get_runtime_mask(src_md) << ":"
           << get_runtime_mask(wei_md);
    }
    ss << "," << dims2fmt_str_matmul(src_md, wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_pooling(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();
    auto ws_md = pd->workspace_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    if (!memory_desc_wrapper(ws_md).is_zero()) {
        ss << " " << md2fmt_str("ws", ws_md, format_kind::undef);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->IC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << pd->OD() << "kd" << pd->KD() << "sd"
           << pd->KSD() << "dd" << pd->KDD() << "pd" << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << pd->OH() << "kh" << pd->KH() << "sh"
           << pd->KSH() << "dh" << pd->KDH() << "ph" << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW() << "kw" << pd->KW() << "sw"
       << pd->KSW() << "dw" << pd->KDW() << "pw" << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_prelu(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md(0);
    auto wei_md = pd->weights_md(0);
    auto diff_data_md = pd->diff_src_md(0);
    auto diff_wei_md = pd->diff_weights_md(0);

    ss << md2fmt_str("data", data_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("wei", wei_md, pd->weights_md(0, true)->format_kind);
    if (!memory_desc_wrapper(diff_data_md).is_zero()) {
        ss << " "
           << md2fmt_str("diff", diff_data_md,
                      pd->diff_src_md(0, true)->format_kind);
    }
    if (!memory_desc_wrapper(diff_wei_md).is_zero()) {
        ss << " "
           << md2fmt_str("diff_wei", diff_wei_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(data_md) << ":" << md2dim_str(wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_reduction(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " p:" << pd->desc()->p
       << " eps:" << pd->desc()->eps << ",";
    ss << md2dim_str(src_md) << ":" << md2dim_str(dst_md);

    return ss.str();
}

std::string mds2str_reorder(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    std::string s;
    s += md2fmt_str("src", src_md, src_user_format_kind);
    s += " ";
    s += md2fmt_str("dst", dst_md, dst_user_format_kind);
    return s;
}

std::string dims2fmt_str_reorder(const memory_desc_t *src_md) {
    return md2dim_str(src_md);
}

template <typename pd_t>
std::string init_info_reorder(const engine_t *e, pd_t *pd) {
    stringstream_t ss;

    const auto src_ek = pd->desc()->src_engine_kind;
    const auto dst_ek = pd->desc()->dst_engine_kind;

    if (src_ek != dst_ek)
        ss << src_ek << "2" << dst_ek;
    else
        ss << e;

    ss << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_reorder(src_md, pd->invariant_src_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime-dim-mask:" << get_runtime_mask(src_md);
    }
    ss << "," << dims2fmt_str_reorder(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_resampling(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->C() << "_";
    if (pd->ndims() >= 5) ss << "id" << pd->ID() << "od" << pd->OD() << "_";
    if (pd->ndims() >= 4) ss << "ih" << pd->IH() << "oh" << pd->OH() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW();

    return ss.str();
}

template <typename pd_t>
std::string init_info_rnn(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    // TODO: shorten the names to consume fewer characters on verbose output.
    ss << md2fmt_str(
            "src_layer", pd->src_md(0), pd->src_md(0, true)->format_kind)
       << " ";
    if (pd->with_src_iter())
        ss << md2fmt_str(
                "src_iter", pd->src_md(1), pd->src_md(1, true)->format_kind)
           << " ";
    ss << md2fmt_str("wei_layer", pd->weights_md(0),
            pd->weights_md(0, true)->format_kind)
       << " ";
    ss << md2fmt_str(
            "wei_iter", pd->weights_md(1), pd->weights_md(1, true)->format_kind)
       << " ";
    if (pd->is_lstm_peephole())
        ss << md2fmt_str("wei_peephole", pd->weights_md(2),
                pd->weights_md(2, true)->format_kind)
           << " ";
    // TODO: separate methods for aux weights?
    if (pd->is_lstm_projection()) {
        auto proj_idx = 2 + pd->is_lstm_peephole();
        ss << md2fmt_str("wei_proj", pd->weights_md(proj_idx),
                pd->weights_md(proj_idx, true)->format_kind)
           << " ";
    }
    if (pd->with_bias()) {
        auto bias_idx = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
        ss << md2fmt_str("bias", pd->weights_md(bias_idx),
                pd->weights_md(bias_idx, true)->format_kind)
           << " ";
    }
    ss << md2fmt_str(
            "dst_layer", pd->dst_md(0), pd->dst_md(0, true)->format_kind);
    if (pd->with_dst_iter())
        ss << " "
           << md2fmt_str("dst_iter", pd->dst_md(1),
                      pd->dst_md(1, true)->format_kind);

    if (!pd->is_fwd()) {
        ss << " ";
        ss << md2fmt_str("diff_src_layer", pd->diff_src_md(0),
                pd->diff_src_md(0, true)->format_kind)
           << " ";
        if (pd->with_src_iter())
            ss << md2fmt_str("diff_src_iter", pd->diff_src_md(1),
                    pd->diff_src_md(1, true)->format_kind)
               << " ";
        ss << md2fmt_str("diff_wei_layer", pd->diff_weights_md(0),
                pd->diff_weights_md(0, true)->format_kind)
           << " ";
        ss << md2fmt_str("diff_wei_iter", pd->diff_weights_md(1),
                pd->diff_weights_md(1, true)->format_kind)
           << " ";
        if (pd->is_lstm_peephole())
            ss << md2fmt_str("diff_wei_peephole", pd->diff_weights_md(2),
                    pd->diff_weights_md(2, true)->format_kind)
               << " ";
        if (pd->is_lstm_projection()) {
            auto proj_idx = 2 + pd->is_lstm_peephole();
            ss << md2fmt_str("diff_wei_proj", pd->weights_md(proj_idx),
                    pd->weights_md(proj_idx, true)->format_kind)
               << " ";
        }
        if (pd->with_bias()) {
            auto bias_idx
                    = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
            ss << md2fmt_str("diff_bias", pd->weights_md(bias_idx),
                    pd->weights_md(bias_idx, true)->format_kind)
               << " ";
        }
        ss << md2fmt_str("diff_dst_layer", pd->diff_dst_md(0),
                pd->diff_dst_md(0, true)->format_kind);
        if (pd->with_dst_iter())
            ss << " "
               << md2fmt_str("diff_dst_iter", pd->diff_dst_md(1),
                          pd->diff_dst_md(1, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->cell_kind()
       << " direction:" << dnnl_rnn_direction2str(pd->direction())
       << " activation:" << pd->activation_kind()
       << " flags:" << rnn_flags2str(pd->desc()->flags) << ",";

    ss << "l" << pd->L() << "t" << pd->T() << "mb" << pd->MB() << "sic"
       << pd->SIC() << "slc" << pd->SLC() << "dhc" << pd->DHC() << "dic"
       << pd->DIC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_shuffle(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->invariant_src_md();

    ss << md2fmt_str("data", data_md, pd->invariant_src_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->axis() << " group:" << pd->group_size() << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_softmax(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->dst_md();
    auto diff_dst_md = pd->diff_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->dst_md(0, true)->format_kind);
    if (!types::is_zero_md(diff_dst_md)) {
        ss << " "
           << md2fmt_str("diff_dst", diff_dst_md,
                      pd->diff_dst_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->alg_kind() << " axis:" << pd->axis() << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_sum(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << md2fmt_str("src", src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(dst_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_sdpa(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    const sdpa_desc_t *desc = pd->desc();
    ss << md2fmt_str(
            "query", desc->qry_md(), pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str(
            "key", desc->key_md(), pd->invariant_src_user_format_kind(1))
       << " ";
    ss << md2fmt_str(
            "val", desc->val_md(), pd->invariant_src_user_format_kind(2))
       << " ";
    if (pd->with_attn_mask())
        ss << md2fmt_str("msk", desc->attn_mask_md(),
                pd->invariant_src_user_format_kind(3))
           << " ";
    ss << md2fmt_str("dst", pd->dst_md(), pd->invariant_dst_user_format_kind())
       << ",";

    std::string delimiter;
    if (pd->with_key_scales() || pd->with_value_scales()) {
        ss << delimiter << "attr-scales:";
        delimiter = "";
        if (pd->with_key_scales()) {
            ss << delimiter << "key:" << desc->kq_scales;
            delimiter = "+";
        }
        if (pd->with_value_scales()) {
            ss << delimiter << "val:" << desc->vs_scales;
            delimiter = "+";
        }
        delimiter = " ";
    }
    if (pd->with_key_zp() || pd->with_value_zp()) {
        ss << delimiter << "attr-zero-points:";
        delimiter = "";
        if (pd->with_key_zp()) {
            ss << delimiter << "key:" << desc->kq_zero_points;
            delimiter = "+";
        }
        if (pd->with_value_zp()) {
            ss << delimiter << "val:" << desc->vs_zero_points;
            delimiter = "+";
        }
        delimiter = " ";
    }
    ss << delimiter << pd->attr();

    delimiter = " ";
    ss << ",alg:" << desc->softmax_alg;
    if (pd->with_attn_mask()) {
        auto *md = desc->attn_mask_md();
        ss << delimiter << "msk:buffer_" << (md->dims[2] == 1 ? 1 : 2) << 'd';
    } else if (pd->with_causal_mask()) {
        ss << delimiter;
        if (desc->mask_type == attn_mask_type::top_left)
            ss << "msk:causal_top_left";
        else
            ss << "msk:causal_bottom_right";
    }
    if (pd->with_attn_scale()) {
        ss << delimiter << "scl:";
        if (desc->invert_scale)
            ss << "div:";
        else
            ss << "mul:";
        ss << dnnl_dt2str(desc->scale_md()->data_type) << ":";
        if (pd->with_host_scale())
            ss << "host";
        else
            ss << "device";
    }

    ss << "," << md2dim_str(desc->qry_md()) << ":" << md2dim_str(desc->key_md())
       << ":" << md2dim_str(desc->val_md());

    return ss.str();
}

} // namespace

std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md) {
    // Note: pass format_kind::undef since runtime dims-ed mds can't have
    // format_kind::any at any stage.
    std::string s;
#if defined(DISABLE_VERBOSE)
    return s;
#endif

    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = mds2str_matmul(src_md, format_kind::undef, wei_md,
                    format_kind::undef, bia_md, format_kind::undef, dst_md,
                    format_kind::undef);
            break;
        case primitive_kind::reorder:
            s = mds2str_reorder(
                    src_md, format_kind::undef, dst_md, format_kind::undef);
            break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md) {
    std::string s;
#if defined(DISABLE_VERBOSE)
    return s;
#endif

    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = dims2fmt_str_matmul(src_md, wei_md);
            break;
        case primitive_kind::reorder: s = dims2fmt_str_reorder(src_md); break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

void pd_info_t::init(engine_t *engine, const primitive_desc_t *pd) {
    // Handles VERBOSE_DISABLE since `is_initialized_` is set to `true`.
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
    // clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        str_ = init_info_##kind(engine, (const kind##_pd_t *)pd); \
        break

        switch ((int)pd->kind()) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(concat);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(gated_mlp);
            CASE(gemm);
            CASE(group_normalization);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(matmul);
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            CASE(sum);
            CASE(sdpa);
            case primitive_kind::zero_pad:
              str_ = "zero_pad, unknown info";
              break;
            default:
              str_ = "unknown primitive info";
              assert(!"unknown primitive kind");
        }
#undef CASE
        // clang-format on

        is_initialized_ = true;
    });
}

} // namespace impl
} // namespace dnnl
