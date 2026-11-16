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

#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#include "cpu/rnn/rnn_reorders.hpp"

#include <assert.h>

namespace dnnl {
namespace impl {
namespace cpu {

DEFINE_PD_CREATE(rnn_data_reorder_t::pd_t)
DEFINE_PD_CREATE(rnn_weights_reorder_s8_t::pd_t)
DEFINE_PD_CREATE(rnn_weights_reorder_t::pd_t)
DEFINE_PD_CREATE(rnn_brgemm_weights_reorder_s8_t::pd_t)

namespace {

void init_dims(dim_t &L, dim_t &D, dim_t &I, dim_t &G, dim_t &O,
        const memory_desc_wrapper &mdw) {
    const auto dims = mdw.dims();
    const auto ndims = mdw.ndims();
    L = dims[0];
    D = dims[1];
    I = dims[2];
    G = 0;
    O = 0;
    // weights_layer/weights_iter case
    if (ndims == 5) {
        G = dims[3];
        O = dims[4];
    }
    // projection weights case
    if (ndims == 4) {
        G = 1;
        O = dims[3];
    }
    assert(G != 0 && O != 0);
}

void quantize_igo(int8_t *scratch_quantized, const memory_desc_wrapper &src_d,
        const float *src, int mask, float *scales) {
    // TODO: trivial strides assumes here.
    //       Use proper strides where appropriate
    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    assert(scales != nullptr);
    parallel(0, [=](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(L * D * I, nthr, ithr, start, end);
        for (dim_t ldi = start; ldi < end; ldi++) {
            for (dim_t go = 0; go < G * O; go++) {
                const float s = scales[(mask == 0) ? 0 : go];
                scratch_quantized[ldi * G * O + go]
                        = q10n::qz_b0_t<float, int8_t>()(
                                src[ldi * G * O + go], s);
            }
        }
    });
}

void quantize_goi(int8_t *scratch_quantized, const memory_desc_wrapper &src_d,
        const float *src, int mask, float *scales) {
    // TODO: trivial strides assumes here.
    //       Use proper strides where appropriate
    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    assert(scales != nullptr);
    parallel_nd(L * D, G * O, [=](dim_t ld, dim_t go) {
        const float s = scales[(mask == 0) ? 0 : go];
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < I; i++) {
            scratch_quantized[ld * I * G * O + i * G * O + go]
                    = q10n::qz_b0_t<float, int8_t>()(
                            src[ld * G * O * I + go * I + i], s);
        }
    });
}

void compensate_igo(float *compensation, const memory_desc_wrapper &src_d,
        int8_t *scratch_quantized, int32_t *scratch_compensation,
        size_t scratch_comp_sz, int nthr) {
    // TODO: trivial strides assumed here.
    //       Use proper strides where appropriate
    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    // We parallelize on LD and GO
    // TODO: maybe restrict parallelism as we might have large
    // parallelisation overhead if dimensions are small
    const int LD_nthr = static_cast<int>(nstl::min<dim_t>(L * D, nthr));
    const int GO_nthr
            = static_cast<int>(nstl::min<dim_t>(G * O, nthr / LD_nthr));
    parallel(nthr, [=](const int ithr, const int nthr) {
        int LD_ithr = -1;
        int GO_ithr = -1;
        dim_t LD_s = -1, LD_e = -1;
        dim_t GO_s = -1, GO_e = -1;
        if (ithr < LD_nthr * GO_nthr) {
            LD_ithr = ithr % LD_nthr;
            GO_ithr = ithr / LD_nthr;
            balance211(L * D, LD_nthr, LD_ithr, LD_s, LD_e);
            balance211(G * O, GO_nthr, GO_ithr, GO_s, GO_e);
        }
        int32_t *compensation_s32
                = scratch_compensation + ithr * scratch_comp_sz;
        for (dim_t ld = LD_s; ld < LD_e; ld++) {
            if (I == 1) {
                PRAGMA_OMP_SIMD()
                for (dim_t go = GO_s; go < GO_e; go++)
                    compensation[ld * G * O + go] = q10n::saturate<float>(
                            scratch_quantized[ld * I * G * O + go]);
            } else {
                // We split the loop on I in three to avoid conditionals or zeroing compensation
                dim_t i = 0;
                PRAGMA_OMP_SIMD()
                for (dim_t go = GO_s; go < GO_e; go++)
                    compensation_s32[go]
                            = scratch_quantized[go + G * O * (i + I * (ld))];
                // 1 <= i < I-1
                for (i = 1; i < I - 1; i++) {
                    PRAGMA_OMP_SIMD()
                    for (dim_t go = GO_s; go < GO_e; go++)
                        compensation_s32[go] += scratch_quantized[go
                                + G * O * (i + I * (ld))];
                }
                // i = I-1
                PRAGMA_OMP_SIMD()
                for (dim_t go = GO_s; go < GO_e; go++)
                    compensation[ld * G * O + go] = static_cast<float>(
                            q10n::saturate<float>(compensation_s32[go]
                                    + scratch_quantized[go
                                            + G * O * (i + I * (ld))]));
            }
        }
    });
}

void compensate_goi(float *compensation, const memory_desc_wrapper &src_d,
        int8_t *scratch_quantized) {
    // TODO: trivial strides assumed here.
    //       Use proper strides where appropriate
    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    parallel_nd(L * D, G * O, [=](dim_t ld, dim_t go) {
        int32_t compensation_s32 = 0;
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < I; i++) {
            compensation_s32
                    += scratch_quantized[ld * I * G * O + i * G * O + go];
        }
        // TODO: do not convert to f32 if this compensation is not
        // going to be added to a bias (e.g. like in lstm
        // projection where it is directly added to the s32
        // accumulators)
        compensation[ld * G * O + go]
                = static_cast<float>(q10n::saturate<float>(compensation_s32));
    });
}

template <typename data_t>
void transpose_ld(
        data_t *output, const data_t *input, dim_t LD, dim_t M, dim_t N) {
    parallel_nd(LD, N, [=](dim_t ld, dim_t i) {
        for (dim_t j = 0; j < M; j++) {
            output[ld * M * N + i * M + j] = input[ld * M * N + j * N + i];
        }
    });
}

} // namespace

bool rnn_data_reorder_t::is_dense() const {
    const memory_desc_wrapper &input_d = pd()->src_md();
    const memory_desc_wrapper &output_d = pd()->dst_md();
    return utils::everyone_is(1,
            input_d.blocking_desc().strides[input_d.ndims() - 1],
            output_d.blocking_desc().strides[output_d.ndims() - 1]);
}

/* This function assumes that only the innermost dimension (C) is
   dense (that is to say, stride is 1).  This is enough to have
   good performance and allow non trivial strides on other
   dimensions (to allow an "optimized" path for views for
   example).
 */
template <typename out_data_t>
status_t rnn_data_reorder_t::execute_dense(out_data_t *output,
        const float *input, const float scale, const float shift) const {
    const memory_desc_wrapper &input_d = pd()->src_md();
    const memory_desc_wrapper &output_d = pd()->dst_md();
    const dim_t outer_dim
            = utils::array_product(input_d.dims(), input_d.ndims() - 1);
    const dim_t inner_dim = input_d.dims()[input_d.ndims() - 1];

    parallel(0, [=](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(outer_dim, nthr, ithr, start, end);
        for (dim_t i = start; i < end; ++i) {
            const dim_t off_in = input_d.off_l(i * inner_dim);
            const dim_t off_out = output_d.off_l(i * inner_dim);
            const float *__restrict i_ = input + off_in;
            out_data_t *__restrict o_ = output + off_out;
            PRAGMA_OMP_SIMD()
            for (dim_t j = 0; j < inner_dim; ++j) {
                const float in = i_[j] * scale + shift;
                o_[j] = q10n::qz_a1b0_t<float, out_data_t>()(in);
            }
        }
    });
    return status::success;
}

template <typename out_data_t>
status_t rnn_data_reorder_t::execute_generic(out_data_t *output,
        const float *input, float scale, float shift) const {
    const memory_desc_wrapper &input_d = pd()->src_md();
    const memory_desc_wrapper &output_d = pd()->dst_md();
    const size_t nelems = input_d.nelems();
    parallel_nd(nelems, [=](size_t i) {
        const float in = input[input_d.off_l(i)] * scale + shift;
        output[output_d.off_l(i)] = q10n::qz_a1b0_t<float, out_data_t>()(in);
    });
    return status::success;
}

status_t rnn_data_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const float *, DNNL_ARG_FROM);
    const float scale = pd()->attr()->rnn_data_qparams_.scale_;
    const float shift = pd()->attr()->rnn_data_qparams_.shift_;
    const bool dense = is_dense();

    if (pd()->dst_md()->data_type == data_type::s8) {
        auto output = CTX_OUT_MEM(int8_t *, DNNL_ARG_TO);
        return dense ? execute_dense(output, input, scale, shift)
                     : execute_generic(output, input, scale, shift);
    }
    auto output = CTX_OUT_MEM(uint8_t *, DNNL_ARG_TO);
    return dense ? execute_dense(output, input, scale, shift)
                 : execute_generic(output, input, scale, shift);
}

status_t rnn_weights_reorder_s8_t::execute(const exec_ctx_t &ctx) const {
    // TODO: trivial strides assumed here.
    //       Use proper strides where appropriate

    using namespace format_tag;

    auto src = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const memory_desc_wrapper &src_d = pd()->src_md();
    const memory_desc_wrapper &dst_d = pd()->dst_md();
    if (src_d.has_zero_dim()) {
        assert(dst_d.has_zero_dim());
        return status::success;
    }

    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    /* Quantize src & compute compensation */
    auto scratch_quantized
            = (int8_t *__restrict)ctx.get_scratchpad_grantor()
                      .template get<void>(memory_tracking::names::
                                      key_reorder_rnn_weights_quantization);
    auto scratch_compensation
            = (int32_t *__restrict)ctx.get_scratchpad_grantor()
                      .template get<void>(memory_tracking::names::
                                      key_reorder_rnn_weights_reduction);
    float *comp = reinterpret_cast<float *>(
            dst + dst_d.rnn_packed_desc().offset_compensation);
    float *scales = nullptr;
    int mask = 0;
    if (src_d.ndims() == 5) {
        scales = pd()->attr()->rnn_weights_qparams_.scales_;
        mask = pd()->attr()->rnn_weights_qparams_.mask_;
    }
    if (src_d.ndims() == 4) {
        scales = pd()->attr()->rnn_weights_projection_qparams_.scales_;
        mask = pd()->attr()->rnn_weights_projection_qparams_.mask_;
    }
    /* Step 1: we quantize if we need to */
    if (src_d.data_type() == data_type::f32) {
        switch (pd()->itag_) {
            case ldigo:
            case ldio:
                quantize_igo(scratch_quantized, src_d, (const float *)src, mask,
                        scales);
                break;
            case ldgoi:
            case ldoi:
                quantize_goi(scratch_quantized, src_d, (const float *)src, mask,
                        scales);
                break;
            default: assert(!"Unsupported reorder");
        }
    } else
        scratch_quantized = (int8_t *__restrict)src;

    /* Step 2: we pre-compute the compensation */
    switch (pd()->itag_) {
        case ldigo:
        case ldio:
            compensate_igo(comp, src_d, scratch_quantized, scratch_compensation,
                    pd()->thr_scratch_comp_sz_, pd()->nthr_);
            break;
        case ldgoi:
        case ldoi: compensate_goi(comp, src_d, scratch_quantized); break;
        default: assert(!"Unsupported reorder");
    }

    /* Step 3: we pack the matrix */
    const auto off_igo = [&](dim_t l, dim_t d, dim_t i, dim_t g, dim_t o) {
        return o + O * (g + G * (i + I * (d + D * l)));
    };
    const int n_parts = dst_d.rnn_packed_desc().n_parts;
    const size_t *size_packed_cell = dst_d.rnn_packed_desc().part_pack_size;
    const int *parts = dst_d.rnn_packed_desc().parts;
    const dim_t n = dst_d.rnn_packed_desc().n;
    const dim_t ldb = dst_d.rnn_packed_desc().ldb;
    char *to_pack = dst;

    for (dim_t l = 0; l < L; l++) {
        for (dim_t d = 0; d < D; d++) {
            for (dim_t p = 0; p < n_parts; p++) {
                dim_t g = (p > 0) ? parts[p - 1] : 0;
                dim_t m_p = parts[p] * O;
                dim_t k_p = I;
                dim_t lda = (dim_t)G * O;
                CHECK(pd()->gemm_pack("A", "N", "N", &m_p, &n, &k_p, &lda, &ldb,
                        scratch_quantized + off_igo(l, d, 0, g, 0), to_pack));
                to_pack += size_packed_cell[p];
            }
        }
    }
    return status::success;
}

status_t rnn_weights_reorder_t::execute(const exec_ctx_t &ctx) const {
    // TODO: trivial strides assumed here.
    //       Use proper strides where appropriate

    using namespace format_tag;
    using namespace rnn_packed_format;

    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const memory_desc_wrapper &input_d = pd()->src_md();
    const memory_desc_wrapper &output_d = pd()->dst_md();
    if (input_d.has_zero_dim()) {
        assert(output_d.has_zero_dim());
        return status::success;
    }

    const rnn_packed_desc_t &rnn_pdata = output_d.rnn_packed_desc();
    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, input_d);

    /* Pack */
    const bool from_igo = utils::one_of(pd()->itag_, ldigo, ldio);
    const bool to_igo = utils::one_of(rnn_pdata.format, ldigo_p, ldio_p);
    const int n_parts = rnn_pdata.n_parts;
    const size_t *size_packed_cell = rnn_pdata.part_pack_size;
    const int *parts = rnn_pdata.parts;
    const dim_t n = rnn_pdata.n;
    const bool is_bf16_out = output_d.data_type() == data_type::bf16;

    /* Convert to fp32*/
    const char *input_cvt = input;
    if (input_d.data_type() == data_type::f32 && is_bf16_out) {
        auto cvt = ctx.get_scratchpad_grantor().template get<bfloat16_t>(
                memory_tracking::names::key_reorder_rnn_weights_xf16_cvt);
        parallel_nd(L * D, [=](dim_t ld) {
            types::cvt_from_float(cvt + ld * G * O * I,
                    (const float *)input + ld * G * O * I, G * O * I);
        });
        input_cvt = (const char *)cvt;
    }

    /* Transpose weights prior to packing to ensure that packed GEMM
     * algorithm will be dispatched */
    const char *input_tr = input_cvt;
    if (from_igo != to_igo) {
        auto tr = ctx.get_scratchpad_grantor().template get<char>(
                memory_tracking::names::key_reorder_rnn_weights_transposition);
        const dim_t M = to_igo ? G * O : I;
        const dim_t N = to_igo ? I : G * O;
        if (is_bf16_out)
            transpose_ld((bfloat16_t *)tr, (const bfloat16_t *)input_cvt, L * D,
                    M, N);
        else
            transpose_ld((float *)tr, (const float *)input_cvt, L * D, M, N);
        input_tr = tr;
    }

    const auto off_igo = [&](dim_t l, dim_t d, dim_t i, dim_t g, dim_t o) {
        return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
    };
    const auto off_goi = [&](dim_t l, dim_t d, dim_t i, dim_t g, dim_t o) {
        return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
    };
    const dim_t lda = to_igo ? G * O : I;
    const dim_t ldb = rnn_pdata.ldb;
    for (dim_t l = 0; l < L; l++) {
        for (dim_t d = 0; d < D; d++) {
            for (dim_t p = 0; p < n_parts; p++) {
                const dim_t g = (p > 0) ? parts[p - 1] : 0;
                const dim_t m_p = to_igo ? parts[p] * O : I;
                const dim_t k_p = to_igo ? I : parts[p] * O;
                const dim_t off = to_igo ? off_igo(l, d, 0, g, 0)
                                         : off_goi(l, d, 0, g, 0);
                if (is_bf16_out) {
                    CHECK(gemm_bf16bf16f32_pack("A", "N", "N", &m_p, &n, &k_p,
                            &lda, &ldb, (const bfloat16_t *)input_tr + off,
                            (bfloat16_t *)output));
                } else {
                    CHECK(sgemm_pack("A", "N", "N", &m_p, &n, &k_p, &lda, &ldb,
                            (const float *)input_tr + off, (float *)output));
                }
                output += size_packed_cell[p];
            }
        }
    }
    return status::success;
}

status_t rnn_brgemm_weights_reorder_s8_t::execute(const exec_ctx_t &ctx) const {
    using namespace format_tag;
    using namespace utils;
    using namespace memory_extra_flags;

    auto src = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto dst = CTX_OUT_MEM(int8_t *, DNNL_ARG_TO);
    const memory_desc_wrapper &src_d = pd()->src_md();
    const memory_desc_wrapper &dst_d = pd()->dst_md();
    if (src_d.has_zero_dim()) {
        assert(dst_d.has_zero_dim());
        return status::success;
    }

    const int o_block = static_cast<int>(dst_d.blocking_desc().inner_blks[0]);
    static constexpr int i_block = 4;

    dim_t L, D, I, G, O;
    init_dims(L, D, I, G, O, src_d);

    const auto &pdims = dst_d.padded_dims();
    const dim_t pI = pdims[2];
    const dim_t pO = (src_d.ndims() == 5) ? pdims[4] : pdims[3];
    const dim_t IB = pI / i_block;
    const dim_t OB = pO / o_block;

    const size_t compensation_offset = (size_t)L * D * G * pI * pO;

    /* Quantize src & compute compensation */
    auto scratch_quantized
            = (int8_t *__restrict)ctx.get_scratchpad_grantor()
                      .template get<void>(memory_tracking::names::
                                      key_reorder_rnn_weights_quantization);
    auto scratch_compensation
            = (int32_t *__restrict)ctx.get_scratchpad_grantor()
                      .template get<void>(memory_tracking::names::
                                      key_reorder_rnn_weights_reduction);
    float *comp = reinterpret_cast<float *>(dst + compensation_offset);
    const bool req_s8s8_comp = dst_d.extra().flags & rnn_u8s8_compensation;
    const auto mask_ok = [&](int mask) {
        return mask
                == ((src_d.ndims() == 5) ? 27 /* 11011 */
                                         : 13 /* 1101 */);
    };

    float *scales = nullptr;
    int mask = 0;
    if (src_d.ndims() == 5) {
        scales = pd()->attr()->rnn_weights_qparams_.scales_;
        mask = pd()->attr()->rnn_weights_qparams_.mask_;
    }
    if (src_d.ndims() == 4) {
        scales = pd()->attr()->rnn_weights_projection_qparams_.scales_;
        mask = pd()->attr()->rnn_weights_projection_qparams_.mask_;
    }
    if (src_d.data_type() == data_type::f32) {
        quantize_igo(
                scratch_quantized, src_d, (const float *)src, mask, scales);
    } else
        scratch_quantized = (int8_t *__restrict)src;

    if (req_s8s8_comp && mask_ok(dst_d.extra().compensation_mask))
        compensate_igo(comp, src_d, scratch_quantized, scratch_compensation,
                pd()->thr_scratch_comp_sz_, pd()->nthr_);

    const auto off_plain = [=](dim_t l, dim_t d, dim_t i, dim_t g, dim_t o) {
        return ((((dim_t)l * D + d) * I + i) * G + g) * O + o;
    };

    const auto off_blk = [=](dim_t l, dim_t d, dim_t g, dim_t ob, dim_t ib) {
        return (((((dim_t)l * D + d) * G + g) * OB + ob) * IB + ib) * i_block
                * o_block;
    };
    const auto off_inner_blk = [=](int xdim, int y, int x, int folding_factor) {
        const int row = (xdim) * (y / folding_factor) * folding_factor;
        const int col = x * folding_factor + (y % folding_factor);
        return row + col;
    };
    const auto kernel_plain_to_blocked
            = [=](const int8_t *inp, int8_t *out, dim_t ib, dim_t ob) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < i_block * o_block; i++)
            out[i] = 0;

        for_(int i = 0; i < i_block; i++)
        for (int o = 0; o < o_block; o++) {
            if ((i + ib * i_block < I) && (o + ob * o_block < O))
                out[off_inner_blk(o_block, i, o, i_block)] = inp[i * G * O + o];
        }
    };

    parallel_nd(L, D, G, OB, IB,
            [=](dim_t l, dim_t d, dim_t g, dim_t ob, dim_t ib) {
        auto inp = &scratch_quantized[off_plain(
                l, d, ib * i_block, g, ob * o_block)];
        auto out = &dst[off_blk(l, d, g, ob, ib)];

        kernel_plain_to_blocked(inp, out, ib, ob);
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
