/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_im2row_convolution.hpp"

#include <algorithm>
#include <array>
#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// Collect NCHW source into NHWC patches using 4-channel by 4-pixel transpose
// tiles. Each call covers a channel block and handles partial tiles without
// reading outside src. The output may also be converted from f32 to bf16.
struct jit_kai_nchw_im2row_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_kai_nchw_im2row_t)

    struct call_params_t {
        const void *src;
        void *dst;
        dim_t channels, width;
        dim_t src_width; // Pixels left in the source row, for safe vector loads.
    };

    jit_kai_nchw_im2row_t(size_t src_dt_size, size_t dst_dt_size,
            dim_t src_channel_stride, dim_t src_pixel_stride,
            dim_t dst_row_size)
        : src_dt_size_(src_dt_size)
        , dst_dt_size_(dst_dt_size)
        , src_channel_stride_(src_channel_stride * src_dt_size)
        , src_pixel_stride_(src_pixel_stride * src_dt_size)
        , dst_pixel_stride_(dst_row_size * dst_dt_size) {}

    void operator()(const call_params_t &p) const {
        jit_generator_t::operator()(&p);
    }

private:
    const size_t src_dt_size_, dst_dt_size_, src_channel_stride_,
            src_pixel_stride_, dst_pixel_stride_;

    void generate() override {
        using namespace Xbyak_aarch64;
        RegPoolManager regs;
        const auto params = regs.alloc<XReg>(abi_param1.getIdx());
        const auto src = regs.alloc<XReg>();
        const auto dst = regs.alloc<XReg>();
        const auto channels = regs.alloc<XReg>();
        const auto width = regs.alloc<XReg>();
        const auto src_width = regs.alloc<XReg>();
        const auto channel_src = regs.alloc<XReg>();
        const auto pixel_src = regs.alloc<XReg>();
        const auto pixel_dst = regs.alloc<XReg>();
        const auto scalar_value = regs.alloc<WReg>();
        const auto scratch = regs.alloc<XReg>();
        const auto channel_index = regs.alloc<XReg>();
        const auto channel_tiles = regs.alloc<XReg>();
        const auto pixel_index = regs.alloc<XReg>();
        const std::array<VReg, 4> tile = {regs.alloc<VReg>(),
                regs.alloc<VReg>(), regs.alloc<VReg>(), regs.alloc<VReg>()};
        const std::array<VReg, 4> tmp = {regs.alloc<VReg>(), regs.alloc<VReg>(),
                regs.alloc<VReg>(), regs.alloc<VReg>()};

        preamble();
        ldr(src, ptr(params, static_cast<int>(offsetof(call_params_t, src))));
        ldr(dst, ptr(params, static_cast<int>(offsetof(call_params_t, dst))));
        ldr(channels,
                ptr(params,
                        static_cast<int>(offsetof(call_params_t, channels))));
        ldr(width,
                ptr(params, static_cast<int>(offsetof(call_params_t, width))));
        ldr(src_width,
                ptr(params,
                        static_cast<int>(offsetof(call_params_t, src_width))));

        Label scalar;
        cmp(width, 4);
        b(LT, scalar);
        lsr(channel_tiles, channels, 2);
        asm_for(channel_index, channel_tiles, [&] {
            for (int c = 0; c < 4; ++c) {
                add_imm(channel_src, src, c * src_channel_stride_, scratch);
                if (src_pixel_stride_ == src_dt_size_) {
                    if (src_dt_size_ == 4)
                        ldr(QReg(tile[c].getIdx()), ptr(channel_src));
                    else
                        ldr(DReg(tile[c].getIdx()), ptr(channel_src));
                } else {
                    Label gather, loaded;
                    if (src_pixel_stride_ == 2 * src_dt_size_) {
                        // Read eight contiguous pixels and keep the even
                        // lanes, only when the extra pixel is in this row.
                        cmp(src_width, 8);
                        b(LT, gather);
                        ldr(QReg(tile[c].getIdx()), ptr(channel_src));
                        if (src_dt_size_ == 4) {
                            ldr(QReg(tmp[0].getIdx()), ptr(channel_src, 16));
                            uzp1(tile[c].s4, tile[c].s4, tmp[0].s4);
                        } else {
                            uzp1(tile[c].h8, tile[c].h8, tile[c].h8);
                        }
                        b(loaded);
                    }
                    L(gather);
                    for (int w = 0; w < 4; ++w) {
                        add_imm(pixel_src, channel_src, w * src_pixel_stride_,
                                scratch);
                        if (src_dt_size_ == 4)
                            ld1(tile[c].s4[w], ptr(pixel_src));
                        else
                            ld1(tile[c].h4[w], ptr(pixel_src));
                    }
                    L(loaded);
                }
            }
            // Four channel vectors become four pixel vectors.
            if (src_dt_size_ == 4) {
                trn1(tmp[0].s4, tile[0].s4, tile[1].s4);
                trn2(tmp[1].s4, tile[0].s4, tile[1].s4);
                trn1(tmp[2].s4, tile[2].s4, tile[3].s4);
                trn2(tmp[3].s4, tile[2].s4, tile[3].s4);
                trn1(tile[0].d2, tmp[0].d2, tmp[2].d2);
                trn1(tile[1].d2, tmp[1].d2, tmp[3].d2);
                trn2(tile[2].d2, tmp[0].d2, tmp[2].d2);
                trn2(tile[3].d2, tmp[1].d2, tmp[3].d2);
            } else {
                trn1(tmp[0].h4, tile[0].h4, tile[1].h4);
                trn2(tmp[1].h4, tile[0].h4, tile[1].h4);
                trn1(tmp[2].h4, tile[2].h4, tile[3].h4);
                trn2(tmp[3].h4, tile[2].h4, tile[3].h4);
                trn1(tile[0].s2, tmp[0].s2, tmp[2].s2);
                trn1(tile[1].s2, tmp[1].s2, tmp[3].s2);
                trn2(tile[2].s2, tmp[0].s2, tmp[2].s2);
                trn2(tile[3].s2, tmp[1].s2, tmp[3].s2);
            }
            for (int w = 0; w < 4; ++w) {
                add_imm(pixel_dst, dst, w * dst_pixel_stride_, scratch);
                if (src_dt_size_ != dst_dt_size_)
                    bfcvtn(tile[w].h4, tile[w].s4);
                if (dst_dt_size_ == 4)
                    str(QReg(tile[w].getIdx()), ptr(pixel_dst));
                else
                    str(DReg(tile[w].getIdx()), ptr(pixel_dst));
            }
            add_imm(src, src, 4 * src_channel_stride_, scratch);
            add(dst, dst, 4 * dst_dt_size_);
        });
        and_(channels, channels, 3);

        L(scalar);
        asm_for(channel_index, channels, [&] {
            mov(pixel_src, src);
            mov(pixel_dst, dst);
            asm_for(pixel_index, width, [&] {
                if (src_dt_size_ != dst_dt_size_) {
                    ldr(SReg(tile[0].getIdx()), ptr(pixel_src));
                    bfcvtn(tile[0].h4, tile[0].s4);
                    str(HReg(tile[0].getIdx()), ptr(pixel_dst));
                } else if (src_dt_size_ == 4) {
                    ldr(scalar_value, ptr(pixel_src));
                    str(scalar_value, ptr(pixel_dst));
                } else {
                    ldrh(scalar_value, ptr(pixel_src));
                    strh(scalar_value, ptr(pixel_dst));
                }
                add_imm(pixel_src, pixel_src, src_pixel_stride_, scratch);
                add_imm(pixel_dst, pixel_dst, dst_pixel_stride_, scratch);
            });
            add_imm(src, src, src_channel_stride_, scratch);
            add(dst, dst, dst_dt_size_);
        });
        postamble();
    }
};

namespace {

size_t im2row_size_bytes(
        const kai_im2row_convolution_fwd_t::pd_t &pd, size_t src_dt_size) {
    return static_cast<size_t>(pd.gemm_args()._Msize) * pd.gemm_args()._Ksize
            * src_dt_size;
}

void copy_strided_width(char *&dst, const char *&src, dim_t width_run,
        size_t row_bytes, size_t src_step) {
    for (dim_t i = 0; i < width_run; ++i) {
        std::memcpy(dst, src, row_bytes);
        src += src_step;
        dst += row_bytes;
    }
}

} // namespace

data_type_t kai_im2row_convolution_fwd_t::pd_t::src_preparation_dt(
        const cpu_convolution_fwd_pd_t &pd) {

    // NHWC im2row preserves FP32 for KleidiAI's fast-mode packing and selection,
    // which appears to be faster than fusing. This requires more investigation
    if (memory_desc_matches_tag(*pd.src_md(), format_tag::nhwc))
        return pd.src_md()->data_type;

    // Fuse conversion with the NCHW transpose.
    if (memory_desc_matches_tag(*pd.src_md(), format_tag::nchw)
            && pd.src_md()->data_type == data_type::f32 && mayiuse_bf16()
            && kai_utils::use_fast_mode(*pd.src_md(), *pd.attr()))
        return data_type::bf16;
    return pd.src_md()->data_type;
}

status_t kai_im2row_convolution_fwd_t::pd_t::init(const engine_t *engine) {
    name_ = "im2row:kai";
    CHECK(init_common(engine));
    if (is_1x1(*this)) name_ = "im2row_1x1:kai";
    init_gemm_args(MB() * OH() * OW(), IC() * KH() * KW(), 1, 1, 1, false);
    // NCHW im2row supports arbitrary patch sizes. The NHWC copy path
    // currently supports only 1x1 patches.
    VDISPATCH_CONV(!src_is_nhwc() || is_1x1(*this),
            "NHWC im2row only supports 1x1 kernels");
    gemm_src_dt_ = src_preparation_dt(*this);
    if (src_is_nhwc()) {
        VDISPATCH_CONV(has_zero_top_left_padding(*this),
                "only supports zero top and left padding");
        VDISPATCH_CONV(source_pixels_in_bounds(*this),
                "requires every 1x1 patch to lie within the source image");
    }
    CHECK(init_gemm(engine));
    // Flattened K cannot contain padding between kernel positions.
    VDISPATCH_CONV(!gemm_args()._fixed_format || is_1x1(*this)
                    || weights_md_.padded_dims[1] == IC(),
            "im2row requires unpadded input channels in fixed-format weights");
    CHECK(init_src_preparation(engine));
    auto scratchpad = scratchpad_registry().registrar();
    book_datapath_scratchpad(scratchpad);
    return status::success;
}

status_t kai_im2row_convolution_fwd_t::pd_t::init_src_preparation(
        const engine_t *engine) {
    const bool kernel_is_1x1 = is_1x1(*this);
    if (!src_is_nhwc()) {
        // We can do im2row using standard reorder for unit-stride 1x1,
        // but more complicated cases require a custom im2row kernel
        src_preparation_.strategy = src_preparation_t::strategy_t::nchw_im2row;
        const size_t src_dt_size = types::data_type_size(src_md()->data_type);
        const size_t dst_dt_size = types::data_type_size(gemm_src_dt_);
        const dim_t row_size = gemm_args()._Ksize;
        for (dim_t kh = 0; kh < KH(); ++kh) {
            for (dim_t kw = 0; kw < KW(); ++kw) {
                const dim_t top = padT() - kh * (KDH() + 1);
                const dim_t left = padL() - kw * (KDW() + 1);
                const dim_t first_h
                        = std::max<dim_t>(0, utils::div_up(top, KSH()));
                const dim_t first_w
                        = std::max<dim_t>(0, utils::div_up(left, KSW()));
                const dim_t last_h
                        = std::min(OH(), utils::div_up(IH() + top, KSH()));
                const dim_t last_w
                        = std::min(OW(), utils::div_up(IW() + left, KSW()));
                src_preparation_.zero_pad |= first_h != 0 || first_w != 0
                        || last_h != OH() || last_w != OW();
                src_preparation_t::region_t region;
                if (last_h > first_h && last_w > first_w) {
                    region.first_h = first_h;
                    region.first_w = first_w;
                    region.src_w = first_w * KSW() - left;
                    region.height = last_h - first_h;
                    region.width = last_w - first_w;
                    region.src_offset = ((first_h * KSH() - top) * IW()
                                                + first_w * KSW() - left)
                            * src_dt_size;
                    region.dst_offset = ((first_h * OW() + first_w) * row_size
                                                + (kh * KW() + kw) * IC())
                            * dst_dt_size;
                }
                src_preparation_.regions.push_back(region);
            }
        }
        if (!kernel_is_1x1 || (KSH() > 1 || KSW() > 1)) return status::success;
        const auto &region = src_preparation_.regions[0];
        if (region.height == 0 || region.width == 0) return status::success;
        src_preparation_.strategy = src_preparation_t::strategy_t::nchw_reorder;
        const dims_t dims = {MB(), IC(), region.height, region.width};
        const dims_t src_strides = {IC() * IH() * IW(), IH() * IW(), IW(), 1};
        const dims_t dst_strides = {OH() * OW() * IC(), 1, OW() * IC(), IC()};
        memory_desc_t reorder_src_md, reorder_dst_md;
        CHECK(memory_desc_init_by_strides(
                reorder_src_md, 4, dims, src_md()->data_type, src_strides));
        CHECK(memory_desc_init_by_strides(
                reorder_dst_md, 4, dims, gemm_src_dt_, dst_strides));
        VDISPATCH_CONV_SC(
                reorder_primitive_desc_create(src_preparation_.reorder_pd,
                        engine, &reorder_src_md, &reorder_dst_md),
                VERBOSE_PRIMITIVE_CREATION_FAIL, "NCHW im2row reorder");
        return status::success;
    }
    src_preparation_.strategy = src_preparation_t::strategy_t::nhwc_copy;
    return status::success;
}

void kai_im2row_convolution_fwd_t::pd_t::book_datapath_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    if (src_preparation_.reorder_pd)
        scratchpad.book(memory_tracking::names::key_nested_multiple
                        + attr()->post_ops_.len(),
                src_preparation_.reorder_pd->scratchpad_registry());
    const size_t bytes
            = im2row_size_bytes(*this, types::data_type_size(gemm_src_dt_));
    scratchpad.book<char>(memory_tracking::names::key_conv_gemm_col, bytes);
}

status_t kai_im2row_convolution_fwd_t::init(engine_t *engine) {
    CHECK(init_common(engine));
    const auto *pd = static_cast<const pd_t *>(this->pd());
    using strategy_t = pd_t::src_preparation_t::strategy_t;
    switch (pd->src_preparation_.strategy) {
        case strategy_t::nchw_reorder:
            CHECK(pd->src_preparation_.reorder_pd->create_primitive(
                    nchw_reorder_, engine));
            break;
        case strategy_t::nchw_im2row:
            nchw_im2row_jit_kernel_ = std::make_shared<jit_kai_nchw_im2row_t>(
                    types::data_type_size(pd->src_md()->data_type),
                    types::data_type_size(pd->gemm_src_dt_),
                    pd->IH() * pd->IW(), pd->KSW(), pd->gemm_args()._Ksize);
            CHECK(nchw_im2row_jit_kernel_->create_kernel());
            break;
        case strategy_t::nhwc_copy: break;
    }
    return status::success;
}

void kai_im2row_convolution_fwd_t::copy_nhwc_im2row(
        const kernel_call_args_t &args, char *im2row) const {
    const auto &pd
            = static_cast<const kai_im2row_convolution_fwd_t::pd_t &>(args.pd);
    const dim_t MB = pd.MB();
    const dim_t IC = pd.IC();
    const dim_t OH = pd.OH();
    const dim_t OW = pd.OW();
    const dim_t KSH = pd.KSH();
    const dim_t KSW = pd.KSW();
    const size_t src_dt_size = types::data_type_size(pd.src_md()->data_type);
    const size_t row_bytes = static_cast<size_t>(IC) * src_dt_size;
    const dim_t rows = MB * OH * OW;
    const dim_t batch_rows = OH * OW;
    const size_t src_pixel_stride_bytes = static_cast<size_t>(IC) * src_dt_size;
    const size_t src_scanline_stride_bytes = pd.IW() * src_pixel_stride_bytes;
    const size_t src_image_stride_bytes = pd.IH() * src_scanline_stride_bytes;
    const size_t src_h_step_bytes
            = static_cast<size_t>(KSH) * src_scanline_stride_bytes;
    const size_t src_ow_step_bytes
            = static_cast<size_t>(KSW) * src_pixel_stride_bytes;
    const char *src_base = args.src_base;
    // Limit copying workers by scanline count, including the batch dimension.
    const int active_nthr = static_cast<int>(
            std::min<dim_t>(dnnl_get_current_num_threads(), MB * OH));

    // Keep the OpenMP team size stable across im2row and GEMM while limiting
    // the number of threads that do copy work.
    const int team_nthr = active_nthr > 1 ? dnnl_get_current_num_threads() : 1;

    parallel(team_nthr, [=](int ithr, int) {
        if (ithr >= active_nthr) return;

        dim_t start = 0;
        dim_t end = 0;
        balance211(rows, active_nthr, ithr, start, end);

        dim_t mb = start / batch_rows;
        dim_t rem = start - mb * batch_rows;
        dim_t oh = rem / OW;
        dim_t ow = rem - oh * OW;
        dim_t row = start;
        char *dst = im2row + static_cast<size_t>(row) * row_bytes;
        const char *src = src_base
                + static_cast<size_t>(mb) * src_image_stride_bytes
                + static_cast<size_t>(oh) * src_h_step_bytes
                + static_cast<size_t>(ow) * src_ow_step_bytes;
        // For 1x1 with unit horizontal stride, consecutive patches in an
        // output row occupy contiguous NHWC storage. Copy each run into the
        // im2row buffer.
        const bool contiguous_width = KSW == 1;

        while (row < end) {
            const dim_t width_run = std::min(end - row, OW - ow);
            if (contiguous_width) {
                const size_t bytes = static_cast<size_t>(width_run) * row_bytes;
                std::memcpy(dst, src, bytes);
                src += static_cast<size_t>(width_run) * src_pixel_stride_bytes;
                dst += bytes;
            } else {
                copy_strided_width(
                        dst, src, width_run, row_bytes, src_ow_step_bytes);
            }
            row += width_run;
            ow += width_run;
            if (ow == OW && row < end) {
                ow = 0;
                if (++oh == OH) {
                    oh = 0;
                    ++mb;
                }
                src = src_base
                        + static_cast<size_t>(mb) * src_image_stride_bytes
                        + static_cast<size_t>(oh) * src_h_step_bytes;
            }
        }
    });
}

status_t kai_im2row_convolution_fwd_t::prepare_src(
        const kernel_call_args_t &args, char *im2row) const {
    const auto &pd = static_cast<const pd_t &>(args.pd);
    const auto &prep = pd.src_preparation_;
    const size_t src_dt_size = types::data_type_size(pd.src_md()->data_type);
    using strategy_t = pd_t::src_preparation_t::strategy_t;
    switch (prep.strategy) {
        case strategy_t::nhwc_copy:
            copy_nhwc_im2row(args, im2row);
            return status::success;
        case strategy_t::nchw_reorder: {
            if (prep.zero_pad)
                parallel_nd(pd.MB() * pd.OH(), [&](dim_t row) {
                    const size_t bytes = pd.OW() * pd.IC()
                            * types::data_type_size(pd.gemm_src_dt_);
                    std::memset(im2row + row * bytes, 0, bytes);
                });
            const auto &region = prep.regions[0];
            return execute_reorder(args.ctx, *nchw_reorder_,
                    prep.reorder_pd->src_md(), prep.reorder_pd->dst_md(),
                    args.src_base + region.src_offset,
                    im2row + region.dst_offset,
                    memory_tracking::names::key_nested_multiple
                            + pd.attr()->post_ops_.len());
        }
        case strategy_t::nchw_im2row: break;
    }

    const size_t dst_dt_size = types::data_type_size(pd.gemm_src_dt_);
    const dim_t row_size = pd.gemm_args()._Ksize;

    const dim_t mb_count = pd.MB(), ih = pd.IH(), iw = pd.IW();
    const dim_t ic = pd.IC(), oh = pd.OH(), ow = pd.OW();
    const dim_t sh = pd.KSH(), sw = pd.KSW();
    // Assign all kernel positions of an output tile to the same
    // thread, keeping im2row writes local and reusing input channels.
    parallel_nd(mb_count, oh, utils::div_up(ow, 4), utils::div_up(ic, 32),
            [&](dim_t mb, dim_t h, dim_t wb, dim_t cb) {
        const dim_t c = cb * 32;
        const dim_t width = std::min<dim_t>(4, ow - wb * 4);
        const dim_t channels = std::min<dim_t>(32, ic - c);
        for (size_t k = 0; k < prep.regions.size(); ++k) {
            const auto &region = prep.regions[k];
            const bool valid_h
                    = h >= region.first_h && h < region.first_h + region.height;
            if (prep.zero_pad) {
                for (dim_t w = wb * 4; w < wb * 4 + width; ++w) {
                    if (valid_h && w >= region.first_w
                            && w < region.first_w + region.width)
                        continue;
                    auto *dst = im2row
                            + (((mb * oh + h) * ow + w) * row_size + k * ic + c)
                                    * dst_dt_size;
                    std::memset(dst, 0, channels * dst_dt_size);
                }
            }
            if (!valid_h) continue;
            const dim_t start_w = std::max(wb * 4, region.first_w);
            const dim_t end_w = std::min(
                    std::min(wb * 4 + 4, ow), region.first_w + region.width);
            if (start_w >= end_w) continue;
            const dim_t rh = h - region.first_h;
            const dim_t rw = start_w - region.first_w;
            const auto *src = args.src_base + region.src_offset
                    + ((mb * ic + c) * ih * iw + rh * sh * iw + rw * sw)
                            * src_dt_size;
            auto *dst = im2row + region.dst_offset
                    + (((mb * oh + rh) * ow + rw) * row_size + c) * dst_dt_size;
            (*nchw_im2row_jit_kernel_)({src, dst, channels, end_w - start_w,
                    iw - region.src_w - rw * sw});
        }
    });

    return status::success;
}

status_t kai_im2row_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;
    kernel_call_args_t args {ctx, *pd(), *kernel};
    CHECK(prepare(args));
    CHECK(setup_kernel_arrays(args));
    return execute_kernel(args);
}

status_t kai_im2row_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd = static_cast<const pd_t &>(args.pd);
    auto *im2row = args.ctx.get_scratchpad_grantor().get<char>(
            memory_tracking::names::key_conv_gemm_col);
    CHECK(prepare_src(args, im2row));
    const void *A = im2row;
    const int lda = static_cast<int>(pd.gemm_args()._Ksize);
    const int A_batch_stride = 0;
    const int A_multi_stride = 0;
    const int C_batch_stride = 0;
    const int C_multi_stride = 0;

    args.kernel.set_arrays_generic(A, lda, A_batch_stride, A_multi_stride,
            args.B(), args.ldb(), args.B_multi_stride(), args.C(), args.ldc(),
            C_batch_stride, C_multi_stride, args.bias(),
            args.bias_multi_stride());
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
