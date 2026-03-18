/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

void jit_generator_t::transpose(const Xbyak::Reg64 &reg_src,
        const Xbyak::Reg64 &reg_dst, dim_t src_stride, dim_t dst_stride,
        int nrows, int ncolumns, data_type_t dt, Xbyak::Ymm &ymm_tmp,
        Xbyak::Ymm &ymm_mask, Xbyak::Xmm &xmm_upper_mask) {
    // no row padding for dst, so no work needed to be done
    if (ncolumns == 0) return;

    // Note: For stores we assume, the memory is padded, hence avoiding use of
    // mask stores.
    const auto xmm_lower_mask = Xbyak::Xmm(ymm_mask.getIdx());
    const auto xmm_tmp = Xbyak::Xmm(ymm_tmp.getIdx());

    // Only avx2 version is supported for now. TODO: support other cases.
    // The transpose size is always calculated for f32 because bf16 is
    // is supported via upconversion.
    const int transpose_size = vreg_traits_t<Xbyak::Ymm>::vlen
            / types::data_type_size(data_type::f32);
    assert(is_valid_isa(avx2));
    assert(nrows <= transpose_size && ncolumns <= transpose_size);

    assert(utils::one_of(dt, data_type::f32, data_type::bf16, data_type::f16));

    if (transpose_size > nrows) uni_vxorps(ymm_tmp, ymm_tmp, ymm_tmp);

    // Load up to 4 f32 values into an XMM register.
    // If the row is out of range then load zeros.
    // If the row has a tail then use a mask load.
    auto load_src_f32 = [= COMPAT_THIS_CAPTURE](Xbyak::Xmm vmm, int r, int c) {
        // Number of f32 values that fit in the XMM register.
        const int simd_w = 4;
        const auto addr = ptr[reg_src + r * src_stride
                + c * types::data_type_size(data_type::f32)];

        if (r >= nrows) {
            uni_vxorps(vmm, vmm, vmm);
        } else if (c + simd_w <= ncolumns) {
            vmovups(vmm, addr);
        } else if (c == 0) {
            vmaskmovps(vmm, xmm_lower_mask, addr);
        } else {
            vmaskmovps(vmm, xmm_upper_mask, addr);
        }
    };

    // Load up to 4 bf16 values and convert them to f32.
    auto load_src_xf16 = [= COMPAT_THIS_CAPTURE](Xbyak::Xmm vmm, int r, int c) {
        if (r >= nrows) {
            vpxor(vmm, vmm, vmm);
            return;
        }

        vpxor(xmm_tmp, xmm_tmp, xmm_tmp);

        const int simd_w = 4;
        const int rem = ncolumns - c;
        const dim_t dt_size = 2;

        if (rem >= simd_w) {
            // Load 4 x xf16
            vmovq(xmm_tmp, ptr[reg_src + r * src_stride + c * dt_size]);
        } else {
            // Load the remaining xf16 values one by one.
            for (int i = 0; i < rem; i++) {
                vpinsrw(xmm_tmp, xmm_tmp,
                        ptr[reg_src + r * src_stride + (c + i) * dt_size], i);
            }
        }

        if (dt == data_type::bf16) {
            // Upconvert 4x16-bit values to 4x32-bit values.
            vpmovzxwd(vmm, xmm_tmp);
            // Shift bf16 bits into the upper 16 bits.
            vpslld(vmm, vmm, 16);
        } else if (dt == data_type::f16) {
            // Convert 4 x f16 to 4 x f32
            vcvtph2ps(vmm, xmm_tmp);
        } else {
            assert(!"unsupported data type");
        }
    };

    // Choose the load path based on source data type.
    auto load_src = [= COMPAT_THIS_CAPTURE](Xbyak::Xmm vmm, int r, int c) {
        if (dt == data_type::f32) {
            load_src_f32(vmm, r, c);
        } else if (utils::one_of(dt, data_type::bf16, data_type::f16)) {
            load_src_xf16(vmm, r, c);
        } else {
            assert(!"unsupported data type");
        }
    };

    // Fill the upper 128-bit half of the YMM register for the f32 path.
    // The lower half was already loaded by load_src_f32().
    auto vinsert_f32 = [= COMPAT_THIS_CAPTURE](Xbyak::Ymm ymm, int r, int c) {
        const int xmm_simd_w = 4;
        const auto addr = ptr[reg_src + r * src_stride
                + c * types::data_type_size(data_type::f32)];

        if (r >= nrows) {
            // Upper half zeroed.
            vperm2i128(ymm, ymm, ymm_tmp, 0x30);
        } else if (c + xmm_simd_w <= ncolumns) {
            vinsertf128(ymm, ymm, addr, 1);
        } else {
            vmaskmovps(xmm_tmp, c == 0 ? xmm_lower_mask : xmm_upper_mask, addr);
            vinsertf128(ymm, ymm, xmm_tmp, 1);
        }
    };

    // Fill the upper 128-bit half of the YMM register for the xf16 path.
    // Values are first loaded and upconverted to f32.
    auto vinsert_xf16 = [= COMPAT_THIS_CAPTURE](Xbyak::Ymm ymm, int r, int c) {
        if (r >= nrows) {
            vperm2i128(ymm, ymm, ymm_tmp, 0x30);
        } else {
            load_src_xf16(xmm_tmp, r, c);
            vinsertf128(ymm, ymm, xmm_tmp, 1);
        }
    };

    // Choose the upper-half insert path based on source data.
    auto vinsert = [= COMPAT_THIS_CAPTURE](Xbyak::Ymm ymm, int r, int c) {
        if (dt == data_type::f32) {
            vinsert_f32(ymm, r, c);
        } else if (utils::one_of(dt, data_type::bf16, data_type::f16)) {
            vinsert_xf16(ymm, r, c);
        } else {
            assert(!"unsupported data type");
        }
    };

    // Store one full transposed f32 column.
    auto store_dst_f32 = [= COMPAT_THIS_CAPTURE](int col, Xbyak::Ymm ymm) {
        vmovups(ptr[reg_dst + col * dst_stride], ymm);
    };

    // Convert f32 values back to bf16 and store one transposed column.
    auto store_dst_bf16 = [= COMPAT_THIS_CAPTURE](int col, Xbyak::Ymm ymm) {
        // Reuse mask register as a temporary XMM register in the bf16 path.
        auto xmm_store_tmp = xmm_lower_mask;
        // Shift bf16 bits back into the lower 16 bits.
        vpsrld(ymm_tmp, ymm, 16);
        // Pack 8x32-bit values into 8x16-bit values.
        vextracti128(xmm_store_tmp, ymm_tmp, 1);
        vpackusdw(xmm_store_tmp, xmm_tmp, xmm_store_tmp);

        vmovdqu(ptr[reg_dst + col * dst_stride], xmm_store_tmp);
    };

    // Convert f32 values back to f16 and store one transposed column.
    auto store_dst_f16 = [= COMPAT_THIS_CAPTURE](int col, Xbyak::Ymm ymm) {
        Xbyak::Xmm xmm_lo = xmm_tmp;
        Xbyak::Xmm xmm_hi = xmm_lower_mask; // reuse as temp
        Xbyak::Xmm xmm_src_hi = xmm_upper_mask; // reuse as temp

        // Convert lower 4 f32 values to f16
        vcvtps2ph(xmm_lo, Xbyak::Xmm(ymm.getIdx()), 0);

        // Convert upper 4 f32 values to f16
        vextractf128(xmm_src_hi, ymm, 1);
        vcvtps2ph(xmm_hi, xmm_src_hi, 0);

        // Combine two 64-bit f16 halves into one Xmm and store 8 f16 values.
        punpcklqdq(xmm_lo, xmm_hi);
        vmovdqu(ptr[reg_dst + col * dst_stride], xmm_lo);
    };

    // Choose the store path based on source data type.
    auto store_dst = [= COMPAT_THIS_CAPTURE](int col, Xbyak::Ymm ymm) {
        if (dt == data_type::f32) {
            store_dst_f32(col, ymm);
        } else if (dt == data_type::bf16) {
            store_dst_bf16(col, ymm);
        } else if (dt == data_type::f16) {
            store_dst_f16(col, ymm);
        } else {
            assert(!"unsupported data type");
        }
    };

    // Intel(R) Software Optimization manual
    // Example 15-20. 8x8 Matrix Transpose Using VINSERTPS
    auto transpose_8x4 = [= COMPAT_THIS_CAPTURE](int col) {
        load_src(xmm0, 0, col);
        vinsert(ymm0, 4, col);
        load_src(xmm1, 1, col);
        vinsert(ymm1, 5, col);
        vunpcklpd(ymm8, ymm0, ymm1);
        vunpckhpd(ymm9, ymm0, ymm1);

        load_src(xmm2, 2, col);
        vinsert(ymm2, 6, col);
        load_src(xmm3, 3, col);
        vinsert(ymm3, 7, col);
        vunpcklpd(ymm10, ymm2, ymm3);
        vunpckhpd(ymm11, ymm2, ymm3);

        vshufps(ymm4, ymm8, ymm10, 0x88);
        store_dst(col, ymm4);

        if (col + 1 < ncolumns) {
            vshufps(ymm5, ymm8, ymm10, 0xDD);
            store_dst(col + 1, ymm5);
        }

        if (col + 2 < ncolumns) {
            vshufps(ymm6, ymm9, ymm11, 0x88);
            store_dst(col + 2, ymm6);
        }

        if (col + 3 < ncolumns) {
            vshufps(ymm7, ymm9, ymm11, 0xDD);
            store_dst(col + 3, ymm7);
        }
    };

    // First handle columns 0..3
    transpose_8x4(0);
    // If needed handle columns 4..7
    if (ncolumns > 4) transpose_8x4(4);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
