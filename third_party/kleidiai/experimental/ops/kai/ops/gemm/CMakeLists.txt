#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
message(STATUS "Entered gemm directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_GEMM_CORE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/gemm_bf16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_bf16bf16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_fp16fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_int16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_int8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_q8_mixed.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_qint8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_quint8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_s8fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_s8fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_u8s8fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_uint16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_uint8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/interleave_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mergeresults.cpp
    ${CMAKE_CURRENT_LIST_DIR}/misc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quantized.cpp
    ${CMAKE_CURRENT_LIST_DIR}/row_sums/a64_sumrows_s8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/row_sums/a64_sumrows_u8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transform.cpp)

set(KLEIDIAI_OPS_GEMM_FP16_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/quantized-fp16.cpp)

# -------------------------------------------------------------------------------------------------

# Add extracted kernel .S files to this shared assembly list. All assembly
# sources use the combined SVE architecture settings; enabling these extensions
# does not insert additional instructions into functions that do not use them.
set(KLEIDIAI_OPS_GEMM_SOURCES_ASM
    ${CMAKE_CURRENT_LIST_DIR}/row_sums/a64_sumrows_s8_asm.S
    ${CMAKE_CURRENT_LIST_DIR}/row_sums/a64_sumrows_u8_asm.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_bf16_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_fp16_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_fp32_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_s16_s16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_s8_s16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_s8_s16_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_u8_u16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block1_u8_u16_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block2_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block2_fp32_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block4_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block4_fp32_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block4_s8_s8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block4_s8_s8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block4_u8_u8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block8_s8_s8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block8_s8_s8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/a64_interleave8_block8_u8_u8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block2_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block2_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block2_fp32_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block4_s8_s8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block4_s8_s8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block4_u8_u8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_block4_u8_u8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave1VL_fp32_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block2_bf16_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block2_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block2_fp32_bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block4_s8_s8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block4_s8_s8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block4_u8_u8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_block4_u8_u8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave2VL_fp32_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_block2_fp16_fp16.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_block4_s8_s8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_block4_s8_s8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_block4_u8_u8.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_block4_u8_u8_summing.S
    ${CMAKE_CURRENT_LIST_DIR}/indirect-interleaves/sme_interleave4VL_fp32_fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_bf16fp32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16_mla_6x32.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16fp32_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16fp32fp16_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32bf16fp32_mmla_4x24.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32bf16fp32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_bf16fp32_dot_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_fp16_mla_8x24.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_fp32_mla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_bf16fp32_dot_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_bf16fp32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_bf16fp32bf16_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16_mla_6x32.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16_mla_6x32_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16fp32_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16fp32fp16_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_4x24.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_4x24_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_6x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_8x4.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_8x4_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32bf16fp32_mmla_4x24.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32bf16fp32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_dot_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_dot_4x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_mmla_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_dot_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_dot_6x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_dot_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_dot_6x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_dot_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_dot_4x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_mmla_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8qa_dot_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8qa_mmla_4x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8s32_dot_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8s32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_dot_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_dot_6x16_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_mmla_6x16.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_dot_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_mmla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_mmla_8x12_a510.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp16_mla_8x24.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp16_mla_8x24_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp16_mla_8x24_x1.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp32_mla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp32_mla_8x12_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_fp32_mla_8x12_x1.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_nomerge_fp32_mla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s16s32_mla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_dot_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_dot_8x12_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_dot_8x12_x1.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_mmla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_mmla_8x12_a510.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u16u32_mla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8s8s32_mmla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_dot_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_dot_8x12_a55.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_dot_8x12_x1.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_mmla_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_mmla_8x12_a510.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_bf16_mla_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_bf16fp32_dot_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp16_mla_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp16fp32fp16_dot_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp32_mla_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp32bf16fp32_dot_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_s8qa_dot_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_u8qa_dot_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16_mopa_1VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16_mopa_2VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL_lhs2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16_mopa_1VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16_mopa_2VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_bf16fp32_dot_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp16_mla_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp32_mla_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp32bf16fp32_dot_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_s8qa_dot_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_u8qa_dot_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL_lhs2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_1VLx4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_2VLx2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_4VLx1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_bf16fp32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16_mla_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16fp32_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16fp32fp16_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32_mla_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32bf16fp32_mmla_4x6VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_bf16fp32_dot_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp16_mla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp16_mla_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp32_mla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp32_mla_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_bf16fp32_dot_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_bf16fp32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16_mla_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16fp32_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16fp32fp16_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_8x1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_8x1VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32bf16fp32_mmla_4x6VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32bf16fp32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qa_dot_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qa_mmla_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qs_dot_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qs_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_dot_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_dot_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8qa_dot_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8qa_mmla_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8qa_dot_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8qa_mmla_4x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8s32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_dot_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_dot_6x4VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_mmla_6x4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_bf16fp32_dot_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_bf16fp32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp16_mla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp16_mla_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mla_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_dot_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_dot_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8s8s32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_dot_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_dot_8x3VL_a64fx.S
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_mmla_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_fp16_fp16_8x24.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_fp32_bf16_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_fp32_fp16_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_fp32_fp32_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_s32_s32_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/a64_merge_u32_u32_8x12.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/sve_merge_fp16_fp16_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/sve_merge_fp32_bf16_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/sve_merge_fp32_fp32_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/sve_merge_s32_s32_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/merges/sve_merge_u32_u32_8x3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_2_1x8.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_2_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_4_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_4_4x1_bf16fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_4_4x1_fp16fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_interleave_8_2x1.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_12_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_12_1x8.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_12_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_12_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_12_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16_1x8.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_16_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_24.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_24_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_24_s8s16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_24_u8u16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_32.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_48.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_48_bf16fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_48_fp16fp32.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_4_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_4_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_4_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_64.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/a64_transpose_interleave_96.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_interleave_1VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_interleave_1VL_2x1.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_interleave_1VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_interleave_2VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_interleave_2VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_16VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_16VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_16VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_16VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_1VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_1VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_1VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_2VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_2VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_2VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_4VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_4VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_4VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_8VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_8VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_8VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sme_transpose_interleave_8VL_2x2_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_12VL_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_1VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_1VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_2VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_2VL_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_2VL_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_3VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_3VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_3VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_4VL.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_4VL_1x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_4VL_2x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_6VL_1x8.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_6VL_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_6VL_2x4_fp32bf16.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_6VL_4x2.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_8VL_1x8.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_8VL_2x4.S
    ${CMAKE_CURRENT_LIST_DIR}/transforms/sve_transpose_interleave_8VL_2x4_fp32bf16.S
)


# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_GEMM_PRIVATE_INCLUDES
    # Internal sources
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../
    # Public headers
    ${CMAKE_CURRENT_LIST_DIR}/../../../include)

# -------------------------------------------------------------------------------------------------

# Helper to create an OBJECT lib, apply common compile options/compile definitions/include
# directories, and arch
function(arm_add_obj_lib OBJ_NAME)
    # Parse flags
    set(PREFIX ARG)
    set(OPTIONS)
    set(SINGLE_VALUE_ARGS ARCH)
    set(MULTI_VALUE_ARGS SOURCES)
    cmake_parse_arguments(
        ${PREFIX}
        "${OPTIONS}"
        "${SINGLE_VALUE_ARGS}"
        "${MULTI_VALUE_ARGS}"
        ${ARGN})

    # Sanity check
    if(NOT ${PREFIX}_SOURCES)
        message(FATAL_ERROR "No SOURCES provided for '${OBJ_NAME}'")
    endif()

    # Define the object library
    add_library(${OBJ_NAME} OBJECT ${${PREFIX}_SOURCES})
    target_include_directories(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_GEMM_PRIVATE_INCLUDES})
    target_compile_definitions(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
    # Keep C/C++ options scoped to these languages. Extracted assembly is compiled
    # separately with the combined SVE architecture settings below.
    target_compile_options(
        ${OBJ_NAME} PRIVATE
        "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${KLEIDIAI_OPS_CCXX_FLAGS}>")
    if(${PREFIX}_ARCH)
        target_compile_options(${OBJ_NAME} PRIVATE ${${PREFIX}_ARCH})
    else()
        message(STATUS "Architecture flag unset for '${OBJ_NAME}'")
    endif()
    set_target_properties(
        ${OBJ_NAME}
        PROPERTIES C_STANDARD ${KLEIDIAI_OPS_C_STANDARD}
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS OFF
                   CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   POSITION_INDEPENDENT_CODE ON)
endfunction()

# -------------------------------------------------------------------------------------------------

# Define object libs, each with their appropriate architecture flags

# GEMM sources; non-SVE, non-FP16
arm_add_obj_lib(
    gemm_core
    SOURCES ${KLEIDIAI_OPS_GEMM_CORE_SOURCES}
    ARCH ${KLEIDIAI_OPS_ARCH})

# GEMM sources; AArch64-only buckets
arm_add_obj_lib(
    gemm_fp16
    SOURCES ${KLEIDIAI_OPS_GEMM_FP16_SOURCES}
    ARCH ${KLEIDIAI_OPS_FP16_ARCH})

# -------------------------------------------------------------------------------------------------

# Source properties are set beside the target that compiles these files.
# No object library is needed until this directory contains assembly sources.
if(KLEIDIAI_OPS_GEMM_SOURCES_ASM)
    if(MSVC)
        set_source_files_properties(${KLEIDIAI_OPS_GEMM_SOURCES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
    endif()

    add_library(gemm_asm OBJECT ${KLEIDIAI_OPS_GEMM_SOURCES_ASM})
    target_include_directories(gemm_asm PRIVATE ${KLEIDIAI_OPS_GEMM_PRIVATE_INCLUDES})
    target_compile_definitions(gemm_asm PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
    if(NOT MSVC)
        target_compile_options(gemm_asm PRIVATE ${KLEIDIAI_OPS_SVE_ARCH})
    endif()
    set_target_properties(gemm_asm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(kleidiai_ops PRIVATE $<TARGET_OBJECTS:gemm_asm>)
endif()

# Add the compiled GEMM objects to the final kleidiai_ops library.
target_sources(
    kleidiai_ops
    PRIVATE $<TARGET_OBJECTS:gemm_core>
            $<$<TARGET_EXISTS:gemm_fp16>:$<TARGET_OBJECTS:gemm_fp16>>)

message(STATUS "Leaving gemm directory")
# -------------------------------------------------------------------------------------------------
