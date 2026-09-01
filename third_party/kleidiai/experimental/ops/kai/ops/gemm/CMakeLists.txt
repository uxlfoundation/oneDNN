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
    ${CMAKE_CURRENT_LIST_DIR}/interleave-8way.cpp
    ${CMAKE_CURRENT_LIST_DIR}/interleave_indirect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a32_sgemm_8x6/a53.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a32_sgemm_8x6/a55r1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a32_sgemm_8x6/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_bf16fp32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32bf16fp32_mmla_4x24/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp32bf16fp32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_bf16fp32_dot_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_bf16fp32_mmla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_fp32_mla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_s16_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_s8_4x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_s8_8x12/a55r1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_s8_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_s8_8x12/x1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_u16_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_u8_4x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_u8_8x12/a55r1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_u8_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_gemm_u8_8x12/x1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_bf16fp32_dot_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_bf16fp32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_4x24/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_4x24/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_6x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_8x4/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32_mla_8x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32bf16fp32_mmla_4x24/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp32bf16fp32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_dot_4x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_dot_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qa_mmla_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_dot_6x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_dot_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8qs_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_dot_6x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_dot_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_s8s32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_dot_4x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_dot_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8qa_mmla_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8qa_dot_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8qa_mmla_4x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8s32_dot_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8s8s32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_dot_6x16/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_dot_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_u8u32_mmla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_dot_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_mmla_8x12/a510.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_bf16fp32_mmla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_mmla_8x12/a510.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_s8s32_mmla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8s8s32_mmla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_mmla_8x12/a510.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_interleaved_u8u32_mmla_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x12/a53.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x12/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x12/a55r1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x12/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x12/x1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_sgemm_8x6/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_fp32_mla_6x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_fp32_mla_8x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_s8s32_dot_6x4/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_s8s32_dot_6x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_s8s32_dot_8x4/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_s8s32_dot_8x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_u8u32_dot_6x4/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_u8u32_dot_6x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_u8u32_dot_8x4/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_smallK_hybrid_u8u32_dot_8x4/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mergeresults.cpp
    ${CMAKE_CURRENT_LIST_DIR}/misc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quantized.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rowsum_indirect_s8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rowsum_indirect_u8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transform.cpp)

set(KLEIDIAI_OPS_GEMM_SVE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/interleave_indirect-sve.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_bf16_mla_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_bf16fp32_dot_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp16_mla_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp16fp32fp16_dot_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp32_mla_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_fp32bf16fp32_dot_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_s8qa_dot_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_gemv_u8qa_dot_16VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16_mopa_1VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16_mopa_2VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16_mopa_1VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16_mopa_2VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8q_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_s8s32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme2_interleaved_nomerge_u8q_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_bf16fp32_dot_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp16_mla_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp16fp32fp16_mla_8VL_rhs2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp32_mla_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_fp32bf16fp32_dot_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_s8qa_dot_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_gemv_u8qa_dot_8VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp16fp32fp16_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_fp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8q_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8qfp32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_s8s32_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_1VLx4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_2VLx2VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sme_interleaved_nomerge_u8q_mopa_4VLx1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_bf16fp32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16_mla_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16fp32_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp16fp32fp16_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32_mla_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffhybrid_fp32bf16fp32_mmla_4x6VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_bf16fp32_dot_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp16_mla_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp16_mla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp32_mla_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_ffinterleaved_fp32_mla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_bf16fp32_dot_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_bf16fp32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16_mla_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16fp32_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp16fp32fp16_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_8x1VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32_mla_8x1VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32bf16fp32_mmla_4x6VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_fp32bf16fp32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qa_dot_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qa_mmla_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qs_dot_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8qs_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_dot_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_dot_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_s8s32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8qa_dot_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8qa_mmla_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8qa_dot_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8qa_mmla_4x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8s8s32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_dot_6x4VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_dot_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_hybrid_u8u32_mmla_6x4VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_bf16fp32_dot_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_bf16fp32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp16_mla_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp16_mla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mla_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_fp32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_dot_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_dot_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_s8s32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8s8s32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_dot_8x3VL/a64fx.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_dot_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/sve_interleaved_u8u32_mmla_8x3VL/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mergeresults-sve.cpp
    ${CMAKE_CURRENT_LIST_DIR}/transform-sve.cpp)

set(KLEIDIAI_OPS_GEMM_FP16_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16_mla_6x32/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16fp32_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffhybrid_fp16fp32fp16_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_ffinterleaved_fp16_mla_8x24/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hgemm_8x24/a55r1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hgemm_8x24/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hgemm_8x24/x1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16_mla_6x32/a55.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16_mla_6x32/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16fp32_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kernels/a64_hybrid_fp16fp32fp16_mla_6x16/generic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quantized-fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mergeresults-fp16.cpp)

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_GEMM_PRIVATE_INCLUDES
    # Internal sources
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../
    # Public headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)

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
    target_compile_options(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_CCXX_FLAGS})
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
if(NOT KLEIDIAI_OPS_ENABLE_AARCH32)
    arm_add_obj_lib(
        gemm_sve
        SOURCES ${KLEIDIAI_OPS_GEMM_SVE_SOURCES}
        ARCH ${KLEIDIAI_OPS_SVE_ARCH})

    arm_add_obj_lib(
        gemm_fp16
        SOURCES ${KLEIDIAI_OPS_GEMM_FP16_SOURCES}
        ARCH ${KLEIDIAI_OPS_FP16_ARCH})
endif()

# -------------------------------------------------------------------------------------------------

# Append the sources to the single kleidiai_ops project target
target_sources(
    kleidiai_ops
    PRIVATE $<TARGET_OBJECTS:gemm_core> $<$<TARGET_EXISTS:gemm_sve>:$<TARGET_OBJECTS:gemm_sve>>
            $<$<TARGET_EXISTS:gemm_fp16>:$<TARGET_OBJECTS:gemm_fp16>>)

message(STATUS "Leaving gemm directory")
# -------------------------------------------------------------------------------------------------
