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

#ifndef GPU_INTEL_INCLUDE_CONFIG_H
#define GPU_INTEL_INCLUDE_CONFIG_H

// Data type presence flags. These are set based on the -D flags passed to the
// kernel compiler and control which conversion/IO overloads get instantiated.

#if DT_BF8 || SRC_DT_BF8 || WEI_DT_BF8 || DST_DT_BF8 || BIA_DT_BF8 || A_DT_BF8 \
        || B_DT_BF8 || C_DT_BF8 || DATA_DT_BF8 || POST_OP_USING_BF8 \
        || SRC_SCALES_DT_BF8 || WEI_SCALES_DT_BF8 || DST_SCALES_DT_BF8 \
        || BIAS_DT_BF8
#define MATH_UTILS_DECLARE_BF8 1
#endif

#if DT_HF8 || SRC_DT_HF8 || WEI_DT_HF8 || DST_DT_HF8 || BIA_DT_HF8 || A_DT_HF8 \
        || B_DT_HF8 || C_DT_HF8 || DATA_DT_HF8 || POST_OP_USING_HF8 \
        || SRC_SCALES_DT_HF8 || WEI_SCALES_DT_HF8 || DST_SCALES_DT_HF8 \
        || BIAS_DT_HF8
#define MATH_UTILS_DECLARE_HF8 1
#endif

#if DT_F4_E2M1 || SRC_DT_F4_E2M1 || WEI_DT_F4_E2M1 || DST_DT_F4_E2M1 \
        || BIA_DT_F4_E2M1 || A_DT_F4_E2M1 || B_DT_F4_E2M1 || C_DT_F4_E2M1 \
        || DATA_DT_F4_E2M1 || POST_OP_USING_F4_E2M1 || BIAS_DT_F4_E2M1
#define MATH_UTILS_DECLARE_F4_E2M1 1
#endif

#if DT_F4_E3M0 || SRC_DT_F4_E3M0 || WEI_DT_F4_E3M0 || DST_DT_F4_E3M0 \
        || BIA_DT_F4_E3M0 || A_DT_F4_E3M0 || B_DT_F4_E3M0 || C_DT_F4_E3M0 \
        || DATA_DT_F4_E3M0 || POST_OP_USING_F4_E3M0 || BIAS_DT_F4_E3M0
#define MATH_UTILS_DECLARE_F4_E3M0 1
#endif

#if DT_S4 || SRC_DT_S4 || WEI_DT_S4 || DST_DT_S4 || BIA_DT_S4 || A_DT_S4 \
        || B_DT_S4 || C_DT_S4 || DATA_DT_S4 || WEI_ZP_DT_S4 || SRC_ZP_DT_S4
#define MATH_UTILS_DECLARE_S4 1
#endif

#if DT_U4 || SRC_DT_U4 || WEI_DT_U4 || DST_DT_U4 || BIA_DT_U4 || A_DT_U4 \
        || B_DT_U4 || C_DT_U4 || DATA_DT_U4 || WEI_ZP_DT_U4 || SRC_ZP_DT_U4
#define MATH_UTILS_DECLARE_U4 1
#endif

#if DT_BF16 || SRC_DT_BF16 || WEI_DT_BF16 || DST_DT_BF16 || BIA_DT_BF16 \
        || A_DT_BF16 || B_DT_BF16 || C_DT_BF16 || SUM_DT_BF16 || DATA_DT_BF16 \
        || POST_OP_USING_BF16 || SRC_SCALES_DT_BF16 || WEI_SCALES_DT_BF16 \
        || DST_SCALES_DT_BF16 || BIAS_DT_BF16
#define MATH_UTILS_DECLARE_BF16 1
#endif

#if DST_SCALES_DT_E8M0 || SRC_SCALES_DT_E8M0 || WEI_SCALES_DT_E8M0
#define MATH_UTILS_DECLARE_E8M0 1
#endif

#endif
