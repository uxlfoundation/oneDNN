//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "sme_interleave1VL_bf16_bf16.hpp"
#include "sme_interleave1VL_block2_bf16_bf16.hpp"
#include "sme_interleave1VL_block2_fp16_fp16.hpp"
#include "sme_interleave1VL_block4_s8_s8.hpp"
#include "sme_interleave1VL_block4_u8_u8.hpp"
#include "sme_interleave1VL_block4_s8_s8_summing.hpp"
#include "sme_interleave1VL_block4_u8_u8_summing.hpp"
#include "sme_interleave1VL_fp16_fp16.hpp"
#include "sme_interleave1VL_fp32_fp32.hpp"
#include "sme_interleave2VL_block2_bf16_bf16.hpp"
#include "sme_interleave2VL_block2_fp16_fp16.hpp"
#include "sme_interleave2VL_block4_s8_s8.hpp"
#include "sme_interleave2VL_block4_s8_s8_summing.hpp"
#include "sme_interleave2VL_block4_u8_u8.hpp"
#include "sme_interleave2VL_block4_u8_u8_summing.hpp"
#include "sme_interleave2VL_fp16_fp16.hpp"
#include "sme_interleave2VL_bf16_bf16.hpp"
#include "sme_interleave2VL_fp32_fp32.hpp"
#include "sme_interleave4VL_block2_bf16_bf16.hpp"
#include "sme_interleave4VL_block2_fp16_fp16.hpp"
#include "sme_interleave4VL_block4_s8_s8.hpp"
#include "sme_interleave4VL_block4_u8_u8.hpp"
#include "sme_interleave4VL_block4_s8_s8_summing.hpp"
#include "sme_interleave4VL_block4_u8_u8_summing.hpp"
#include "sme_interleave4VL_fp32_fp32.hpp"

#include "sme2_interleave1VL_block2_fp32_bf16.hpp"
#include "sme2_interleave2VL_block2_fp32_bf16.hpp"
#include "sme2_interleave4VL_block2_fp32_bf16.hpp"
