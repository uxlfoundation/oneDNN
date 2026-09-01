//
// SPDX-FileCopyrightText: Copyright 2020, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "a32_interleave6_block1_fp32_fp32.hpp"
#include "a64_interleave4_block16_s8_s8.hpp"
#include "a64_interleave4_block16_s8_s8_summing.hpp"
#include "a64_interleave4_block16_u8_u8_summing.hpp"
#include "a64_interleave8_block1_bf16_fp32.hpp"
#include "a64_interleave8_block1_fp16_fp16.hpp"
#include "a64_interleave8_block1_fp16_fp32.hpp"
#include "a64_interleave8_block1_fp32_fp32.hpp"
#include "a64_interleave8_block1_s16_s16.hpp"
#include "a64_interleave8_block1_s8_s16.hpp"
#include "a64_interleave8_block1_s8_s16_summing.hpp"
#include "a64_interleave8_block1_u8_u16.hpp"
#include "a64_interleave8_block1_u8_u16_summing.hpp"
#include "a64_interleave8_block2_bf16_bf16.hpp"
#include "a64_interleave8_block2_fp32_fp32.hpp"
#include "a64_interleave8_block4_bf16_bf16.hpp"
#include "a64_interleave8_block4_fp32_bf16.hpp"
#include "a64_interleave8_block4_s8_s8.hpp"
#include "a64_interleave8_block4_s8_s8_summing.hpp"
#include "a64_interleave8_block4_u8_u8_summing.hpp"
#include "a64_interleave8_block8_s8_s8.hpp"
#include "a64_interleave8_block8_s8_s8_summing.hpp"
#include "a64_interleave8_block8_u8_u8_summing.hpp"
