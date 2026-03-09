/*******************************************************************************
* Copyright 2019 Intel Corporation
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

// DATATYPE SPECIALIZATION MECHANISM
// ===
//
// This header enables individual OpenCL kernels to generalize across datatypes
// by defining macros as stand-ins for types, conversions, and constants. Based
// on a given configuration macro (one of DT_F32, DT_F64, etc - see
// kernel_ctx_t::set_data_type), this section defines families of macros that
// each correspond to an abstract type and whose members define their associated
// properties. These abstract types (and base names of macro families) are:
//
//   - DATA: A data type corresponding to the configuration macro (DT_F32,
//       DT_F64, etc). In practice, DATA is made to represent the source data
//       type of the kernel.
//
//   - DEF_ACC: A data type suitable for accumulation over types of DATA.
//
//   - FLT_ACC: A strictly floating point version of DEF_ACC.
//
//   - POST_OP: A data type by convention for performing internal operations on
//       outputs after the main function of kernels.
//
// Per-DT sections define DATA_T, DEF_ACC_DATA_T, POST_OP_DATA_T,
// CONVERT_DATA_T, CONVERT_FLOAT_T, and FLT_ACC_DATA_T.
//
// Similar mechanisms for source and destination data types are controlled by
// SRC_DT_ and DST_DT_ prefixed macros defined in types_specific.h.
//
// HOST-SIDE IMPLEMENTATION
//
// oneDNN builds kernels after configuring the above mechanisms with host-side
// functions:
//
//   - kernel_ctx_t::set_data_type() (see src/gpu/compute/kernel_ctx.hpp)
//     - sets one of macro DT_F32, DT_F64, etc
//     - typically set to kernel input type (matching def_data_type(.., "SRC))
//
//   - def_data_type(.., "SRC" or "DST")
//     - sets one of SRC_DT_U8, SRC_DT_S8, etc (same for DST)
//     - header sets SRC_DATA_T and DST_DATA_T

#ifndef GPU_INTEL_INCLUDE_TYPES_H
#define GPU_INTEL_INCLUDE_TYPES_H

#include "gpu/intel/include/custom_types.h"
#include "gpu/intel/include/math_utils.h"
#include "gpu/intel/include/types_specific.h"
#include "gpu/intel/include/utils.h"

#define auto __auto_type
#define typeof(x) __typeof__(x)

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define unroll_for_by(factor) __attribute__((opencl_unroll_hint(factor))) for
#define unroll_2_for unroll_for_by(2)
#define unroll_4_for unroll_for_by(4)
#define unroll_8_for unroll_for_by(8)
#define unroll_16_for unroll_for_by(16)

#define for_ for

#if defined(DT_F16) || defined(SRC_DT_F16) || defined(SRC0_DT_F16) \
        || defined(SRC1_DT_F16) || defined(DST_DT_F16) || defined(WEI_DT_F16) \
        || defined(BIA_DT_F16) || defined(ACC_DT_F16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if DT_F64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if DT_F32 == 1
#define DATA_T float
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T convert_float
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T float

#elif DT_F64 == 1
#define DATA_T double
#define DEF_ACC_DATA_T double
#define POST_OP_DATA_T double
#define CONVERT_DATA_T convert_double
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T double

#elif DT_F16 == 1
#define DATA_T half
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T convert_half
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T float

#elif DT_BF16 == 1
#if WITH_PUNNING
#define DATA_T ushort
#else
#define DATA_T bf16
#endif
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T(v) cvt_f32_to_bf16(convert_float(v))
#define CONVERT_FLOAT_T cvt_bf16_to_f32
#define FLT_ACC_DATA_T float

#elif DT_BF8 == 1
#if WITH_PUNNING
#define DATA_T uchar
#else
#define DATA_T f8_e5m2
#endif
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T(v) cvt_hf_to_f8_e5m2(convert_half(v))
#define CONVERT_FLOAT_T(v) convert_float(cvt_f8_e5m2_to_hf(v))
#define FLT_ACC_DATA_T float

#elif DT_HF8 == 1
#if WITH_PUNNING
#define DATA_T uchar
#else
#define DATA_T f8_e4m3
#endif
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T(v) cvt_hf_to_f8_e4m3(convert_half(v))
#define CONVERT_FLOAT_T(v) convert_float(cvt_f8_e4m3_to_hf(v))
#define FLT_ACC_DATA_T float

#elif DT_F4_E3M0 == 1
#if WITH_PUNNING
#define DATA_T uchar
#else
#define DATA_T f4_e3m0
#endif
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T(v) cvt_f32_to_f4_e3m0(v)
#define CONVERT_FLOAT_T(v) cvt_f4_e3m0_to_f32(v)
#define FLT_ACC_DATA_T float

#elif DT_F4_E2M1 == 1
#if WITH_PUNNING
#define DATA_T uchar
#else
#define DATA_T f4_e2m1
#endif
#define DEF_ACC_DATA_T float
#define POST_OP_DATA_T float
#define CONVERT_DATA_T(v) cvt_f32_to_f4_e2m1(v)
#define CONVERT_FLOAT_T(v) cvt_f4_e2m1_to_f32(v)
#define FLT_ACC_DATA_T float

#elif DT_S8 == 1
#define DATA_T char
#define DEF_ACC_DATA_T int
#define POST_OP_DATA_T float
#define CONVERT_DATA_T convert_char_sat_rte
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T float

#elif DT_U8 == 1
#define DATA_T uchar
#define DEF_ACC_DATA_T int
#define POST_OP_DATA_T float
#define CONVERT_DATA_T convert_uchar_sat_rte
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T float

#elif DT_S32 == 1
#define DATA_T int
#define DEF_ACC_DATA_T int
#define POST_OP_DATA_T float
#define CONVERT_DATA_T convert_int_sat_rte
#define CONVERT_FLOAT_T convert_float
#define FLT_ACC_DATA_T float

#elif !defined(DT_UNDEF)
#error "Unexpected data type"
#endif

#if VECT_DT_N == 1
#define VECT_N(type) type
#else
#define VECT_N(type) CONCAT2(type, VECT_DT_N)
#endif

#define VECT_DATA_T VECT_N(DATA_T)
#define VECT_FLOAT_T VECT_N(float)

#define OFF_MD_2(prefix, x0, x1, x2, x3, x4, x5) \
    ((((x0) / CONCAT2(prefix, _B0_2)) / CONCAT2(prefix, _B0_1) \
             * CONCAT2(prefix, _S0_0)) \
            + (((x0) / CONCAT2(prefix, _B0_2)) % CONCAT2(prefix, _B0_1) \
                    * CONCAT2(prefix, _S0_1)) \
            + (((x0) % CONCAT2(prefix, _B0_2)) * CONCAT2(prefix, _S0_2)) \
            + (((x1) / CONCAT2(prefix, _B1_2)) / CONCAT2(prefix, _B1_1) \
                    * CONCAT2(prefix, _S1_0)) \
            + (((x1) / CONCAT2(prefix, _B1_2)) % CONCAT2(prefix, _B1_1) \
                    * CONCAT2(prefix, _S1_1)) \
            + (((x1) % CONCAT2(prefix, _B1_2)) * CONCAT2(prefix, _S1_2)) \
            + (((x2) / CONCAT2(prefix, _B2_2)) / CONCAT2(prefix, _B2_1) \
                    * CONCAT2(prefix, _S2_0)) \
            + (((x2) / CONCAT2(prefix, _B2_2)) % CONCAT2(prefix, _B2_1) \
                    * CONCAT2(prefix, _S2_1)) \
            + (((x2) % CONCAT2(prefix, _B2_2)) * CONCAT2(prefix, _S2_2)) \
            + (((x3) / CONCAT2(prefix, _B3_2)) / CONCAT2(prefix, _B3_1) \
                    * CONCAT2(prefix, _S3_0)) \
            + (((x3) / CONCAT2(prefix, _B3_2)) % CONCAT2(prefix, _B3_1) \
                    * CONCAT2(prefix, _S3_1)) \
            + (((x3) % CONCAT2(prefix, _B3_2)) * CONCAT2(prefix, _S3_2)) \
            + (((x4) / CONCAT2(prefix, _B4_2)) / CONCAT2(prefix, _B4_1) \
                    * CONCAT2(prefix, _S4_0)) \
            + (((x4) / CONCAT2(prefix, _B4_2)) % CONCAT2(prefix, _B4_1) \
                    * CONCAT2(prefix, _S4_1)) \
            + (((x4) % CONCAT2(prefix, _B4_2)) * CONCAT2(prefix, _S4_2)) \
            + (((x5) / CONCAT2(prefix, _B5_2)) / CONCAT2(prefix, _B5_1) \
                    * CONCAT2(prefix, _S5_0)) \
            + (((x5) / CONCAT2(prefix, _B5_2)) % CONCAT2(prefix, _B5_1) \
                    * CONCAT2(prefix, _S5_1)) \
            + (((x5) % CONCAT2(prefix, _B5_2)) * CONCAT2(prefix, _S5_2)))

#define OFF_MD(prefix, x0, x1, x2, x3, x4, x5) \
    CONCAT2(OFF_MD_, CONCAT2(prefix, _NLEVELS))(prefix, x0, x1, x2, x3, x4, x5)

#endif
