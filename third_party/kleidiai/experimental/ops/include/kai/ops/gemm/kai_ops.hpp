//
// SPDX-FileCopyrightText: Copyright 2018-2022, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/gemm/kai_ops_local.hpp"
#include "kai/ops/gemm/gemm_common.hpp"
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace kai {
namespace ops {
enum class WeightFormat
{
    UNSPECIFIED    = 0x1,
    ANY            = 0x2,
    OHWI           = 0x100100,
    OHWIo2         = 0x100200,
    OHWIo4         = 0x100400,
    OHWIo8         = 0x100800,
    OHWIo16        = 0x101000,
    OHWIo32        = 0x102000,
    OHWIo64        = 0x104000,
    OHWIo128       = 0x108000,
    OHWIo4i2       = 0x200400,
    OHWIo4i2_bf16  = 0x200410,
    OHWIo8i2       = 0x200800,
    OHWIo8i2_bf16  = 0x200810,
    OHWIo16i2      = 0x201000,
    OHWIo16i2_bf16 = 0x201010,
    OHWIo32i2      = 0x202000,
    OHWIo32i2_bf16 = 0x202010,
    OHWIo64i2      = 0x204000,
    OHWIo64i2_bf16 = 0x204010,
    OHWIo4i4       = 0x400400,
    OHWIo4i4_bf16  = 0x400410,
    OHWIo8i4       = 0x400800,
    OHWIo8i4_bf16  = 0x400810,
    OHWIo16i4      = 0x401000,
    OHWIo16i4_bf16 = 0x401010,
    OHWIo32i4      = 0x402000,
    OHWIo32i4_bf16 = 0x402010,
    OHWIo64i4      = 0x404000,
    OHWIo64i4_bf16 = 0x404010,
    OHWIo2i8       = 0x800200,
    OHWIo4i8       = 0x800400,
    OHWIo8i8       = 0x800800,
    OHWIo16i8      = 0x801000,
    OHWIo32i8      = 0x802000,
    OHWIo64i8      = 0x804000
};

struct KernelDescription
{
    std::string name;
    bool        is_default     = false;
    uint64_t    cycle_estimate = 0;

    KernelDescription(std::string n, bool d = false, uint64_t c = 0)
        : name(std::move(n)), is_default(d), cycle_estimate(c)
    {
    }
    KernelDescription() noexcept = default;
};

struct GemmConfig
{
    std::string  filter;
    unsigned int inner_block_size = 0;
    unsigned int outer_block_size = 0;
    WeightFormat weight_format    = WeightFormat::ANY;

    GemmConfig() = default;
};

struct Activation
{
    enum class Type
    {
        None,
        ReLU,
        BoundedReLU
    };

    Type  type;
    float param1;
    float param2;

    Activation(Type type = Type::None, float p1 = 0.0f, float p2 = 0.0f) : type(type), param1(p1), param2(p2)
    {
    }
};

struct GemmArgs
{
public:
    const CPUInfo    *_ci;
    unsigned int      _Msize; // num of tiles
    unsigned int      _Nsize; // output channels
    unsigned int      _Ksize; // input channels
    unsigned int      _Ksections;
    unsigned int      _nbatches;
    unsigned int      _nmulti; // n_gemms to be performed
    bool              _indirect_input;
    Activation        _act;
    int               _maxthreads;
    bool              _fixed_format;
    bool              _fast_mode;
    bool              _accumulate;
    const GemmConfig *_cfg;

    GemmArgs(const CPUInfo    *ci,
             unsigned int      M,
             unsigned int      N,
             unsigned int      K,
             unsigned int      Ksections,
             unsigned int      nbatches,
             unsigned int      nmulti,
             bool              indirect_input,
             Activation        act,
             const int         maxthreads,
             bool              fixed_format = false,
             bool              fast_mode    = false,
             bool              accumulate   = false,
             const GemmConfig *cfg          = nullptr)
        : _ci(ci),
          _Msize(M),
          _Nsize(N),
          _Ksize(K),
          _Ksections(Ksections),
          _nbatches(nbatches),
          _nmulti(nmulti),
          _indirect_input(indirect_input),
          _act(act),
          _maxthreads(maxthreads),
          _fixed_format(fixed_format),
          _fast_mode(fast_mode),
          _accumulate(accumulate),
          _cfg(cfg)
    {
    }
};

struct Requantize32
{
public:
    const int32_t *bias                     = nullptr;
    size_t         bias_multi_stride        = 0;
    int32_t        a_offset                 = 0;
    int32_t        b_offset                 = 0;
    int32_t        c_offset                 = 0;
    bool           per_channel_requant      = false;
    int32_t        per_layer_left_shift     = 0;
    int32_t        per_layer_right_shift    = 0;
    int32_t        per_layer_mul            = 0;
    const int32_t *per_channel_left_shifts  = nullptr;
    const int32_t *per_channel_right_shifts = nullptr;
    const int32_t *per_channel_muls         = nullptr;
    int32_t        minval                   = 0;
    int32_t        maxval                   = 0;

    Requantize32() = default;

    // Constructor for per-tensor quantization
    Requantize32(const int32_t *bias,
                 size_t         bias_multi_stride,
                 int32_t        a_offset,
                 int32_t        b_offset,
                 int32_t        c_offset,
                 int32_t        requant_shift,
                 int32_t        requant_mul,
                 int32_t        minv,
                 int32_t        maxv)
        : bias(bias),
          bias_multi_stride(bias_multi_stride),
          a_offset(a_offset),
          b_offset(b_offset),
          c_offset(c_offset),
          per_channel_requant(false),
          per_layer_left_shift(std::max<int32_t>(requant_shift, 0)),
          per_layer_right_shift(std::min<int32_t>(requant_shift, 0)),
          per_layer_mul(requant_mul),
          minval(minv),
          maxval(maxv)
    {
    }

    // Constructor for per-channel quantization
    Requantize32(const int32_t *bias,
                 size_t         bias_multi_stride,
                 int32_t        a_offset,
                 int32_t        b_offset,
                 int32_t        c_offset,
                 const int32_t *requant_left_shifts,
                 const int32_t *requant_right_shifts,
                 const int32_t *requant_muls,
                 int32_t        minv,
                 int32_t        maxv)
        : bias(bias),
          bias_multi_stride(bias_multi_stride),
          a_offset(a_offset),
          b_offset(b_offset),
          c_offset(c_offset),
          per_channel_requant(true),
          per_channel_left_shifts(requant_left_shifts),
          per_channel_right_shifts(requant_right_shifts),
          per_channel_muls(requant_muls),
          minval(minv),
          maxval(maxv)
    {
    }
};

struct DequantizeFloat
{
public:
    float scale = 0;

    DequantizeFloat() = default;

    // Constructor
    DequantizeFloat(const float scale) : scale(scale)
    {
    }
};

struct Nothing
{
};

template <typename Tlop, typename Trop, typename Tret>
using UniqueGemmCommon = std::unique_ptr<GemmCommon<Tlop, Trop, Tret>>;

/* Low level API calls.
 * These are implemented as 'GemmArgs' versions, or with the arguments explicitly listed. */

/* get_gemm_method(): Given the templated types and provided parameters,
 * which is the preferred method to implement this GEMM?  */
template <typename Top, typename Tret, class OutputStage = Nothing>
KernelDescription get_gemm_method(const GemmArgs &args, const OutputStage & = {});

template <typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
UniqueGemmCommon<Tlop, Trop, Tret> gemm(const GemmArgs &args, const OutputStage & = {});

template <typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs &args, const OutputStage & = {});

template <typename Tlop, typename Trop, typename Tret, class OutputStage = Nothing>
bool has_opt_gemm(WeightFormat &weight_format, const GemmArgs &args, const OutputStage & = {});

}  // namespace ops
}  // namespace kai
