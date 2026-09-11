//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/conv/winograd.hpp"
#include <memory>
#include <string>

namespace kai {
namespace ops {
namespace winograd {

enum class MethodConstraints
{
  None,
  RequiresSVE  = 0x1,
  RequiresSVE2 = 0x2,
  RequiresSME  = 0x4,
  RequiresSME2 = 0x8,
  LargerShape  = 0x10, // Input tensor shape is larger than the output transform tile shape.
};

constexpr inline bool operator!(const MethodConstraints &c)
{
  return c == MethodConstraints::None;
}

constexpr inline MethodConstraints operator|(const MethodConstraints &a, const MethodConstraints &b)
{
  return static_cast<MethodConstraints>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

constexpr inline MethodConstraints operator&(const MethodConstraints &a, const MethodConstraints &b)
{
  return static_cast<MethodConstraints>(static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
}

inline bool constraints_met(const MethodConstraints &c, const CPUInfo *ci, const ConvolutionArgs &, const WinogradConfig *)
{
  return (
    (!(c & MethodConstraints::RequiresSVE) || (ci->has_sve())) &&
    (!(c & MethodConstraints::RequiresSVE2) || (ci->has_sve2())) &&
    (!(c & MethodConstraints::RequiresSME) || (ci->has_sme())) &&
    (!(c & MethodConstraints::RequiresSME2) || (ci->has_sme2()))
    // Add further constraints here
  );
}

inline bool output_transform_constraints_met(const output_transform::ITransform *transform, const MethodConstraints &c, const CPUInfo *ci, const ConvolutionArgs &conv_args, const WinogradConfig *cfg)
{
  return (
    constraints_met(c, ci, conv_args, cfg) &&
    (!(c & MethodConstraints::LargerShape) || (conv_args.input_shape.rows > transform->get_output_rows() && conv_args.input_shape.cols > transform->get_output_cols()))
  );
}

namespace weight_transform {

template <typename TIn, typename TOut=TIn>
struct TransformImplementation
{
  std::unique_ptr<const ITransform> transform;
  MethodConstraints constraints;

  TransformImplementation(const ITransform *transform, const MethodConstraints &constraints = MethodConstraints::None)
  : transform(transform), constraints(constraints)
  {
  }
};

template <typename TIn, typename TOut=TIn>
const TransformImplementation<TIn, TOut> *implementation_list(void);

}  // namespace weight_transform

namespace input_transform
{

template <typename TIn, typename TOut=TIn>
struct TransformImplementation
{
  std::unique_ptr<const ITransform> transform;
  MethodConstraints constraints;

  TransformImplementation(const ITransform *transform, const MethodConstraints &constraints = MethodConstraints::None)
  : transform(transform), constraints(constraints)
  {
  }
};

template <typename TIn, typename TOut=TIn>
const TransformImplementation<TIn, TOut> *implementation_list(void);

}  // namespace input_transform

namespace output_transform
{

template <typename TIn, typename TOut=TIn>
struct TransformImplementation
{
  std::unique_ptr<const ITransform> transform;
  MethodConstraints constraints;

  TransformImplementation(const ITransform *transform, const MethodConstraints &constraints = MethodConstraints::None)
  : transform(transform), constraints(constraints)
  {
  }
};

template <typename TIn, typename TOut=TIn>
const TransformImplementation<TIn, TOut> *implementation_list(void);

}  // namespace output_transform

namespace{

template <typename T>
constexpr T iceildiv(T num, T den)
{
  return (num + den - 1) / den;
}

template <typename T>
constexpr T iroundup(T num, T den)
{
  return den * iceildiv(num, den);
}

}

template <typename TWeight, typename TWinogradIn>
inline std::vector<const weight_transform::ITransform *> get_weight_transforms(
  const CPUInfo *ci, const ConvolutionArgs &conv_args, const WinogradConfig *cfg
)
{
  // Get target inner tile size
  const auto target_inner_tile_rows = cfg->output_rows == 0 ? 0 : (conv_args.kernel_shape.rows + cfg->output_rows - 1);
  const auto target_inner_tile_cols = cfg->output_cols == 0 ? 0 : (conv_args.kernel_shape.cols + cfg->output_cols - 1);

  std::vector<const weight_transform::ITransform *> weight_transforms;
  for (auto impl = weight_transform::implementation_list<TWeight, TWinogradIn>();
       impl->transform.get() != nullptr; impl++)
  {
    // If this transform supports the requested kernel size, then add it to the
    // list of weight transforms.
    if (
      constraints_met(impl->constraints, ci, conv_args,  cfg) &&
      impl->transform->get_kernel_rows() == conv_args.kernel_shape.rows &&
      impl->transform->get_kernel_cols() == conv_args.kernel_shape.cols &&
      (target_inner_tile_rows == 0 || target_inner_tile_rows == impl->transform->get_transformed_tile_rows()) &&
      (target_inner_tile_cols == 0 || target_inner_tile_cols == impl->transform->get_transformed_tile_cols()) &&
      (cfg->weight_transform_filter == "" || std::strstr(impl->transform->get_name().c_str(), cfg->weight_transform_filter.c_str()))
    )
    {
      weight_transforms.push_back(impl->transform.get());
    }
  }

  return weight_transforms;
}

template <typename TIn, typename TWinogradIn>
inline std::vector<const input_transform::ITransform *> get_input_transforms(
  const CPUInfo *ci, const ConvolutionArgs &conv_args, const WinogradConfig *cfg
)
{
  // Get target inner tile size
  const auto target_inner_tile_rows = cfg->output_rows == 0 ? 0 : (conv_args.kernel_shape.rows + cfg->output_rows - 1);
  const auto target_inner_tile_cols = cfg->output_cols == 0 ? 0 : (conv_args.kernel_shape.cols + cfg->output_cols - 1);

  std::vector<const input_transform::ITransform *> input_transforms;
  for (auto impl = input_transform::implementation_list<TIn, TWinogradIn>();
       impl->transform.get() != nullptr; impl++)
  {
    if(
      constraints_met(impl->constraints, ci, conv_args,  cfg) &&
      (target_inner_tile_rows == 0 || target_inner_tile_rows == impl->transform->get_input_rows()) &&
      (target_inner_tile_cols == 0 || target_inner_tile_cols == impl->transform->get_input_cols()) &&
      (cfg->input_transform_filter == "" || std::strstr(impl->transform->get_name().c_str(), cfg->input_transform_filter.c_str()))
    )
    {
      input_transforms.push_back(impl->transform.get());
    }
  }

  return input_transforms;
}

template <typename TWinogradOut, typename TOut>
inline std::vector<const output_transform::ITransform *> get_output_transforms(
  const CPUInfo *ci, const ConvolutionArgs &conv_args, const WinogradConfig *cfg
)
{
  std::vector<const output_transform::ITransform *> output_transforms;
  for (auto impl = output_transform::implementation_list<TWinogradOut, TOut>();
       impl->transform.get() != nullptr; impl++)
  {
    if(
      output_transform_constraints_met(impl->transform.get(), impl->constraints, ci, conv_args,  cfg) &&
      impl->transform->get_kernel_rows() == conv_args.kernel_shape.rows &&
      impl->transform->get_kernel_cols() == conv_args.kernel_shape.cols &&
      (cfg->output_rows == 0 || cfg->output_rows == impl->transform->get_output_rows()) &&
      (cfg->output_cols == 0 || cfg->output_cols == impl->transform->get_output_cols()) &&
      (cfg->output_transform_filter == "" || std::strstr(impl->transform->get_name().c_str(), cfg->output_transform_filter.c_str()))
    )
    {
      output_transforms.push_back(impl->transform.get());
    }
  }

  return output_transforms;
}

template <typename TIn, typename TWeight, typename TOut, typename TWinogradIn, typename TWinogradOut>
bool get_implementation(
  WinogradImpl &dest,  // Destination for the selected implementation
  const CPUInfo *ci,
  const ConvolutionArgs &conv_args,
  int max_threads,
  bool fast_mode,
  const WinogradConfig *cfg,
  const kai::ops::GemmConfig *gemm_cfg
)
{
  // Get vectors of valid weight, input and output transforms; then select the
  // combination which produces the biggest output tile.
  const auto weight_transforms = get_weight_transforms<TWeight, TWinogradIn>(ci, conv_args, cfg);
  const auto input_transforms = get_input_transforms<TIn, TWinogradIn>(ci, conv_args, cfg);
  const auto output_transforms = get_output_transforms<TWinogradOut, TOut>(ci, conv_args, cfg);

  // Now attempt to select a complete set of Winograd transformations which can
  // solve the problem. Work backwards from the output transform to find
  // matching input implementations.
  bool success = false;
  for (auto output_transform = output_transforms.cbegin();
       !success && output_transform != output_transforms.cend();
       output_transform++)
  {
    // Look for matching weight transforms, if we find one then we look for
    // matching input transforms.
    for (auto weight_transform = weight_transforms.cbegin();
         !success && weight_transform != weight_transforms.cend();
         weight_transform++)
    {
      // If this weight transform is compatible, then look for a matching input
      // transform
      if ((*output_transform)->get_input_rows() == (*weight_transform)->get_transformed_tile_rows() &&
          (*output_transform)->get_input_cols() == (*weight_transform)->get_transformed_tile_cols())
      {
        for (auto input_transform = input_transforms.cbegin();
             !success && input_transform != input_transforms.cend();
             input_transform++)
        {
          // If the input transform is suitable, then set the configuration and
          // indicate success.
          if ((*input_transform)->get_input_rows() == (*output_transform)->get_input_rows() &&
              (*input_transform)->get_input_cols() == (*output_transform)->get_input_cols())
          {
            dest.output_transform = *output_transform;
            dest.input_transform = *input_transform;
            dest.weight_transform = *weight_transform;
            success = true;
          }
        }
      }
    }
  }

  if (!success)
  {
    return false;
  }

  // If we're able to construct the Winograd elements, then specify the GEMM
  // arguments required to perform the multiply-accumulate step of the
  // convolution.
  const auto n_output_row_tiles = iceildiv(conv_args.output_shape.rows, dest.output_transform->get_output_rows());
  const auto n_output_col_tiles = iceildiv(conv_args.output_shape.cols, dest.output_transform->get_output_cols());
  const auto n_output_patches = n_output_row_tiles * n_output_col_tiles;

  const int n_multis = dest.input_transform->get_input_rows() *
                       dest.input_transform->get_input_cols();

  dest.gemm_args.reset(new kai::ops::GemmArgs(
    ci,
    n_output_patches,  // M
    conv_args.n_output_channels,  // N
    conv_args.n_input_channels,  // K
    1,  // K-sections
    conv_args.n_batches,  // # Batches
    n_multis,
    false,  // Indirect input
    {},  // No activation
    max_threads,
    false, // Not fixed format
    fast_mode,
    false, // Not accumulate
    gemm_cfg
  ));

  // Also provide hints for the Winograd memory layout
  auto &ws = dest.winograd_spec;
  ws.weight_ld_row = iroundup(conv_args.n_output_channels, 4u);
  ws.weight_ld_matrix = conv_args.n_input_channels * ws.weight_ld_row;
  ws.weight_matrix_size_bytes = n_multis * ws.weight_ld_matrix * sizeof(TWinogradIn);

  ws.input_ld_row = iroundup(conv_args.n_input_channels, 4u);
  ws.input_ld_matrix = iroundup(n_output_patches, 4u) * ws.input_ld_row;
  ws.input_ld_batch = n_multis * ws.input_ld_matrix;
  ws.input_matrix_size_bytes = conv_args.n_batches * ws.input_ld_batch * sizeof(TWinogradIn);

  ws.output_ld_row = ws.weight_ld_row;
  ws.output_ld_matrix = n_output_patches * ws.output_ld_row;
  ws.output_ld_batch = n_multis * ws.output_ld_matrix;
  ws.output_matrix_size_bytes = conv_args.n_batches * ws.output_ld_batch * sizeof(TWinogradOut);

  return true;
}

}  // namespace winograd
}  // namespace ops
}  // namespace kai
