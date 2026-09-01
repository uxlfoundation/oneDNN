//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/conv/winograd.hpp"

#include "addressing.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>

namespace kai {
namespace ops {
namespace winograd {
namespace output_transform {

/* Driver class for the Winograd output transforms.
 *
 * This provides a base implementation which handles iteration over the output
 * tensor; subclasses are responsible for managing working space and executing
 * the transform on individual tiles.
 */
template <typename TIn, typename TOut=TIn>
class TransformBase : public ITransform
{
  const std::string m_name;
  const unsigned int m_output_rows, m_output_cols;
  const unsigned int m_kernel_rows, m_kernel_cols;

  protected:
  virtual size_t get_working_space_per_thread(const ConvolutionArgs &) const
  {
    return 0;
  }

  virtual void initialise_thread_working_space(const ConvolutionArgs &, void *) const
  {
    // Nothing to do
  }

  virtual void execute_tile(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_matrix,
    const TIn *bias,
    TOut *outptr, size_t ld_out_row, size_t ld_out_col,
    TOut activation_min, TOut activation_max,
    unsigned int valid_rows, unsigned int valid_cols,
    void *working_space
  ) const = 0;

  void execute_internal(
    const ConvolutionArgs &args,
    const TIn *inptr, size_t ld_in_batch, size_t ld_in_matrix, size_t ld_in_row,
    const TIn *bias,
    TOut *outptr, size_t ld_out_batch, size_t ld_out_row, size_t ld_out_col,
    void *working_space, unsigned int thread_id, unsigned int n_threads
  ) const
  {
    // Get the working space for this thread, and initialise it.
    working_space = reinterpret_cast<char *>(working_space) +
                    this->get_working_space_per_thread(args) * thread_id;
    this->initialise_thread_working_space(args, working_space);

    // Get the activation values
    auto activation_min = static_cast<TOut>(-std::numeric_limits<float>::infinity());
    auto activation_max = static_cast<TOut>(+std::numeric_limits<float>::infinity());
    switch (args.activation.type)
    {
      case kai::ops::Activation::Type::BoundedReLU:
        activation_max = static_cast<TOut>(args.activation.param1);
        // Fall through
      case kai::ops::Activation::Type::ReLU:
        activation_min = static_cast<TOut>(0);
        break;
      default:
        break;
    }

    // Determine the number of tiles in a row, we use this to get the right
    // offset into the input data.
    const auto n_tile_cols = (args.output_shape.cols + this->get_output_cols() - 1) / this->get_output_cols();

    // Execute over all batches
    for (unsigned int batch = 0; batch < args.n_batches; batch++)
    {
      auto inptr_row = inptr + thread_id*n_tile_cols*ld_in_row;
      auto outptr_row = outptr + thread_id*ld_out_row*this->get_output_rows();
      inptr += ld_in_batch;
      outptr += ld_out_batch;

      // Stripe rows of tiles over threads.
      for (auto out_i = thread_id * this->get_output_rows();
           out_i < args.output_shape.rows;
           out_i += n_threads * this->get_output_rows())
      {
        auto inptr_tile = inptr_row;
        auto outptr_tile = outptr_row;
        inptr_row += n_threads * n_tile_cols * ld_in_row;
        outptr_row += n_threads * this->get_output_rows() * ld_out_row;

        // Iterate over all columns
        for (auto out_j = 0u; out_j < args.output_shape.cols;
             out_j += this->get_output_cols())
        {
          // Execute the tile
          this->execute_tile(
            args.n_output_channels,
            inptr_tile, ld_in_matrix,
            bias,
            outptr_tile, ld_out_row, ld_out_col,
            activation_min, activation_max,
            args.output_shape.rows - out_i,  // Number of valid rows remaining
            args.output_shape.cols - out_j,  // Number of valid columns remaining
            working_space
          );

          // Progress the pointers
          inptr_tile += ld_in_row;
          outptr_tile += this->get_output_cols() * ld_out_col;
        }
      }
    }
  }

  public:
  TransformBase(const std::string &name,
                unsigned int output_rows, unsigned int output_cols,
                unsigned int kernel_rows, unsigned int kernel_cols)
  : m_name(name),
    m_output_rows(output_rows), m_output_cols(output_cols),
    m_kernel_rows(kernel_rows), m_kernel_cols(kernel_cols)
  {
  }

  const std::string &get_name(void) const override { return m_name; }

  unsigned int get_input_rows(void) const override final { return m_kernel_rows + m_output_rows - 1; }
  unsigned int get_input_cols(void) const override final { return m_kernel_cols + m_output_cols - 1; }

  unsigned int get_output_rows(void) const override final { return m_output_rows; }
  unsigned int get_output_cols(void) const override final { return m_output_cols; }

  unsigned int get_kernel_rows(void) const override final { return m_kernel_rows; }
  unsigned int get_kernel_cols(void) const override final { return m_kernel_cols; }

  size_t get_working_space_size(const ConvolutionArgs &args, unsigned int n_threads) const override
  {
    return n_threads * this->get_working_space_per_thread(args);
  }

  void execute(
    const ConvolutionArgs &args,
    const void *inptr, size_t ld_in_batch, size_t ld_in_matrix, size_t ld_in_row,
    const void *bias,
    void *outptr, size_t ld_out_batch, size_t ld_out_row, size_t ld_out_col,
    void *working_space, unsigned int thread_id, unsigned int n_threads
  ) const override
  {
    execute_internal(
      args,
      reinterpret_cast<const TIn *>(inptr), ld_in_batch, ld_in_matrix, ld_in_row,
      reinterpret_cast<const TIn *>(bias),
      reinterpret_cast<TOut *>(outptr), ld_out_batch, ld_out_row, ld_out_col,
      working_space, thread_id, n_threads
    );
  }
};

template <typename TIn, typename TOut=TIn>
class TransformUnpadded : public TransformBase<TIn, TOut>
{
  using Kernel = std::function<void(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_matrix,
    const TIn *bias,
    TOut *outptr, size_t ld_out_row, size_t ld_out_col,
    TOut activation_min, TOut activation_max
  )>;
  const Kernel m_kernel;

  protected:
  size_t get_working_space_per_thread(const ConvolutionArgs &args) const override
  {
    // We create a buffer the size of the output tile
    const auto n_output_points = this->get_output_rows() * this->get_output_cols();
    return sizeof(TOut) * n_output_points * args.n_output_channels;
  }

  void execute_tile(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_matrix,
    const TIn *bias,
    TOut *outptr, size_t ld_out_row, size_t ld_out_col,
    TOut activation_min, TOut activation_max,
    unsigned int valid_rows, unsigned int valid_cols,
    void *working_space
  ) const override final
  {
    // Get copies of the output tensor parameters
    auto kernel_outptr = outptr;
    auto kernel_ld_out_row = ld_out_row, kernel_ld_out_col = ld_out_col;

    // If there's padding on either the left or the right, then we execute the
    // kernel into the output buffer and then perform a copy.
    if (valid_rows < this->get_output_rows() ||
        valid_cols < this->get_output_cols())
    {
      // Override the kernel output parameters
      kernel_outptr = reinterpret_cast<TOut *>(working_space);
      kernel_ld_out_col = n_channels;
      kernel_ld_out_row = kernel_ld_out_col * this->get_output_cols();
    }

    // Execute the kernel
    m_kernel(
      n_channels,
      inptr, ld_in_matrix,
      bias,
      kernel_outptr, kernel_ld_out_row, kernel_ld_out_col,
      activation_min, activation_max
    );

    // If necessary, copy from the working space into the destination tensor.
    if (valid_rows < this->get_output_rows() ||
        valid_cols < this->get_output_cols())
    {
      const auto last_row = std::min(valid_rows, this->get_output_rows());
      const auto last_col = std::min(valid_cols, this->get_output_cols());

      for (auto i = 0u; i < last_row; i++)
      {
        auto patch_tile = kernel_outptr;
        auto out_tile = outptr;
        kernel_outptr += kernel_ld_out_row;
        outptr += ld_out_row;

        for (auto j = 0u; j < last_col; j++)
        {
          memcpy(out_tile, patch_tile, sizeof(TOut) * n_channels);
          patch_tile += kernel_ld_out_col;
          out_tile += ld_out_col;
        }
      }
    }
  }

  public:
  TransformUnpadded(const std::string &name,
                    unsigned int output_rows, unsigned int output_cols,
                    unsigned int kernel_rows, unsigned int kernel_cols,
                    const Kernel kernel)
  : TransformBase<TIn, TOut>(name, output_rows, output_cols, kernel_rows, kernel_cols),
    m_kernel(kernel)
  {
  }

  /* Utility method to get a transposed variant of a kernel, this transposed
   * version simply calls the original kernel with the output row and column
   * strides swapped.
   */
  static constexpr Kernel get_transposed_kernel(const Kernel &kernel)
  {
    return [kernel] (
      const unsigned int n_channels,
      const TIn *const inptr, const size_t ld_in_matrix,
      const TIn *const bias,
      TOut *const outptr, const size_t ld_out_row, const size_t ld_out_col,
      const TOut activation_min, const TOut activation_max
    ) {
      kernel(n_channels, inptr, ld_in_matrix, bias,
             outptr, ld_out_col, ld_out_row,
             activation_min, activation_max);
    };
  }
};

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
