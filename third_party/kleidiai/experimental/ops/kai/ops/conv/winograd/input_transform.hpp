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

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

namespace {

template <typename T>
constexpr T iceildiv(const T a, const T b)
{
  return (a + b - 1) / b;
}

}

/* Driver class for the Winograd input transforms.
 *
 * This provides a base implementation which handles iteration over the input
 * tensor; subclasses are responsible for managing working space and executing
 * the transform on individual tiles.
 */
template <typename TIn, typename TOut=TIn>
class TransformBase : public ITransform
{
  const std::string m_name;
  const unsigned int m_input_rows, m_input_cols;

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
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col,
    TOut *outptr, size_t ld_out_matrix,
    unsigned int pad_top, unsigned int valid_rows,
    unsigned int pad_left, unsigned int valid_cols,
    void *working_space
  ) const = 0;

  void execute_internal(
    const ConvolutionArgs &args,
    const TIn *inptr, size_t ld_in_batch, size_t ld_in_row, size_t ld_in_col,
    TOut *outptr, size_t ld_out_batch, size_t ld_out_matrix, size_t ld_out_row,
    void *working_space, unsigned int thread_id, unsigned int n_threads
  ) const
  {
    // Get the working space for this thread, and initialise it.
    working_space = reinterpret_cast<char *>(working_space) +
                    this->get_working_space_per_thread(args) * thread_id;
    this->initialise_thread_working_space(args, working_space);

    // Get tile traversal parameters
    const auto tile_stride_rows = std::max(1u, m_input_rows - args.kernel_shape.rows + 1);
    const auto tile_stride_cols = std::max(1u, m_input_cols - args.kernel_shape.cols + 1);
    const auto n_tile_rows = iceildiv(
      args.output_shape.rows, m_input_rows - args.kernel_shape.rows + 1);
    const auto n_tile_cols = iceildiv(
      args.output_shape.cols, m_input_cols - args.kernel_shape.cols + 1);

    // Execute over all batches
    for (unsigned int batch = 0; batch < args.n_batches; batch++)
    {
      auto outptr_tile = outptr + thread_id * n_tile_cols * ld_out_row;

      // For a single batch, stripe the rows over the threads.
      for (auto tile_i = thread_id; tile_i < n_tile_rows; tile_i += n_threads)
      {
        // Compute pointers and padding for this row of tiles
        const auto start_i = tile_i * tile_stride_rows;
        const auto pad_top = start_i < args.pad_top ? args.pad_top - start_i : 0;
        const auto inptr_row = inptr + (pad_top ? 0 : start_i - args.pad_top) * ld_in_row;
        const auto valid_rows = args.input_shape.rows - (pad_top ? 0 : start_i - args.pad_top);

        // Iterate over columns
        for (auto tile_j = 0u; tile_j < n_tile_cols; tile_j++)
        {
          // Compute pointers and padding for this tile, then delegate to
          // execute the kernel.
          const auto start_j = tile_j * tile_stride_cols;
          const auto pad_left = start_j < args.pad_left ? args.pad_left - start_j : 0;
          const auto inptr_tile = inptr_row + (pad_left ? 0 : start_j - args.pad_left) * ld_in_col;
          const auto valid_cols = args.input_shape.cols - (pad_left ? 0 : start_j - args.pad_left);

          this->execute_tile(
            args.n_input_channels,
            inptr_tile, ld_in_row, ld_in_col,
            outptr_tile, ld_out_matrix,
            pad_top, valid_rows, pad_left, valid_cols,
            working_space
          );
          outptr_tile += ld_out_row;
        }

        outptr_tile += (n_threads - 1) * n_tile_cols * ld_out_row;
      }

      inptr += ld_in_batch;
      outptr += ld_out_batch;
    }
  }

  public:
  TransformBase(const std::string &name, unsigned int input_rows, unsigned int input_cols)
  : m_name(name), m_input_rows(input_rows), m_input_cols(input_cols)
  {
  }

  const std::string &get_name(void) const override { return m_name; }

  unsigned int get_input_rows(void) const override final { return m_input_rows; }
  unsigned int get_input_cols(void) const override final { return m_input_cols; }

  size_t get_working_space_size(const ConvolutionArgs &args, unsigned int n_threads) const override
  {
    return n_threads * this->get_working_space_per_thread(args);
  }

  void execute(
    const ConvolutionArgs &args,
    const void *inptr, size_t ld_in_batch, size_t ld_in_row, size_t ld_in_col,
    void *outptr, size_t ld_out_batch, size_t ld_out_matrix, size_t ld_out_row,
    void *working_space, unsigned int thread_id, unsigned int n_threads
  ) const override
  {
    execute_internal(
      args,
      reinterpret_cast<const TIn *>(inptr), ld_in_batch, ld_in_row, ld_in_col,
      reinterpret_cast<TOut *>(outptr), ld_out_batch, ld_out_matrix, ld_out_row,
      working_space, thread_id, n_threads
    );
  }
};

template <typename TIn, typename TOut=TIn>
class TransformDirect : public TransformBase<TIn, TOut>
{
  using Kernel = std::function<void(
    unsigned int,  // Number of channels
    const TIn *,  size_t, size_t,  // Pointer to first valid input element, row and column stride
    unsigned int, unsigned int, unsigned int, unsigned int,  // Top, left, bottom and right padding
    TOut *, size_t  // Base output pointer, stride between matrices
  )>;
  const Kernel m_kernel;

  protected:
  void execute_tile(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col,
    TOut *outptr, size_t ld_out_matrix,
    unsigned int pad_top, unsigned int valid_rows,
    unsigned int pad_left, unsigned int valid_cols,
    void *
  ) const override
  {
    const auto end_i = this->get_input_rows() - pad_top;
    const auto pad_bottom = end_i < valid_rows ? 0 : end_i - valid_rows;
    const auto end_j = this->get_input_cols() - pad_left;
    const auto pad_right = end_j < valid_cols ? 0 : end_j - valid_cols;

    // Execute the kernel
    m_kernel(
      n_channels, inptr, ld_in_row, ld_in_col,
      pad_top, pad_left, pad_bottom, pad_right,
      outptr, ld_out_matrix
    );
  }

  public:
  TransformDirect(const std::string &name, unsigned int input_rows, unsigned int input_cols, Kernel kernel)
  : TransformBase<TIn, TOut>(name, input_rows, input_cols), m_kernel(kernel)
  {
  }
};

template <typename TIn, typename TOut=TIn>
class TransformIndirect : public TransformBase<TIn, TOut>
{
  using Kernel = std::function<void(
    unsigned int,  // Number of channels
    const TIn *const *,  // Input pointers (one per point)
    TOut *, size_t   // Base output pointer, stride between matrices
  )>;
  const Kernel m_kernel;

  struct Workspace
  {
    const TIn **inptrs;
    const TIn *input_buffer;
  };

  size_t sizeof_inptr_array(void) const
  {
    return sizeof(const TIn **) * this->get_input_rows() * this->get_input_cols();
  }

  protected:
  size_t get_working_space_per_thread(const ConvolutionArgs &args) const override
  {
    return sizeof(Workspace) + sizeof_inptr_array() + sizeof(TIn) * args.n_input_channels;
  }

  void initialise_thread_working_space(const ConvolutionArgs &args, void *buffer) const override
  {
    Workspace *ws = reinterpret_cast<Workspace *>(buffer);
    buffer = ws + 1;

    ws->inptrs = reinterpret_cast<const TIn **>(buffer);
    buffer = reinterpret_cast<char *>(buffer) + sizeof_inptr_array();

    ws->input_buffer = reinterpret_cast<const TIn *>(buffer);
    memset(buffer, 0, sizeof(TIn) * args.n_input_channels);
  }

  void execute_tile(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col,
    TOut *outptr, size_t ld_out_matrix,
    unsigned int pad_top, unsigned int valid_rows,
    unsigned int pad_left, unsigned int valid_cols,
    void *working_space
  ) const override
  {
    // Get the working space
    auto ws = reinterpret_cast<Workspace *>(working_space);

    // Construct the input pointer array based on the given arguments
    fill_pointer_array<const TIn>(
      ws->inptrs, this->get_input_rows(), this->get_input_cols(),
      inptr, ld_in_row, ld_in_col,
      ws->input_buffer,
      pad_top, valid_rows,
      pad_left, valid_cols
    );

    // Execute the kernel
    m_kernel(n_channels, ws->inptrs, outptr, ld_out_matrix);
  }

  public:
  TransformIndirect(const std::string &name, unsigned int input_rows, unsigned int input_cols, Kernel kernel)
  : TransformBase<TIn, TOut>(name, input_rows, input_cols), m_kernel(kernel)
  {
  }
};

template <typename TIn, typename TOut=TIn>
class TransformUnpadded : public TransformBase<TIn, TOut>
{
  using Kernel = std::function<void(
    unsigned int,  // Number of channels
    const TIn *,  size_t, size_t,  // Pointer to first input element, row and column stride
    TOut *, size_t // Base output pointer, stride between matrices
  )>;
  const Kernel m_kernel;

  protected:
  size_t get_working_space_per_thread(const ConvolutionArgs &args) const override
  {
    const auto input_points = this->get_input_rows() * this->get_input_cols();
    return sizeof(TIn) * input_points * args.n_input_channels;
  }

  void execute_tile(
    unsigned int n_channels,
    const TIn *inptr, size_t ld_in_row, size_t ld_in_col,
    TOut *const outptr, const size_t ld_out_matrix,
    const unsigned int pad_top, const unsigned int valid_rows,
    const unsigned int pad_left, const unsigned int valid_cols,
    void *const working_space
  ) const override
  {
    // If there's any padding, then copy the valid portion of the tensor into
    // the working space and reset the pointer, row and column strides to point
    // at this copy of the data.
    if (pad_top || valid_rows < this->get_input_rows() ||
        pad_left || valid_cols < this->get_input_cols())
    {
      const auto patch_ld_col = n_channels;
      const auto patch_ld_row = patch_ld_col * this->get_input_cols();
      auto patch = reinterpret_cast<TIn *>(working_space) +
                   pad_top*patch_ld_row + pad_left*patch_ld_col;

      // Fill the input patch with padding
      memset(working_space, 0, sizeof(TIn) * this->get_input_rows() * patch_ld_row);

      // Determine the bounds for which to copy
      const auto last_i = std::min(valid_rows + pad_top, this->get_input_rows());
      const auto last_j = std::min(valid_cols + pad_left, this->get_input_cols());

      // Copy across the valid portion of the patch
      for (auto i = pad_top; i < last_i; i++)
      {
        auto inptr_col = inptr;
        inptr += ld_in_row;

        auto patch_col = patch;
        patch += patch_ld_row;

        for (auto j = pad_left; j < last_j; j++)
        {
          // Perform the copy and progress both input and patch pointers
          memcpy(patch_col, inptr_col, n_channels * sizeof(TIn));
          inptr_col += ld_in_col;
          patch_col += patch_ld_col;
        }
      }

      // Override the input pointer and strides
      inptr = reinterpret_cast<const TIn *>(working_space);
      ld_in_col = patch_ld_col;
      ld_in_row = patch_ld_row;
    }

    // Call the kernel
    m_kernel(n_channels, inptr, ld_in_row, ld_in_col, outptr, ld_out_matrix);
  }

  public:
  TransformUnpadded(const std::string &name, unsigned int input_rows, unsigned int input_cols, Kernel kernel)
  : TransformBase<TIn, TOut>(name, input_rows, input_cols), m_kernel(kernel)
  {
  }

  /* Utility method which can be used to get a transposed version of a kernel,
   * this just calls the kernel with the input row and column strides reversed.
   */
  static constexpr Kernel get_transposed_kernel(const Kernel &kernel)
  {
    return [kernel] (
      const unsigned int n_channels,
      const TIn *const inptr, const size_t ld_in_row, const size_t ld_in_col,
      TOut *const outptr, const size_t ld_out_matrix
    ) {
      kernel(n_channels, inptr, ld_in_col, ld_in_row, outptr, ld_out_matrix);
    };
  }
};

}  // namespace input_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
