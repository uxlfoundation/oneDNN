//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/gemm/kai_ops.hpp"
#include <cstddef>

namespace kai {
namespace ops
{
struct Shape2D
{
    unsigned int rows, cols;
};

struct ConvolutionArgs
{
    unsigned int         n_batches;
    Shape2D              input_shape;
    unsigned int         n_input_channels;
    unsigned int         pad_top, pad_left;
    Shape2D              output_shape;
    unsigned int         n_output_channels;
    Shape2D              kernel_shape;
    kai::ops::Activation activation;

    ConvolutionArgs(
        unsigned int   n_batches,
        const Shape2D &input_shape,
        unsigned int   n_input_channels,
        unsigned int pad_top, unsigned int pad_left,
        const Shape2D              &output_shape,
        unsigned int                n_output_channels,
        const Shape2D               kernel_shape,
        const kai::ops::Activation &activation = {})
        : n_batches(n_batches), input_shape(input_shape), n_input_channels(n_input_channels), pad_top(pad_top), pad_left(pad_left), output_shape(output_shape), n_output_channels(n_output_channels),
          kernel_shape(kernel_shape), activation(activation)
    {
    }
};

namespace winograd
{
/* Constrain the selected Winograd implementation.
 */
struct WinogradConfig
{
    unsigned int output_rows = 0, output_cols = 0;
    std::string  input_transform_filter;
    std::string  output_transform_filter;
    std::string  weight_transform_filter;
};

/* Struct describing (suggested) memory layout within the Winograd domain.
 */
struct WinogradDomainSpec
{
    size_t weight_matrix_size_bytes, input_matrix_size_bytes, output_matrix_size_bytes;

    size_t weight_ld_matrix, weight_ld_row;
    size_t input_ld_batch, input_ld_matrix, input_ld_row;
    size_t output_ld_batch, output_ld_matrix, output_ld_row;
};

class ITransformCommon
{
public:
    virtual ~ITransformCommon() = default;

    // Get the name of the transform
    virtual const std::string &get_name(void) const = 0;
};

namespace weight_transform
{
class ITransform : public ITransformCommon
{
public:
    ~ITransform() override = default;

    virtual unsigned int get_kernel_rows(void) const = 0;
    virtual unsigned int get_kernel_cols(void) const = 0;

    virtual unsigned int get_transformed_tile_rows(void) const = 0;
    virtual unsigned int get_transformed_tile_cols(void) const = 0;

    void execute(
        const ConvolutionArgs &args,
        const void *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_input_channel,
        void *outptr, const WinogradDomainSpec &wds,
        unsigned int thread_id, unsigned int n_threads) const
    {
        this->execute(
            args, inptr, ld_in_row, ld_in_col, ld_input_channel,
            outptr, wds.weight_ld_matrix, wds.weight_ld_row,
            thread_id, n_threads);
    }

    virtual void execute(
        const ConvolutionArgs &args,
        const void *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_input_channel,
        void *outptr, size_t ld_out_matrix, size_t ld_out_row,
        unsigned int thread_id, unsigned int n_threads) const = 0;
};

} // namespace weight_transform

namespace input_transform
{
class ITransform : public ITransformCommon
{
public:
    ~ITransform() override = default;

    virtual unsigned int get_input_rows(void) const = 0;
    virtual unsigned int get_input_cols(void) const = 0;

    virtual size_t get_working_space_size(
        const ConvolutionArgs &args,
        unsigned int           n_threads) const = 0;

    void execute(
        const ConvolutionArgs &args,
        const void *inptr, size_t ld_in_batch, size_t ld_in_row, size_t ld_in_col,
        void *outptr, const WinogradDomainSpec &wds,
        void *working_space, unsigned int thread_id, unsigned int n_threads) const
    {
        this->execute(
            args, inptr, ld_in_batch, ld_in_row, ld_in_col,
            outptr, wds.input_ld_batch, wds.input_ld_matrix, wds.input_ld_row,
            working_space, thread_id, n_threads);
    }

    virtual void execute(
        const ConvolutionArgs &args,
        const void *inptr, size_t ld_in_batch, size_t ld_in_row, size_t ld_in_col,
        void *outptr, size_t ld_out_batch, size_t ld_out_matrix, size_t ld_out_row,
        void *working_space, unsigned int thread_id, unsigned int n_threads) const = 0;
};

} // namespace input_transform

namespace output_transform
{
class ITransform : public ITransformCommon
{
public:
    ~ITransform() override = default;

    virtual unsigned int get_input_rows(void) const = 0;
    virtual unsigned int get_input_cols(void) const = 0;

    virtual unsigned int get_output_rows(void) const = 0;
    virtual unsigned int get_output_cols(void) const = 0;

    virtual unsigned int get_kernel_rows(void) const = 0;
    virtual unsigned int get_kernel_cols(void) const = 0;

    virtual size_t get_working_space_size(
        const ConvolutionArgs &args,
        unsigned int           n_threads) const = 0;

    void execute(
        const ConvolutionArgs &args,
        const void *inptr, const WinogradDomainSpec &wds,
        const void *bias,
        void *outptr, size_t ld_out_batch, size_t ld_out_row, size_t ld_out_col,
        void *working_space, unsigned int thread_id, unsigned int n_threads) const
    {
        this->execute(
            args,
            inptr, wds.output_ld_batch, wds.output_ld_matrix, wds.output_ld_row,
            bias,
            outptr, ld_out_batch, ld_out_row, ld_out_col,
            working_space, thread_id, n_threads);
    }

    virtual void execute(
        const ConvolutionArgs &args,
        const void *inptr, size_t ld_in_batch, size_t ld_in_matrix, size_t ld_in_row,
        const void *bias,
        void *outptr, size_t ld_out_batch, size_t ld_out_row, size_t ld_out_col,
        void *working_space, unsigned int thread_id, unsigned int n_threads) const = 0;
};

} // namespace output_transform

struct WinogradImpl
{
    const output_transform::ITransform *output_transform = nullptr;
    const weight_transform::ITransform *weight_transform = nullptr;
    const input_transform::ITransform *input_transform  = nullptr;
    std::unique_ptr<kai::ops::GemmArgs> gemm_args;
    WinogradDomainSpec                  winograd_spec;
};

/* Get pointers to Winograd transforms for the given convolution problem.
 *
 * Assigns to the pointers in the `dest` struct and returns true or false to
 * indicate whether the given problem can be executed or not.
 */
template <typename TIn, typename TWeight = TIn, typename TOut = TIn, typename TWinogradIn = TIn, typename TWinogradOut = TOut>
bool get_implementation(
    WinogradImpl &dest, // Destination for the selected implementation
    const CPUInfo *,
    const ConvolutionArgs &,
    int  max_threads,
    bool fast_mode,
    const WinogradConfig *,
    const kai::ops::GemmConfig *);

} // namespace winograd
}  // namespace ops
}  // namespace kai
