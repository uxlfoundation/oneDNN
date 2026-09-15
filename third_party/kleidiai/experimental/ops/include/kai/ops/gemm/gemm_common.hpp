//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/gemm/convolution_parameters.hpp"
#include "gemm_arrays.hpp"
#include "kai/ops/gemm/ndrange.hpp"

#include <cstddef>

namespace kai {
namespace ops {
// Avoid circular dependency with kai_ops.hpp
struct GemmConfig;

// Abstract class for the GEMM/GEMV functions.
//
// GEMM implementations may be "native" (never require any input
// permutation), "pretransposed" (require permutation up-front) or require
// working space (permute as they go along).  This interface should support
// all of them.

// The real GemmCommon class is templated based on the operand and return
// type.  This is an interface class which is independent of those types.
class IGemmCommon
{
public:
    /* Pass in the pointers to the arrays to be operated on and their
     * strides.  This "generic" version uses void *s, the preferred version
     * is the one provided by templated GemmCommon (below) which takes
     * appropriately typed pointers.  If B is pretransposed (see below) then
     * the settings for B here are ignored.
     */
    virtual void set_arrays_generic(const void                                   *A,
                                    const int                                     lda,
                                    const int                                     A_batch_stride,
                                    const int                                     A_multi_stride,
                                    const void                                   *B,
                                    const int                                     ldb,
                                    /* batches share B */ const int               B_multi_stride,
                                    void                                         *C,
                                    const int                                     ldc,
                                    const int                                     C_batch_stride,
                                    const int                                     C_multi_stride,
                                    const void                                   *bias,
                                    /* no row or batch stride needed */ const int bias_multi_stride) = 0;

    /** @returns an ndrange containing ranges of the compute space which can be
     * broken up and parallelised over
     */
    virtual ndrange_t get_window_size() const = 0;

    /* The maximum thread count is specified when the GEMM is created.  Some
     * implementations need to know how many threads will actually run in
     * order to work properly.
     *
     * In some cases, after creating the GEMM the number of threads needs to
     * be reduced (e.g. not enough work to split across threads).  This
     * method allows the number of actual threads to be run to be set (must
     * be equal or lower).
     *
     * This has an empty default implementation, as GEMMs which don't care
     * about thread count can safely ignore this.
     */
    virtual void set_nthreads(int){};

    /* Whether this GEMM can be dynamically scheduled or not. */
    virtual bool supports_dynamic_scheduling() const
    {
        return false;
    }

    /** Main execute member function
     * @param [in] work_range     specifies the range of work we want to be computed, total range defined by get_window_size()
     * @param [in] thread_locator where are we inside of the thread space
     * @param [in] threadid       a unique threadid
     */
    virtual void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) = 0;

    /*** Working space interface (optional) ***/
    /* Total number of bytes of temporary working space needed.  If zero, it's not necessary to call set_working_space(). */
    virtual size_t get_working_size() const
    {
        return 0;
    }
    /* Provide working space buffer - the void * passed in must remain allocated for the duration of any execute calls. */
    virtual void set_working_space(void *){};

    /*** "Pretransposed" interface (optional) ***/
    /* Is this object set up for pretranspose?  If so, pretranspose_array() needs to be called before execute(); */
    virtual bool B_is_pretransposed() const
    {
        return false;
    }
    /* Does pretranspose still need to be done? */
    virtual bool B_pretranspose_required() const
    {
        return false;
    }
    /* Does pretranspose accept the transposed flag? */
    virtual bool B_pretranspose_supports_transpose() const
    {
        return false;
    }
    /* Total number of bytes of space needed for pretransposed arrays. */
    virtual size_t get_B_pretransposed_array_size() const
    {
        return 0;
    }
    /* Amount of work for the threaded cases */
    virtual size_t get_B_pretranspose_window_size() const
    {
        return 1;
    }
    /* Perform pretranspose - arguments are output, input, input row stride and input multi stride. */
    /* The "real" version of this depends on the templated operand type (see below).  */
    virtual void pretranspose_B_array_generic(void *, const void *, const int, const int, bool) = 0;
    /* Threaded version with window start/end parameters */
    virtual void
    pretranspose_B_array_part_generic(void *, const void *, const int, const int, bool, const size_t, const size_t) = 0;

    /* Set pretransposed data - the void * passed in must previously have been passed to pretranspose_B_array() for the same or a similar GEMM. */
    virtual void set_pretransposed_B_data(void *)
    {
    }

    /*** "Quantized bias" interface (optional) ***/
    /* Set the bias vector for quantized GEMMs */
    virtual void set_quantized_bias(const int32_t *, size_t)
    {
    }

    /*** Indirect interface (optional) ***/
    /* Set the indirect table.  This comprises a number of values per kernel point, and a densely packed array of pointers,
     * multis * batches * kernel_points */
    virtual void set_indirect_parameters_generic(size_t, const void *const *const *)
    {
    }

    /*** Convolution interface (optional) ***/
    /* Set the convolution parameters. */
    virtual void set_convolution_parameters(ConvolutionParameters)
    {
    }

    /*** Dequanize scale interface (optional) ***/
    /* Set the dequantize scale for GEMMs when converting from int to float (float out = scale * float(int out) ) */
    virtual void set_dequantize_scale(const float)
    {
    }

    /*** Introspection interface ***/
    /* Get the configuration of this GEMM */
    virtual GemmConfig get_config() = 0;

    // Destructor
    virtual ~IGemmCommon() = default;
};

/* "Real" GemmCommon class which is templated on the operand and return types.
 *
 * In addition to correctly typed versions of the functions that operate on
 * operand and return data, this class provides a default implementation of
 * 'set_arrays' to capture the provided arguments in protected class
 * members, as essentially any implementation will need these.
 */
template <typename To, typename Tw, typename Tr>
class GemmCommon : public IGemmCommon
{
protected:
    GemmArrays<To, Tw, Tr> _gemm_arrays{};

public:
    void set_gemm_arrays(GemmArrays<To, Tw, Tr> &ga)
    {
        _gemm_arrays = ga;
    }

    const GemmArrays<To, Tw, Tr> &get_gemm_arrays() const
    {
        return _gemm_arrays;
    }

    /* Pass in the pointers to the arrays to be operated on and their
     * strides (templated version with appropriate types). */
    virtual void set_arrays(const To                                     *A,
                            const int                                     lda,
                            const int                                     A_batch_stride,
                            const int                                     A_multi_stride,
                            const Tw                                     *B,
                            const int                                     ldb,
                            /* batches share B */ const int               B_multi_stride,
                            Tr                                           *C,
                            const int                                     ldc,
                            const int                                     C_batch_stride,
                            const int                                     C_multi_stride,
                            const Tr                                     *bias,
                            /* no row or batch stride needed */ const int bias_multi_stride)
    {
        _gemm_arrays.set_arrays(A, lda, A_batch_stride, A_multi_stride, B, ldb,
                   B_multi_stride, C, ldc, C_batch_stride, C_multi_stride,
                   bias, bias_multi_stride);
    }

    /* Implementation of the void * overload which casts its arguments to the appropriate type. */
    void set_arrays_generic(const void                                   *A,
                            const int                                     lda,
                            const int                                     A_batch_stride,
                            const int                                     A_multi_stride,
                            const void                                   *B,
                            const int                                     ldb,
                            /* batches share B */ const int               B_multi_stride,
                            void                                         *C,
                            const int                                     ldc,
                            const int                                     C_batch_stride,
                            const int                                     C_multi_stride,
                            const void                                   *bias,
                            /* no row or batch stride needed */ const int bias_multi_stride) override
    {
        set_arrays(static_cast<const To *>(A), lda, A_batch_stride, A_multi_stride, static_cast<const Tw *>(B), ldb,
                   B_multi_stride, static_cast<Tr *>(C), ldc, C_batch_stride, C_multi_stride,
                   static_cast<const Tr *>(bias), bias_multi_stride);
    }

    /*** "Pretransposed" interface ***/

    /* Compute col sums over all columns */
    virtual void requantize_bias(void *, const Tw *, const int, const int){};

    /* Perform pretranspose - the void * passed in must remain allocated for the duration of any execute calls. */
    /* Arguments are: output buffer pointer, source pointer, source row stride, source multi stride */
    virtual void pretranspose_B_array(void *, const Tw *, const int, const int, bool){};

    /* Implementation of the void * overload which casts its arguments to the appropriate type. */
    void pretranspose_B_array_generic(void *out, const void *in, const int row_stride, const int multi_stride, bool transposed) override
    {
        pretranspose_B_array(out, static_cast<const Tw *>(in), row_stride, multi_stride, transposed);
    }

    /* Threaded versions of the above.
     * The fallback/backwards compatible version of the threaded interface exposes a window size of 1 and
     * just calls the non-threaded functions to do the work.  This is valid as with window size of 1 the only
     * legal values for start and end are 0 and 1 respectively. */
    virtual void
    pretranspose_B_array_part(void *out, const Tw *in, const int row_stride, const int multi_stride, bool transposed, size_t, size_t)
    {
        pretranspose_B_array(out, in, row_stride, multi_stride, transposed);
    };

    void pretranspose_B_array_part_generic(
        void *out, const void *in, const int row_stride, const int multi_stride, bool transposed, size_t start, size_t end) override
    {
        pretranspose_B_array_part(out, static_cast<const Tw *>(in), row_stride, multi_stride, transposed, start, end);
    }

    /*** Indirect interface ***/
    virtual void set_indirect_parameters(size_t, const To *const *const *)
    {
    }

    void set_indirect_parameters_generic(size_t sz, const void *const *const *ptr) override
    {
        set_indirect_parameters(sz, reinterpret_cast<const To *const *const *>(ptr));
    }

    /** Stateless version of the main execute member function
     * @param [in] work_range     specifies the range of work we want to be computed, total range defined by get_window_size()
     * @param [in] thread_locator where are we inside of the thread space
     * @param [in] threadid       a unique threadid
     * @param [out] GemmArrays    structure containing the input/output addresses, and stride info
     */
    virtual void execute_stateless(const ndcoord_t &,
                                   const ndcoord_t &,
                                   int,
                                   GemmArrays<To, Tw, Tr> &)
    {
        // This must be overridden in the derived class to be used
        assert(0);
    }
};
}  // namespace ops
}  // namespace kai
