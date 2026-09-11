//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai {
namespace ops {

struct IGemmArrays
{
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

    virtual void set_working_space(void *workspace) = 0;

    virtual ~IGemmArrays() = default;
};

template <typename To, typename Tw, typename Tr>
struct GemmArrays : public IGemmArrays
{
    const To *_Aptr              = nullptr;
    int       _lda               = 0;
    int       _A_batch_stride    = 0;
    int       _A_multi_stride    = 0;
    const Tw *_Bptr              = nullptr;
    int       _ldb               = 0;
    int       _B_multi_stride    = 0;
    Tr       *_Cptr              = nullptr;
    int       _ldc               = 0;
    int       _C_batch_stride    = 0;
    int       _C_multi_stride    = 0;
    const Tr *_bias              = nullptr;
    int       _bias_multi_stride = 0;
    void     *_workspace         = nullptr;

    GemmArrays() = default;

    GemmArrays(const To *A,
               const int lda,
               const int A_batch_stride,
               const int A_multi_stride,
               const Tw *B,
               const int ldb,
               const int B_multi_stride, /* batches share B */
               Tr       *C,
               const int ldc,
               const int C_batch_stride,
               const int C_multi_stride,
               const Tr *bias,
               const int bias_multi_stride) /* no row or batch stride needed */
        : _Aptr(A),
          _lda(lda),
          _A_batch_stride(A_batch_stride),
          _A_multi_stride(A_multi_stride),
          _Bptr(B),
          _ldb(ldb),
          _B_multi_stride(B_multi_stride),
          _Cptr(C),
          _ldc(ldc),
          _C_batch_stride(C_batch_stride),
          _C_multi_stride(C_multi_stride),
          _bias(bias),
          _bias_multi_stride(bias_multi_stride)
    {
    }

    GemmArrays(const GemmArrays<To, Tw, Tr> &)            = default;
    GemmArrays &operator=(const GemmArrays<To, Tw, Tr> &) = default;
    GemmArrays(GemmArrays<To, Tw, Tr> &&)                 = default;
    GemmArrays &operator=(GemmArrays<To, Tw, Tr> &&)      = default;
    ~GemmArrays() override                                = default;

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
        _Aptr              = A;
        _lda               = lda;
        _A_batch_stride    = A_batch_stride;
        _A_multi_stride    = A_multi_stride;
        _Bptr              = B;
        _ldb               = ldb;
        _B_multi_stride    = B_multi_stride;
        _Cptr              = C;
        _ldc               = ldc;
        _C_batch_stride    = C_batch_stride;
        _C_multi_stride    = C_multi_stride;
        _bias              = bias;
        _bias_multi_stride = bias_multi_stride;
    }

    ///* Implementation of the void * overload which casts its arguments to the appropriate type. */
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

    void set_working_space(void *workspace) override
    {
        _workspace = workspace;
    }
};
}  // namespace ops
}  // namespace kai
