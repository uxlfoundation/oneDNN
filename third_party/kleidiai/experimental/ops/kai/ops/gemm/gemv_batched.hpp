//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "kai/ops/gemm/kai_ops.hpp"

namespace kai {
namespace ops {

/* "Batched GEMV" (where M=1 and nbatches>1) can be executed much more
 * efficiently as a GEMM (with M'=nbatches and nbatches'=1).  This wrapper
 * implements this.  */
template<typename To, typename Tr>
class GemvBatched : public GemmCommon<To, To, Tr> {
private:
    UniqueGemmCommon<To, To, Tr> _subgemm = nullptr;

public:
    GemvBatched(const GemmArgs &args) {
        /* Just create a subgemm with batches->M */
        GemmArgs newargs = args;
        newargs._Msize = args._nbatches;
        newargs._nbatches = 1;
        newargs._cfg = nullptr;
        _subgemm = gemm<To,To,Tr>(newargs);
    }

    void set_arrays(const To *A, const int, const int A_batch_stride, const int A_multi_stride,
                    const To *B, const int ldb, const int B_multi_stride,
                          Tr *C, const int, const int C_batch_stride, const int C_multi_stride,
                    const Tr *bias, const int bias_multi_stride) override {
        /* A and C's batch stride becomes their new row stride.  New batch stride is 0 as nbatches for subgemm is always 1. */
        _subgemm->set_arrays(A, A_batch_stride, 0, A_multi_stride,
                             B, ldb, B_multi_stride,
                             C, C_batch_stride, 0, C_multi_stride,
                             bias, bias_multi_stride);
    }

    ndrange_t get_window_size() const override {
        return _subgemm->get_window_size();
    }

    void set_nthreads(int nthreads) override {
        _subgemm->set_nthreads(nthreads);
    }

    void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) override {
        _subgemm->execute(work_range, thread_locator, threadid);
    }

    size_t get_working_size() const override {
        return _subgemm->get_working_size();
    }

    void set_working_space(void *space) override {
        _subgemm->set_working_space(space);
    }

    bool B_is_pretransposed() const override {
        return _subgemm->B_is_pretransposed();
    }

    bool B_pretranspose_required() const override {
        return _subgemm->B_pretranspose_required();
    }

    size_t get_B_pretransposed_array_size() const override {
        return _subgemm->get_B_pretransposed_array_size();
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride, bool transposed) override {
        _subgemm->pretranspose_B_array(buffer, B, ldb, B_multi_stride, transposed);
    }

    void set_pretransposed_B_data(void *buffer) override {
        _subgemm->set_pretransposed_B_data(buffer);
    }

    GemmConfig get_config() override {
        GemmConfig c = _subgemm->get_config();

        std::string new_filter = "gemv_batched[";
        new_filter.append(c.filter);
        new_filter.append("]");

        c.filter = new_filter;

        return c;
    }
};

}  // namespace ops
}  // namespace kai
