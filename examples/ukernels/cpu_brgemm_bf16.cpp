/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl_ukernel.hpp"

using namespace dnnl;
using namespace dnnl::ukernel;

void brgemm_example() {

    // Create execution dnnl::engine. Needed for reorders to operate over input
    // data.
    dnnl::engine engine(engine::kind::cpu, 0);

    // Create dnnl::stream. Needed for reorders for the same reason.
    dnnl::stream engine_stream(engine);

    // ukernel dimensions.
    // K is for a whole slice, K_blk splits the slice into several chunks,
    // each chunk is designed for a single ukernel call.
    const memory::dim M = 16, K = 128, K_blk = 32, N = 64;
    if (K % K_blk != 0) {
        printf("K_blk must divide K.\n");
        return;
    }
    const memory::dim K_chunks = K / K_blk;

    memory::data_type a_dt = memory::data_type::bf16;
    memory::data_type b_dt = memory::data_type::bf16;
    memory::data_type c_dt = memory::data_type::f32; // Accumulator data type.
    memory::data_type d_dt = memory::data_type::f32; // Output data type.

    // Query the packing requirement from the ukernel. It's enough to query
    // packing requirements once for multiple ukernel objects.
    const auto pack = brgemm::get_B_pack_type(a_dt, b_dt);

    // If the value is `pack_type::undef`, ukernel API is not supported on the
    // target system.
    if (pack == pack_type::undef) {
        printf("Kernel is not supported on this platform.\n");
        return;
    }

    // Packing is required if the returned value is different from
    // `pack_type::no_pack`.
    // If packing is required, specific `ldb` value can be used ahead, since
    // transform has a limited set of supported values.
    bool need_pack = pack != pack_type::no_trans;

    // Row-major/non_trans/ab format for A is assumed.
    const memory::dim lda = K;
    // `ldb` for `need_pack = true` must be one of 16, 32, 48, or 64.
    // const memory::dim ldb = need_pack ? N_block : N;
    const memory::dim ldb = N;
    // Row-major/non_trans/ab format for C is assumed.
    const memory::dim ldc = N; // Leading dimension for accumulator.
    const memory::dim batch_size = K_chunks - 1;

    // A, B, and C tensors dimensions.
    memory::dims A_dims = {M, K};
    memory::dims B_dims = {K, N};
    memory::dims C_dims = {M, N};
    memory::dims D_dims = {M, N};

    // Allocate buffers with user data.
    std::vector<float> A_user_data(product(A_dims));
    std::vector<float> B_user_data(product(B_dims));
    std::vector<float> D_data(product(D_dims)); // For reference comparison
    std::vector<float> D_user_data(product(D_dims)); // For reference comparison

    // Initialize A.
    std::generate(A_user_data.begin(), A_user_data.end(), []() {
        static int i = 0;
        return i++ % 4;
    });
    // Initialize B.
    std::generate(B_user_data.begin(), B_user_data.end(), []() {
        static int i = 6;
        static int sign_gen = 0;
        int sign = (sign_gen++ % 2) ? -1 : 1;
        float val = sign * (i++ % 5);
        return val;
    });

    // Create f32 memories. They are used as data holders and reorder into
    // memories passed to the ukernel.
    auto A_f32_md = memory::desc(
            A_dims, memory::data_type::f32, memory::format_tag::ab);
    auto B_f32_md = memory::desc(
            B_dims, memory::data_type::f32, memory::format_tag::ab);
    auto D_f32_md = memory::desc(
            D_dims, memory::data_type::f32, memory::format_tag::ab);

    auto A_f32_mem = memory(A_f32_md, engine, A_user_data.data());
    auto B_f32_mem = memory(B_f32_md, engine, B_user_data.data());
    auto D_f32_mem = memory(D_f32_md, engine, D_user_data.data());

    // Create ukernel memories in requested data types.
    // Note that all formats are `ab`, except `ba`/trans for B.
    auto A_md = memory::desc(A_dims, a_dt, memory::format_tag::ab);
    auto B_md = memory::desc(B_dims, b_dt, memory::format_tag::ba);
    auto C_md = memory::desc(C_dims, c_dt, memory::format_tag::ab);
    auto D_md = memory::desc(D_dims, d_dt, memory::format_tag::ab);

    auto A_mem = memory(A_md, engine);
    auto B_mem = memory(B_md, engine);
    auto C_mem = memory(C_md, engine);
    auto D_mem = memory(D_md, engine);

    const auto *A_ptr = reinterpret_cast<uint8_t *>(A_mem.get_data_handle());
    auto *B_ptr = reinterpret_cast<uint8_t *>(B_mem.get_data_handle());

    const size_t a_dt_size
            = memory::data_type_size(A_mem.get_desc().get_data_type());
    const size_t b_dt_size
            = memory::data_type_size(B_mem.get_desc().get_data_type());

    // Reorder user data into buffers passed to ukernels in target data types.
    reorder(A_f32_mem, A_mem).execute(engine_stream, A_f32_mem, A_mem);
    reorder(B_f32_mem, B_mem).execute(engine_stream, B_f32_mem, B_mem);
    reorder(D_f32_mem, D_mem).execute(engine_stream, D_f32_mem, D_mem);

    // Create BRGeMM ukernel objects.
    // There are two objects:
    // * `brg0` is the basic ukernel which operates over K dimension and
    //   processes only the very first K_chunk. It uses `set_add_C(false)` which
    //   writes to the accumulator instead of appending to it.
    // * `brg1` is the basic ukernel which operates over K dimension starting
    //   from the second and following K_chunks. It utilizes `set_add_C(true)`
    //   to accumulate into the same C buffer initialized by `brg0` ukernel.
    //   It also uses `batch_size` to process as much as the number of blocks
    //   over K.
    brgemm brg0;

    // Construct a basic brgemm object.
    // `allow_empty` makes the interface to return an empty `brg0` object
    // in case of the critical error.
    brg0 = brgemm(M, N, K_blk, /* batch_size = */ 1, lda, ldb, ldc, a_dt, b_dt,
            c_dt,
            /* allow_empty = */ true);
    if (!brg0) {
        printf("Error: brg0 object was not constructed.\n");
        return;
    }

    // Instruct the ukernel to write (not append!) the result to the C tensor.
    brg0.set_add_C(false);

    // Finalize the initialization.
    // Successful completion returns `true`. Otherwise, `brg0` object can't
    // be used due to lack of support or non-compatible settings. The
    // specific reason may be found by using `ONEDNN_VERBOSE=all` env var.
    const bool ok = brg0.finalize();
    if (!ok) {
        printf("Kernel is not supported on this platform.\n");
        return;
    }

    // Generate the executable code.
    brg0.generate();

    brgemm brg1;
    if (K_chunks > 1) {
        // Construct a basic brgemm object.
        // `allow_empty` makes the interface to return an empty `brg1` object
        // in case of the critical error.
        brg1 = brgemm(M, N, K_blk, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt,
                /* allow_empty = */ true);
        if (!brg1) {
            printf("Error: brg1 object was not constructed.\n");
            return;
        }

        // Instruct the ukernel to append (not write!) the result to the C
        // tensor.
        brg1.set_add_C(true);

        // Finalize the initialization.
        // Successful completion returns `true`. Otherwise, `brg1` object can't
        // be used due to lack of support or non-compatible settings. The
        // specific reason may be found by using `ONEDNN_VERBOSE=all` env var.
        const bool ok = brg1.finalize();
        if (!ok) {
            printf("Kernel is not supported on this platform.\n");
            return;
        }

        // Generate the executable code.
        brg1.generate();
    }

    uint8_t *B_blocked = nullptr;
    void *B_base_ptr = B_ptr;
    size_t blocked_B_size = 0;

    // Create a dedicated object for data transformation.
    if (need_pack) {
        // Transform kernel for tensor B. The ukernel expects B passed in a
        // special VNNI format for bfloat16_t data type.
        //
        // Note: the routine doesn't provide a `batch_size` argument in the
        // constructor as it can be either incorporated into `K` dimension, or
        // manually iterated over in a for-loop on the user side.
        //
        // Note: `in_pack_type` specifies if transposed format should be
        // expected by the transform routine. `in_ld` specifies how many
        // elements between consecutive N points.
        transform pack_B(/* K = */ K_blk * K_chunks, /* N = */ N,
                /* in_pack_type = */ pack_type::trans, /* in_ld = */ K,
                /* out_ld = */ ldb, /* in_dt = */ b_dt, /* out_dt = */ b_dt);

        // Size of the packed tensor.
        blocked_B_size = ldb * K_blk * memory::data_type_size(b_dt);

        B_blocked = new uint8_t[blocked_B_size * K_chunks];
        B_base_ptr = B_blocked;

        // Generate the executable code.
        pack_B.generate();

        // Pack B routine execution.
        // Note: usually should be split to process only a part of B that the
        // ukernel will execute.
        pack_B.execute(B_ptr, B_blocked);
    }

    // ukernel execution section.
    //
    // Prepare buffers for execution.
    std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(1);
    for (memory::dim i = 0; i < 1; i++) {
        const memory::dim A_offset_i = i * K_blk * a_dt_size;
        const memory::dim B_offset_i
                = need_pack ? i * blocked_B_size : i * N * K_blk * b_dt_size;
        A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
    }

    float *C_ptr = reinterpret_cast<float *>(C_mem.get_data_handle());

    // Query a scratchpad size and initialize a scratchpad buffer if the ukernel
    // is expecting it. This is a service space needed, has nothing in common
    // with accumulation buffer.
    size_t scratchpad_size
            = std::max(brg0.get_scratchpad_size(), brg1.get_scratchpad_size())
            + 64;
    std::vector<uint8_t> scratchpad(scratchpad_size);
    uint8_t *scratchpad_data = scratchpad.data();
    ptrdiff_t alignment = 64 - ((ptrdiff_t)scratchpad_data % 64);
    uint8_t *scratchpad_ptr = scratchpad_data + alignment;

    // A call to initialize hardware features. For example, prepare AMX unit.
    brg0.set_hw_context();

    // An execute call. `A_B_offsets` is a vector of pairs of offsets to A
    // and packed B tensors. `C_ptr` is a pointer to an accumulator buffer.
    brg0.execute(A_ptr, B_base_ptr, A_B_offsets, C_ptr, scratchpad_ptr);

    // Same set of operations for a ukernel with post-ops.
    std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets1(batch_size);
    for (memory::dim i = 0; i < batch_size; i++) {
        const memory::dim A_offset1 = batch_size * K_blk * a_dt_size;
        const memory::dim B_offset1 = need_pack
                ? batch_size * blocked_B_size
                : batch_size * N * K_blk * b_dt_size;
        A_B_offsets1.emplace_back(A_offset1, B_offset1);
    }

    if (brg1) {
        brg1.set_hw_context();
        brg1.execute(A_ptr, B_base_ptr, A_B_offsets1, C_ptr, scratchpad_ptr);
    }

    // Once all computations are done and there are no more calls to ukernels
    // until they delegate control to the application, need to release the
    // hardware context.
    brgemm::release_hw_context();

    // Clean up an extra buffer.
    delete B_blocked;

    // Used for verification results, need unconditional reorder.
    auto user_D_mem = memory(D_f32_md, engine, D_data.data());
    reorder(D_mem, user_D_mem).execute(engine_stream, D_mem, user_D_mem);

    // Skip the check by default as data filling doesn't help with proper
    // verification of the result. Negative result doesn't necessarily mean
    // the functionality is broken. This is just a general sanity check.
    if (true) return;

    // A simplified fast verification that ukernel returned expected results.
    // Note: potential off-by-1 or 2 errors may pop up. This could be solved
    // with more sparse filling.
    bool to_throw = false;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            D_user_data[m * N + n] = 0;
            for (int k = 0; k < K; k++) {
                D_user_data[m * N + n]
                        += A_user_data[m * K + k] * B_user_data[k * N + n];
            }

            const float diff
                    = fabsf(D_user_data[m * N + n] - D_data[m * N + n]);
            if (diff > 1.19e-7) {
                to_throw = true;
                if (true) {
                    printf("Error: [%3d:%3d] Ref:%12g Got:%12g Diff:%12g\n", m,
                            n, D_user_data[m * N + n], D_data[m * N + n], diff);
                }
            }
        }
    }
    if (to_throw) { throw status::runtime_error; }
}

int main(int argc, char **argv) {
    return handle_example_errors({dnnl::engine::kind::cpu}, brgemm_example);
}
