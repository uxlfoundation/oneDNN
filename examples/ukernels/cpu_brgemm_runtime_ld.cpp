/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

/// @example cpu_brgemm.cpp
/// > Annotated version: @ref cpu_brgemm_example_cpp
///
/// @page cpu_brgemm_example_cpp BRGeMM ukernel example
/// This C++ API example demonstrates how to create and execute a BRGeMM
/// ukernel.
///
/// @include cpu_brgemm.cpp

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

using tag = memory::format_tag;
using dt = memory::data_type;

void brgemm_example() {

    // Create execution dnnl::engine. Needed for reorders to operate over input
    // data.
    dnnl::engine engine(engine::kind::cpu, 0);

    // Create dnnl::stream. Needed for reorders for the same reason.
    dnnl::stream engine_stream(engine);

    // ukernel dimensions.
    // K is for a whole tensor, K_blk is for a single ukernel.
    const memory::dim M = 32, K = 32, N = 32;
    const memory::dim M_blk = 32, K_blk = 32, N_blk = 32;
    if (K % K_blk != 0 || M % M_blk != 0 || N % N_blk != 0) {
        printf("Dimensions must divide their blocks.\n");
        return;
    }
    const memory::dim n_M_blocks = M / M_blk;
    const memory::dim n_N_blocks = N / N_blk;
    const memory::dim n_K_blocks = K / K_blk;

    memory::data_type a_dt = dt::u8;
    memory::data_type b_dt = dt::s8;
    memory::data_type c_dt = dt::s32; // Accumulator data type.
    memory::data_type d_dt = dt::f32; // Output data type.
    using a_type = uint8_t;
    using b_type = int8_t;
    using c_type = int;
    using d_type = uint8_t;
#define PRINT_FMT "%d "

    // memory::data_type a_dt = dt::f32;
    // memory::data_type b_dt = dt::f32;
    // memory::data_type c_dt = dt::f32; // Accumulator data type.
    // memory::data_type d_dt = dt::f32; // Output data type.
    // using a_type = float;
    // using b_type = float;
    // using c_type = float;
    // using d_type = float;
    // #define PRINT_FMT "%1.f "

    // Query the packing requirement from the ukernel. It's enough to query
    // packing requirements once for multiple objects.
    // Based on this information, specific `create_ldb` value can be used, since
    // transform has a limited set of values supported.
    const bool need_pack
            = brgemm::get_B_pack_type(a_dt, b_dt) == pack_type::pack32;

    constexpr bool use_runtime_ld = true;

    const memory::dim execute_lda = K;
    const memory::dim create_lda
            = use_runtime_ld ? DNNL_RUNTIME_DIM_VAL : execute_lda;
    // `create_pack_ldb` for `need_pack = true` must be one of 16, 32, 48, or 64.
    if (need_pack
            && !(N_blk == 16 || N_blk == 32 || N_blk == 48 || N_blk == 64)) {
        printf("N_block must be one of 16, 32, 48, or 64.\n");
        return;
    }

    const memory::dim execute_ldb = N; // Assuming `pack::no_trans` for B.
    const memory::dim create_ldb
            = use_runtime_ld ? DNNL_RUNTIME_DIM_VAL : execute_ldb;
    const memory::dim create_pack_ldb = need_pack ? N_blk : create_ldb;

    // Leading dimension for accumulator.
    const memory::dim execute_ldc = N;
    const memory::dim create_ldc
            = use_runtime_ld ? DNNL_RUNTIME_DIM_VAL : execute_ldc;

    // Leading dimension for an actual output.
    const memory::dim execute_ldd = N;
    const memory::dim create_ldd
            = use_runtime_ld ? DNNL_RUNTIME_DIM_VAL : execute_ldd;

    // A, B, and C tensors dimensions.
    memory::dims A_dims = {M, K};
    memory::dims B_dims = {K, N};
    memory::dims C_dims = {M, N};
    memory::dims D_dims = {M, N};
    memory::dims binary_add_dims = {1, 1};
    memory::dims B_scales_dims = {1, N};

    // Allocate buffers with user data.
    std::vector<float> A_user_data(product(A_dims));
    std::vector<float> B_user_data(product(B_dims));
    std::vector<float> binary_add_user_data(product(binary_add_dims));
    std::vector<float> B_scales_user_data(product(B_scales_dims));
    std::vector<float> D_data(product(D_dims)); // For reference comparison
    std::vector<float> D_user_data(product(D_dims)); // For reference comparison

    // Initialize A.
    std::generate(A_user_data.begin(), A_user_data.end(), [&]() {
        static int i = 0;
        return i++ % 6;
        // static int k = 0;
        // return (k++ % K) < (K / 2) ? 1.f : 2.f;
    });
    // Initialize B.
    std::generate(B_user_data.begin(), B_user_data.end(), [&]() {
        static int i = 0;
        static int sign_gen = 0;
        int sign = (sign_gen++ % 2) ? -1 : 1;
        float val = sign * (i++ % 5);
        return val;
        // static int i = 0;
        // return i++ < (K / 2) * N ? 1.f : 2.f;
    });
    // Initialize binary_add.
    std::generate(
            binary_add_user_data.begin(), binary_add_user_data.end(), []() {
                static int i = 3;
                return i++ % 6;
            });
    // Initialize B scales.
    std::generate(B_scales_user_data.begin(), B_scales_user_data.end(), []() {
        // static int i = 4;
        // return (float)(i++ % 16) / 8.f;
        return 1.f;
    });

    // Create f32 memories. They are used as data holders and reorder into
    // memories passed to the ukernel.
    auto A_f32_md = memory::desc(A_dims, dt::f32, tag::ab);
    auto B_f32_md = memory::desc(B_dims, dt::f32, tag::ab);
    auto binary_add_f32_md = memory::desc(binary_add_dims, dt::f32, tag::ab);
    auto B_scales_f32_md = memory::desc(B_scales_dims, dt::f32, tag::ab);
    auto D_f32_md = memory::desc(D_dims, dt::f32, tag::ab);

    auto A_f32_mem = memory(A_f32_md, engine, A_user_data.data());
    auto B_f32_mem = memory(B_f32_md, engine, B_user_data.data());
    auto binary_add_f32_mem
            = memory(binary_add_f32_md, engine, binary_add_user_data.data());
    auto B_scales_f32_mem
            = memory(B_scales_f32_md, engine, B_scales_user_data.data());
    auto D_f32_mem = memory(D_f32_md, engine, D_user_data.data());

    // Create ukernel memories in requested data types.
    // Note that all formats are `ab`.
    auto A_md = memory::desc(A_dims, a_dt, tag::ab);
    auto B_md = memory::desc(B_dims, b_dt, tag::ab);
    auto binary_add_md = memory::desc(binary_add_dims, dt::f32, tag::ab);
    auto B_scales_md = memory::desc(B_scales_dims, dt::f32, tag::ab);
    auto C_md = memory::desc(C_dims, c_dt, tag::ab);
    auto D_md = memory::desc(D_dims, d_dt, tag::ab);

    auto A_mem = memory(A_md, engine);
    auto B_mem = memory(B_md, engine);
    auto binary_add_mem = memory(binary_add_md, engine);
    auto B_scales_mem = memory(B_scales_md, engine);
    auto C_mem = memory(C_md, engine);
    auto D_mem = memory(D_md, engine);

    const auto *A_ptr = reinterpret_cast<a_type *>(A_mem.get_data_handle());
    auto *B_ptr = reinterpret_cast<b_type *>(B_mem.get_data_handle());

    // const size_t a_dt_size
    //         = memory::data_type_size(A_mem.get_desc().get_data_type());
    // const size_t b_dt_size
    //         = memory::data_type_size(B_mem.get_desc().get_data_type());
    // const size_t d_dt_size
    //         = memory::data_type_size(D_mem.get_desc().get_data_type());

    // Reorder user data into buffers passed to ukernels in target data types.
    reorder(A_f32_mem, A_mem).execute(engine_stream, A_f32_mem, A_mem);
    reorder(B_f32_mem, B_mem).execute(engine_stream, B_f32_mem, B_mem);
    reorder(binary_add_f32_mem, binary_add_mem)
            .execute(engine_stream, binary_add_f32_mem, binary_add_mem);
    reorder(B_scales_f32_mem, B_scales_mem)
            .execute(engine_stream, B_scales_f32_mem, B_scales_mem);
    reorder(D_f32_mem, D_mem).execute(engine_stream, D_f32_mem, D_mem);
    // Prepare C buffer. Needed to use a single ukernel in the example with
    // `beta = 1.f`.
    // Note: to avoid this step, the first ukernel should run `beta = 0`, and it
    // will initialize C buffer with intermediate values.
    auto *C_ptr = reinterpret_cast<c_type *>(C_mem.get_data_handle());
    for (memory::dim i = 0; i < M * N; i++) {
        C_ptr[i] = 0;
    }

    auto *D_ptr = reinterpret_cast<d_type *>(D_mem.get_data_handle());

    // Create BRGeMM ukernel objects.
    // There are two objects:
    // * `brg` is the main one which operates over partitioned K dimension. It
    //   utilizes `beta = 1.f` to accumulate into the same buffer. Uses
    //   `n_K_blocks - 1` to process iterations.
    // * `brg_po` is the ukernel that would be called the last in the chain
    //   since it has attributes attached to the object and those will execute
    //   after all accumulation over K dimension is done.
    // Note: `beta = 1.f` makes a ukernel reusable over K but will require
    // zeroing the correspondent piece of accumulation buffer.
    brgemm brg, brg_po;
    if (n_K_blocks > 1) {
        // Construct a basic brgemm object.
        // `allow_empty` makes the interface to return an empty `brg` object
        // in case of the critical error.
        // TODO: restore batch_size. Currently, 1 is used for code
        // simplicity.
        brg = brgemm(M_blk, N_blk, K_blk, 1 /*n_K_blocks - 1*/, create_lda,
                create_pack_ldb, create_ldc, a_dt, b_dt, c_dt,
                /* allow_empty = */ true);
        if (!brg) {
            printf("Error: brg object was not constructed.\n");
            return;
        }

        // Instruct the kernel to append the result to C tensor.
        brg.set_add_C(true);

        // Finalize the initialization.
        // Successful completion returns `true`. Otherwise, `brg` object can't
        // be used due to lack of support or non-compatible settings. The
        // specific reason may be found by using `ONEDNN_VERBOSE=all` env var.
        const bool ok = brg.finalize();
        if (!ok) {
            printf("Kernel is not supported on this platform.\n");
            return;
        }

        // Generate the executable JIT code for the objects.
        brg.generate();
    }

    // Construct a brgemm object with post-ops.
    brg_po = brgemm(M_blk, N_blk, K_blk, 1, create_lda, create_pack_ldb,
            create_ldc, a_dt, b_dt, c_dt, /* allow_empty = */ true);
    if (!brg_po) {
        printf("Error: brg_po object was not constructed.\n");
        return;
    }

    // Instruct the kernel to append the result to the C tensor computed by
    // `brg` ukernel.
    brg_po.set_add_C(true);
    // Specify post-ops.
    brg_po.set_post_ops(create_ldd, d_dt);
    // Specify quantization scales for B.
    if (b_dt == dt::s8 || b_dt == dt::u8) {
        brg_po.set_B_scales(/* mask = */ 2);
    }

    // Finalize the initialization.
    const bool ok = brg_po.finalize();
    if (!ok) {
        printf("Kernel is not supported on this platform.\n");
        return;
    }

    // Generate the executable JIT code for the objects.
    brg_po.generate();

    // Query a scratchpad size and initialize a scratchpad buffer if the ukernel
    // is expecting it. This is a service space needed, has nothing in common
    // with accumulation buffer.
    size_t scratchpad_size = brg_po.get_scratchpad_size();
    std::vector<uint8_t> scratchpad(scratchpad_size);

    b_type *B_blocked = nullptr;
    b_type *B_base_ptr = B_ptr;
    size_t blocked_B_size = 0;

    // If packing is needed, create a dedicated object for data transformation.
    transform pack_B;
    if (need_pack) {
        // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
        // special VNNI format for low precision data types, e.g., bfloat16_t.
        // Note: the routine doesn't provide a `n_K_blocks - 1` argument in the
        // constructor as it can be either incorporated into `K` dimension, or
        // manually iterated over in a for-loop on the user side.
        pack_B = transform(/* K = */ K_blk, /* N = */ N_blk,
                /* in_pack_type = */ pack_type::no_trans,
                /* in_ld = */ create_ldb,
                /* out_ld = */ create_pack_ldb, /* in_dt = */ b_dt,
                /* out_dt = */ b_dt);

        // Size of the packed tensor.
        blocked_B_size = create_pack_ldb * K_blk;

        B_blocked = new b_type[blocked_B_size];
        B_base_ptr = B_blocked;

        // Pack B routine execution.
        // Note: usually should be split to process only that part of B that the
        // ukernel will execute.

        pack_B.generate();
    }

    // BRGeMM ukernel execute section.
    for (memory::dim m = 0; m < n_M_blocks; m++)
        for (memory::dim n = 0; n < n_N_blocks; n++)
            for (memory::dim k = 0; k < n_K_blocks; k++) {
                // Compute offsets
                const memory::dim A_offset
                        = m * M_blk * execute_lda + k * K_blk;
                const memory::dim B_offset
                        = n * N_blk + k * K_blk * execute_ldb;
                const memory::dim B_pack_offset = need_pack
                        ? 0 // Always same buffer
                        : B_offset;
                const memory::dim C_offset
                        = m * M_blk * execute_ldc + n * N_blk;
                const memory::dim D_offset
                        = m * M_blk * execute_ldd + n * N_blk;

                printf("B tensor:\n");
                for (int kk = 0; kk < K_blk; kk++) {
                    for (int nn = 0; nn < N_blk; nn++)
                        printf(PRINT_FMT,
                                (B_ptr + B_offset)[kk * execute_ldb + nn]);
                    printf("\n");
                }
                printf("\n");

                if (need_pack) {
                    pack_B.execute(B_ptr + B_offset, B_base_ptr, execute_ldb);
                }

                printf("B packed tensor:\n");
                for (int kk = 0; kk < K_blk / 4; kk++) {
                    for (int nn = 0; nn < N_blk * 4; nn++)
                        printf(PRINT_FMT, B_base_ptr[kk * N + nn]);
                    printf("\n");
                }
                printf("\n");

                printf("A tensor:\n");
                for (int mm = 0; mm < M_blk; mm++) {
                    for (int kk = 0; kk < K_blk; kk++)
                        printf(PRINT_FMT,
                                (A_ptr + A_offset)[mm * execute_lda + kk]);
                    printf("\n");
                }
                printf("\n");

                std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(1);
                A_B_offsets[0] = std::make_pair(A_offset, B_pack_offset);

                printf("execute_lds[] = {%ld, %ld, %ld, %ld}\n", execute_lda,
                        execute_ldb, execute_ldc, execute_ldd);
                std::vector<memory::dim> actual_lds {
                        execute_lda, execute_ldb, execute_ldc, execute_ldd};

                if (k != n_K_blocks - 1) {
                    if (!brg) {
                        printf("Error: brg object is not found!\n");
                        return;
                    }
                    // Make an object to call HW specialized routines.
                    // For example, prepare AMX unit.
                    brg.set_hw_context();

                    // An execute call. `A_B` is a vector of pointers to A and
                    // packed B tensors. `acc_ptr` is a pointer to an
                    // accumulator buffer.
                    brg.execute(A_ptr, B_base_ptr, A_B_offsets,
                            C_ptr + C_offset, scratchpad.data(), actual_lds);

                    for (int mm = 0; mm < M_blk; mm++) {
                        for (int nn = 0; nn < N_blk; nn++)
                            printf(PRINT_FMT,
                                    (C_ptr + C_offset)[mm * execute_ldc + nn]);
                        printf("\n");
                    }
                } else {
                    // This object also requires this call.
                    brg_po.set_hw_context();

                    // Setting post-ops arguments into an attributes arguments
                    // storage.
                    attr_params params;
                    params.set_B_scales(
                            B_scales_mem.get_data_handle() /* TODO: + N_blk*/);

                    // An execute call. The difference here is when post operations
                    // are requested, an additional D tensor pointer to store final
                    // output result after finishing accumulation and post-ops
                    // application is required.
                    // Additionally, a special `params` object with post operations
                    // handles is required.
                    //
                    // If post operations are not defined, the call is invalid, and
                    // a special API checks the state.
                    if (brg_po.is_execute_postops_valid()) {
                        brg_po.execute(A_ptr, B_base_ptr, A_B_offsets,
                                C_ptr + C_offset, D_ptr + D_offset,
                                scratchpad.data(), params, actual_lds);
                    } else {
                        brg_po.execute(A_ptr, B_base_ptr, A_B_offsets,
                                C_ptr + C_offset, scratchpad.data(),
                                actual_lds);
                    }
                    // for (int mm = 0; mm < M; mm++) {
                    //     for (int nn = 0; nn < N; nn++)
                    //         printf(PRINT_FMT, (D_ptr + D_offset)[mm * execute_ldd + nn]);
                    //     printf("\n");
                    // }
                }
            }

    // // Same set of operations for a ukernel with post-ops.
    // std::vector<std::pair<memory::dim, memory::dim>> A_B_po_offsets;
    // const memory::dim A_offset_po = (n_K_blocks - 1) * K_blk * a_dt_size;
    // const memory::dim B_offset_po = need_pack
    //         ? (n_K_blocks - 1) * blocked_B_size
    //         : (n_K_blocks - 1) * N * K_blk * b_dt_size;
    // A_B_po_offsets.emplace_back(A_offset_po, B_offset_po);

    // // This object also requires this call.
    // brg_po.set_hw_context();

    // // Prepare post-ops arguments and put them in a vector to make sure pointers
    // // are sitting side by side.
    // std::vector<const void *> bin_po_ptrs;
    // bin_po_ptrs.push_back(binary_add_mem.get_data_handle());

    // // Setting post-ops arguments into an attributes arguments storage.
    // attr_params params;
    // params.set_post_ops_args(bin_po_ptrs.data());
    // params.set_B_scales(B_scales_mem.get_data_handle());

    // // An execute call. The difference here is when post operations are
    // // requested, an additional D tensor pointer to store final output result
    // // after finishing accumulation and post-ops application is required.
    // // Additionally, a special `params` object with post operations handles
    // // is required.
    // //
    // // If post operations are not defined, the call is invalid, and a special
    // // API checks the state.
    // if (brg_po.is_execute_postops_valid()) {
    //     brg_po.execute(A_ptr, B_base_ptr, A_B_po_offsets, C_ptr,
    //             D_mem.get_data_handle(), scratchpad.data(), params);
    // } else {
    //     brg_po.execute(
    //             A_ptr, B_base_ptr, A_B_po_offsets, C_ptr, scratchpad.data());
    // }

    // Once all computations are done, need to release HW context.
    brgemm::release_hw_context();

    // Clean up an extra buffer.
    delete B_blocked;

    // Used for verification results, need unconditional reorder.
    auto user_D_mem = memory(D_f32_md, engine, D_data.data());
    if (brg_po.is_execute_postops_valid()) {
        reorder(D_mem, user_D_mem).execute(engine_stream, D_mem, user_D_mem);
    } else {
        reorder(C_mem, user_D_mem).execute(engine_stream, C_mem, user_D_mem);
    }

    // Skip the check by default as data filling doesn't help with proper
    // verification of the result. Negative result doesn't necessarily mean
    // the functionality is broken. This is just a general sanity check.
    // if (true) return;

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
            // B scales ref
            D_user_data[m * N + n] *= B_scales_user_data[n];
            // // Relu post-op ref
            // D_user_data[m * N + n] = std::max(D_user_data[m * N + n], 0.f);
            // // Binary post-op ref
            // D_user_data[m * N + n] += binary_add_user_data[0];

            const bool print_me = false;
            const float diff
                    = fabsf(D_user_data[m * N + n] - D_data[m * N + n]);
            bool ok = diff <= 1.19e-7;
            if (!ok || print_me) {
                to_throw = !ok;
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
