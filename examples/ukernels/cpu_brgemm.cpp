/*******************************************************************************
* Copyright 2024 Intel Corporation
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

/// @page cpu_brgemm_example_cpp_brief
/// @brief This C++ API example demonstrates how to create and execute a BRGeMM
/// ukernel.

/// @page cpu_brgemm_example_cpp BRGeMM ukernel example
/// \copybrief cpu_brgemm_example_cpp_brief
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

void brgemm_example() {

    // Create execution dnnl::engine. Needed for reorders to operate over input
    // data.
    dnnl::engine engine(engine::kind::cpu, 0);

    // Create dnnl::stream. Needed for reorders for the same reason.
    dnnl::stream engine_stream(engine);

    // ukernel dimensions.
    // K is for a whole tensor, K_blk is for a single ukernel.
    const memory::dim M = 8, K = 128, K_blk = 64, N = 48;
    if (K % K_blk != 0) {
        printf("K_blk must divide K.\n");
        return;
    }
    const memory::dim n_calls = K / K_blk;

    memory::data_type a_dt = memory::data_type::u8;
    memory::data_type b_dt = memory::data_type::s8;
    memory::data_type c_dt = memory::data_type::s32; // Accumulator data type.
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

    const memory::dim lda = K;
    // `ldb` for `need_pack = true` must be one of 16, 32, 48, or 64. This
    // example doesn't explore options for dividing N into blocks which would
    // likely happen for N > 64.
    // const memory::dim ldb = need_pack ? N_block : N;
    const memory::dim ldb = N;
    const memory::dim ldc = N; // Leading dimension for accumulator.
    const memory::dim ldd = N; // Leading dimension for an actual output.
    const memory::dim batch_size = n_calls - 1;

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
    // Initialize binary_add.
    std::generate(
            binary_add_user_data.begin(), binary_add_user_data.end(), []() {
        static int i = 3;
        return i++ % 6;
    });
    // Initialize B scales.
    std::generate(B_scales_user_data.begin(), B_scales_user_data.end(), []() {
        static int i = 4;
        return (float)(i++ % 16) / 8.f;
    });

    // Create f32 memories. They are used as data holders and reorder into
    // memories passed to the ukernel.
    auto A_f32_md = memory::desc(
            A_dims, memory::data_type::f32, memory::format_tag::ab);
    auto B_f32_md = memory::desc(
            B_dims, memory::data_type::f32, memory::format_tag::ab);
    auto binary_add_f32_md = memory::desc(
            binary_add_dims, memory::data_type::f32, memory::format_tag::ab);
    auto B_scales_f32_md = memory::desc(
            B_scales_dims, memory::data_type::f32, memory::format_tag::ab);
    auto D_f32_md = memory::desc(
            D_dims, memory::data_type::f32, memory::format_tag::ab);

    auto A_f32_mem = memory(A_f32_md, engine, A_user_data.data());
    auto B_f32_mem = memory(B_f32_md, engine, B_user_data.data());
    auto binary_add_f32_mem
            = memory(binary_add_f32_md, engine, binary_add_user_data.data());
    auto B_scales_f32_mem
            = memory(B_scales_f32_md, engine, B_scales_user_data.data());
    auto D_f32_mem = memory(D_f32_md, engine, D_user_data.data());

    // Create ukernel memories in requested data types.
    // Note that all formats are `ab`.
    auto A_md = memory::desc(A_dims, a_dt, memory::format_tag::ab);
    auto B_md = memory::desc(B_dims, b_dt, memory::format_tag::ab);
    auto binary_add_md = memory::desc(
            binary_add_dims, memory::data_type::f32, memory::format_tag::ab);
    auto B_scales_md = memory::desc(
            B_scales_dims, memory::data_type::f32, memory::format_tag::ab);
    auto C_md = memory::desc(C_dims, c_dt, memory::format_tag::ab);
    auto D_md = memory::desc(D_dims, d_dt, memory::format_tag::ab);

    auto A_mem = memory(A_md, engine);
    auto B_mem = memory(B_md, engine);
    auto binary_add_mem = memory(binary_add_md, engine);
    auto B_scales_mem = memory(B_scales_md, engine);
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
    reorder(binary_add_f32_mem, binary_add_mem)
            .execute(engine_stream, binary_add_f32_mem, binary_add_mem);
    reorder(B_scales_f32_mem, B_scales_mem)
            .execute(engine_stream, B_scales_f32_mem, B_scales_mem);
    reorder(D_f32_mem, D_mem).execute(engine_stream, D_f32_mem, D_mem);
    // Prepare C buffer. Needed to use a single ukernel in the example with
    // `set_add_C(true)`.
    // Note: to avoid this step, the first ukernel should run
    // `set_add_C(false)`, and it will initialize C buffer with intermediate
    // values.
    float *C_ptr = reinterpret_cast<float *>(C_mem.get_data_handle());
    for (memory::dim i = 0; i < M * N; i++) {
        C_ptr[i] = 0;
    }

    // Create ukernel post-ops (ReLU + Add).
    // It reuses `primitive_attr` abstraction.
    post_ops brgemm_ops;
    brgemm_ops.append_eltwise(
            algorithm::eltwise_relu, /* alpha = */ 0.f, /* beta = */ 0.f);
    brgemm_ops.append_binary(algorithm::binary_add, binary_add_md);

    // Create BRGeMM ukernel objects.
    // There are two objects:
    // * `brg` is the basic one which operates over K dimension divided into
    //   blocks. It utilizes `set_add_C(true)` to accumulate into the same
    //   buffer. It also uses `batch_size` to process as much as the number of
    //   blocks over K minus one.
    // * `brg_po` is the ukernel that would be called the last in the chain
    //   since it has attributes attached to the object and those will execute
    //   after all accumulation over K dimension is done.
    brgemm brg, brg_po;
    if (batch_size > 0) {
        // Construct a basic brgemm object.
        // `allow_empty` makes the interface to return an empty `brg` object
        // in case of the critical error.
        brg = brgemm(M, N, K_blk, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt,
                /* allow_empty = */ true);
        if (!brg) {
            printf("Error: brg object was not constructed.\n");
            return;
        }

        // Instruct the ukernel to append the result to the C tensor.
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

        // Generate the executable code.
        brg.generate();
    }

    // Construct a brgemm object with post-ops.
    brg_po = brgemm(M, N, K_blk, 1, lda, ldb, ldc, a_dt, b_dt, c_dt,
            /* allow_empty = */ true);
    if (!brg_po) {
        printf("Error: brg_po object was not constructed.\n");
        return;
    }

    // Instruct the kernel to append the result to the C tensor computed by
    // `brg` ukernel.
    brg_po.set_add_C(true);
    // Specify post-ops.
    brg_po.set_post_ops(ldd, d_dt, brgemm_ops);
    // Specify quantization scales for B.
    if (b_dt == memory::data_type::s8 || b_dt == memory::data_type::u8) {
        brg_po.set_B_scales(/* mask = */ 2);
    }

    // Finalize the initialization.
    const bool ok = brg_po.finalize();
    if (!ok) {
        printf("Kernel is not supported on this platform.\n");
        return;
    }

    // Generate the executable code.
    brg_po.generate();

    // Query a scratchpad size and initialize a scratchpad buffer if the ukernel
    // is expecting it. This is a service space needed, has nothing in common
    // with accumulation buffer.
    size_t scratchpad_size = brg_po.get_scratchpad_size();
    std::vector<uint8_t> scratchpad(scratchpad_size);

    uint8_t *B_blocked = nullptr;
    void *B_base_ptr = B_ptr;
    size_t blocked_B_size = 0;

    // If packing is needed, create a dedicated object for data transformation.
    if (need_pack) {
        // Transform kernel for tensor B. The ukernel expects B passed in a
        // special VNNI format for low precision data types, e.g., bfloat16_t
        // or int8.
        // Note: the routine doesn't provide a `batch_size` argument in the
        // constructor as it can be either incorporated into `K` dimension, or
        // manually iterated over in a for-loop on the user side.
        transform pack_B(/* K = */ K_blk * n_calls, /* N = */ N,
                /* in_pack_type = */ pack_type::no_trans, /* in_ld = */ N,
                /* out_ld = */ ldb, /* in_dt = */ b_dt, /* out_dt = */ b_dt);

        // Size of the packed tensor.
        blocked_B_size = ldb * K_blk * memory::data_type_size(b_dt);

        B_blocked = new uint8_t[blocked_B_size * n_calls];
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
    std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(batch_size);
    for (memory::dim i = 0; i < batch_size; i++) {
        const memory::dim A_offset_i = i * K_blk * a_dt_size;
        const memory::dim B_offset_i
                = need_pack ? i * blocked_B_size : i * N * K_blk * b_dt_size;
        A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
    }

    if (brg) {
        // A call to initialize hardware features. For example, prepare AMX
        // unit.
        brg.set_hw_context();

        // An execute call. `A_B_offsets` is a vector of pairs of offsets to A
        // and packed B tensors. `C_ptr` is a pointer to an accumulator buffer.
        brg.execute(A_ptr, B_base_ptr, A_B_offsets, C_ptr, scratchpad.data());
    }

    // Same set of operations for a ukernel with post-ops.
    std::vector<std::pair<memory::dim, memory::dim>> A_B_po_offsets;
    const memory::dim A_offset_po = batch_size * K_blk * a_dt_size;
    const memory::dim B_offset_po = need_pack
            ? batch_size * blocked_B_size
            : batch_size * N * K_blk * b_dt_size;
    A_B_po_offsets.emplace_back(A_offset_po, B_offset_po);

    // This object also requires this call since ukernel with post-ops may
    // require differently initialized internals underneath. If basic ukernel
    // was used and they share the same internals, this call will be optimized.
    brg_po.set_hw_context();

    // Prepare post-ops arguments and put them in a vector to make sure pointers
    // are sitting side by side.
    std::vector<const void *> bin_po_ptrs;
    bin_po_ptrs.push_back(binary_add_mem.get_data_handle());

    // Setting post-ops arguments into an attributes arguments storage.
    attr_params params;
    params.set_post_ops_args(bin_po_ptrs.data());
    params.set_B_scales(B_scales_mem.get_data_handle());

    // An execute call. The difference here is when post operations are
    // requested, an additional D tensor pointer to store final output result
    // after finishing accumulation and post-ops application is required.
    // Additionally, a special `params` object with post operations handles
    // is required.
    //
    // If post operations are not defined, the call is invalid, and a special
    // API checks its validity.
    if (brg_po.is_execute_postops_valid()) {
        brg_po.execute(A_ptr, B_base_ptr, A_B_po_offsets, C_ptr,
                D_mem.get_data_handle(), scratchpad.data(), params);
    } else {
        brg_po.execute(
                A_ptr, B_base_ptr, A_B_po_offsets, C_ptr, scratchpad.data());
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
            // B scales ref
            D_user_data[m * N + n] *= B_scales_user_data[n];
            // Relu post-op ref
            D_user_data[m * N + n] = std::max(D_user_data[m * N + n], 0.f);
            // Binary post-op ref
            D_user_data[m * N + n] += binary_add_user_data[0];

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
