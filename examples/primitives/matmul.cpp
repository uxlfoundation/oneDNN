/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
using namespace dnnl;

size_t div_up(const size_t a, const size_t b) {
    return (a + b - 1) / b;
}

size_t rnd_up(const size_t a, const size_t b) {
    return div_up(a, b) * b;
}

void *zmalloc_protect(size_t size) {
    const size_t page_sz = getpagesize();

    const size_t block_sz = size + 3 * sizeof(void *);
    const size_t total_sz = rnd_up(block_sz, page_sz) + page_sz;

    void *mem_ptr;
    int rc = ::posix_memalign(&mem_ptr, page_sz, total_sz);
    if (rc != 0) return nullptr;

    uint8_t *ptr_start = (uint8_t *)mem_ptr;
    uint8_t *ptr = ptr_start + total_sz - page_sz - size;

    // Aligned on a page boundary
    void *ptr_protect = ptr + size;

    // Layout of the allocated region:
    // ptr_start   <- start of the allocated region
    // ptr[-16]    <- stores start address: ptr_start
    // ptr[-8]     <- stores protected address: ptr_protect
    // ptr         <- pointer to be returned from the function
    // ptr_protect <- pointer to the block to protect

    // Protect one page right after the block of size bytes
    int err = mprotect(ptr_protect, page_sz, PROT_NONE);
    if (err != 0) {
        printf("Error: mprotect returned \'%s\'.\n", strerror(errno));
        ::free(ptr_start);
        return nullptr;
    }

    // Align down `ptr` on 8 bytes before storing addresses to make behavior
    // defined.
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = ptr - to_align;
    // Save pointers for zfree_protect
    ((void **)ptr_aligned_8)[-2] = ptr_start;
    ((void **)ptr_aligned_8)[-1] = ptr_protect;

    return ptr;
}

void zfree_protect(void *ptr) {
    // Get aligned ptr before obtaining addresses
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = reinterpret_cast<uint8_t *>(ptr) - to_align;

    // Restore read-write access for the protected region
    void *ptr_protect = ((void **)ptr_aligned_8)[-1];
    const size_t page_sz = getpagesize();
    mprotect(ptr_protect, page_sz, PROT_READ | PROT_WRITE);

    // Deallocate the whole region
    void *ptr_start = ((void **)ptr_aligned_8)[-2];
    ::free(ptr_start);
}

void matmul_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim MB1 = 1, MB2 = 16, // batch size
            M = 32, K = 256, N = 32;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {MB1, MB2, M, K};
    memory::dims weights_dims = {MB1, MB2, K, N};
    memory::dims sel_cond_dims = {1, 1, M, N};
    memory::dims dst_dims = {MB1, MB2, M, N};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, bias.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::abcd);
    auto weights_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::abcd);
    auto sel_src1_md = memory::desc(
            {1, 1, 1, 1}, memory::data_type::f32, memory::format_tag::abcd);
    auto sel_cond_md = memory::desc(
            sel_cond_dims, memory::data_type::s8, memory::format_tag::abcd);

    auto dst_md = memory::desc(
            dst_dims, memory::data_type::f32, memory::format_tag::abcd);

    auto src_mem = memory(src_md, engine, nullptr);
    auto weights_mem = memory(weights_md, engine, nullptr);
    auto sel_src1_mem = memory(sel_src1_md, engine, nullptr);
    auto sel_cond_mem = memory(sel_cond_md, engine, nullptr);
    auto dst_mem = memory(dst_md, engine, nullptr);

    auto ptr_src = zmalloc_protect(src_md.get_size());
    auto ptr_weights = zmalloc_protect(weights_md.get_size());
    auto ptr_sel_src1 = zmalloc_protect(sel_src1_md.get_size());
    auto ptr_sel_cond = zmalloc_protect(sel_cond_md.get_size());
    auto ptr_dst = zmalloc_protect(dst_md.get_size());

    src_mem.set_data_handle(ptr_src);
    weights_mem.set_data_handle(ptr_weights);
    sel_src1_mem.set_data_handle(ptr_sel_src1);
    sel_cond_mem.set_data_handle(ptr_sel_cond);
    dst_mem.set_data_handle(ptr_dst);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);

    // Create primitive post-ops (ReLU).
    post_ops matmul_ops;
    matmul_ops.append_binary(
            algorithm::binary_select, sel_src1_md, sel_cond_md);
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(
            engine, src_md, weights_md, dst_md, matmul_attr);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)0) | DNNL_ARG_SRC_1,
            sel_src1_mem});
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)0) | DNNL_ARG_SRC_2,
            sel_cond_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);

    zfree_protect(ptr_src);
    zfree_protect(ptr_weights);
    zfree_protect(ptr_sel_src1);
    zfree_protect(ptr_sel_cond);
    zfree_protect(ptr_dst);
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
