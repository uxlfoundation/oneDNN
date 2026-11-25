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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl.h"

// A null descriptor passed to the binary post-op API is a caller error and has
// to be reported through the status code. It used to be dereferenced, which
// crashed the process for any driver that declared a binary post-op without
// supplying the operand.
TEST(primitive_attr_test, binary_post_op_null_descriptors) {
    const dnnl_dims_t dims = {2, 8};
    dnnl_memory_desc_t src1;
    ASSERT_EQ(
            dnnl_memory_desc_create_with_tag(&src1, 2, dims, dnnl_f32, dnnl_ab),
            dnnl_success);

    // src1 is mandatory for every binary algorithm, in particular for the
    // non-ternary one below which accepts a null src2.
    {
        dnnl_post_ops_t post_ops;
        ASSERT_EQ(dnnl_post_ops_create(&post_ops), dnnl_success);
        EXPECT_EQ(dnnl_post_ops_append_binary_v2(
                          post_ops, dnnl_binary_add, nullptr, nullptr),
                dnnl_invalid_arguments);
        EXPECT_EQ(dnnl_post_ops_len(post_ops), 0);
        dnnl_post_ops_destroy(post_ops);
    }

    // binary select additionally requires the second operand.
    {
        dnnl_post_ops_t post_ops;
        ASSERT_EQ(dnnl_post_ops_create(&post_ops), dnnl_success);
        EXPECT_EQ(dnnl_post_ops_append_binary_v2(
                          post_ops, dnnl_binary_select, src1, nullptr),
                dnnl_invalid_arguments);
        EXPECT_EQ(dnnl_post_ops_len(post_ops), 0);
        dnnl_post_ops_destroy(post_ops);
    }

    // A non-ternary algorithm only takes src1, so a null src2 stays valid.
    {
        dnnl_post_ops_t post_ops;
        ASSERT_EQ(dnnl_post_ops_create(&post_ops), dnnl_success);
        EXPECT_EQ(dnnl_post_ops_append_binary_v2(
                          post_ops, dnnl_binary_add, src1, nullptr),
                dnnl_success);
        EXPECT_EQ(dnnl_post_ops_len(post_ops), 1);
        dnnl_post_ops_destroy(post_ops);
    }

    dnnl_memory_desc_destroy(src1);
}
