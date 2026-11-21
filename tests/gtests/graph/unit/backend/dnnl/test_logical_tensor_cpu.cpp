/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "graph/unit/utils.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "interface/backend.hpp"
#include "interface/logical_tensor.hpp"

namespace graph = dnnl::impl::graph;
namespace dnnl_impl = dnnl::impl::graph::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_logical_tensor, ImplicitEqualLayout) {
    using ltw = graph::logical_tensor_wrapper_t;

    dnnl::memory::desc md({1, 2, 3, 4}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::nchw);
    auto layout_idx
            = dnnl_impl::dnnl_backend_t::get_singleton().set_mem_desc(md);
    ASSERT_TRUE(layout_idx.has_value());
    auto backend_idx = dnnl_impl::dnnl_backend_t::get_singleton().get_id();
    auto id = graph::backend_registry_t::encode_layout_id(
            layout_idx.value(), backend_idx);

    graph::logical_tensor_t lt1 = utils::logical_tensor_init(
            0, {1, 2, 3, 4}, graph::data_type::f32, graph::layout_type::any);
    // set opaque layout id
    lt1.layout_type = graph::layout_type::opaque;
    lt1.layout.layout_id = id;

    // public layout
    graph::logical_tensor_t lt2 = utils::logical_tensor_init(0, {1, 2, 3, 4},
            graph::data_type::f32, graph::layout_type::strided);

    ASSERT_TRUE(ltw(lt1).has_same_layout_as(ltw(lt2)));
    ASSERT_TRUE(ltw(lt2).has_same_layout_as(ltw(lt1)));
}

TEST(test_logical_tensor, MakeMemoryDescRejectsUnregisteredLayoutId) {
    // A logical tensor may carry an opaque layout id that was never registered
    // in the backend - either because a caller built one by hand or because a
    // pass produced it. Looking such a tensor up must report the mistake; it
    // used to dereference an empty optional and take the process down.
    const auto backend_idx
            = dnnl_impl::dnnl_backend_t::get_singleton().get_id();

    // The layout id manager is a process-wide singleton, so a fixed number could
    // be registered by an earlier test. Walk up until an index is provably free.
    size_t unregistered_idx = 4096;
    while (dnnl_impl::dnnl_backend_t::get_singleton()
                    .get_mem_desc(unregistered_idx)
                    .has_value()
            && unregistered_idx < 100000) {
        ++unregistered_idx;
    }
    ASSERT_LT(unregistered_idx, 100000)
            << "could not find a free layout id index";
    const size_t encoded_id = graph::backend_registry_t::encode_layout_id(
            unregistered_idx, backend_idx);

    graph::logical_tensor_t lt = utils::logical_tensor_init(
            0, {64}, graph::data_type::f32, graph::layout_type::strided);
    // the partition decodes the id before the backend sees the tensor
    lt.layout_type = graph::layout_type::opaque;
    lt.layout.layout_id
            = graph::backend_registry_t::decode_layout_id(encoded_id).first;
    ASSERT_EQ(lt.layout.layout_id, unregistered_idx);
    ASSERT_FALSE(dnnl_impl::dnnl_backend_t::get_singleton()
                         .get_mem_desc(lt.layout.layout_id)
                         .has_value());

    // The message has to say what went wrong: a plain any_cast failure on an
    // empty optional says "bad any cast" and hides where the bad id came from.
    try {
        dnnl_impl::make_dnnl_memory_desc(lt);
        FAIL() << "an unregistered layout id must not be accepted";
    } catch (const std::exception &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("unregistered layout id"), std::string::npos)
                << "unexpected message: " << msg;
    }
}
