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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"

namespace dnnl {

// RAII owner of a dnnl_memory_desc_t, constructed with format_tag_any.
// Converts implicitly to const_dnnl_memory_desc_t for use as call arguments.
struct md_t {
    dnnl_memory_desc_t md = nullptr;

    md_t(std::initializer_list<dnnl_dim_t> dims,
            dnnl_data_type_t dt = dnnl_f32) {
        std::vector<dnnl_dim_t> d(dims);
        dnnl_memory_desc_create_with_tag(&md, static_cast<int>(d.size()),
                d.data(), dt, dnnl_format_tag_any);
    }
    ~md_t() {
        if (md) dnnl_memory_desc_destroy(md);
    }
    md_t(const md_t &) = delete;
    md_t &operator=(const md_t &) = delete;
    operator const_dnnl_memory_desc_t() const { return md; }
};

// Destroys pd (if non-null) and returns the status unchanged.
static dnnl_status_t check_pd(dnnl_status_t st, dnnl_primitive_desc_t pd) {
    if (pd) dnnl_primitive_desc_destroy(pd);
    return st;
}

class rnn_prop_kind_test_t : public ::testing::Test {
protected:
    dnnl_engine_t engine_ = nullptr;

    void SetUp() override {
        SKIP_IF(dnnl_engine_get_count(dnnl_cpu) == 0, "CPU engine not found.");
        DNNL_CHECK(dnnl_engine_create(&engine_, dnnl_cpu, 0));
    }
    void TearDown() override {
        if (engine_) dnnl_engine_destroy(engine_);
    }
};

// ---------------------------------------------------------------------------
// Backward functions must reject forward prop_kinds
// ---------------------------------------------------------------------------

TEST_F(rnn_prop_kind_test_t, BackwardRejectsForwardProps) {
    // Prop_kind is the first check in rnn_common_bwd_desc_init, so we can
    // pass nullptr for engine and all descriptors — they are never reached.
    for (auto bad_pk : {dnnl_forward_training, dnnl_forward_inference}) {
        dnnl_primitive_desc_t pd = nullptr;

        // vanilla_rnn
        EXPECT_EQ(check_pd(dnnl_vanilla_rnn_backward_primitive_desc_create(&pd,
                                   nullptr, bad_pk, dnnl_eltwise_relu,
                                   dnnl_unidirectional_left2right, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, 0, 0.0f, 0.0f,
                                   nullptr, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // lstm
        EXPECT_EQ(
                check_pd(dnnl_lstm_backward_primitive_desc_create(&pd, nullptr,
                                 bad_pk, dnnl_unidirectional_left2right,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, 0, nullptr, nullptr),
                        pd),
                dnnl_invalid_arguments);

        // gru
        EXPECT_EQ(check_pd(dnnl_gru_backward_primitive_desc_create(&pd, nullptr,
                                   bad_pk, dnnl_unidirectional_left2right,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, 0,
                                   nullptr, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // lbr_gru
        EXPECT_EQ(check_pd(dnnl_lbr_gru_backward_primitive_desc_create(&pd,
                                   nullptr, bad_pk,
                                   dnnl_unidirectional_left2right, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, 0, nullptr,
                                   nullptr),
                          pd),
                dnnl_invalid_arguments);

        // augru
        EXPECT_EQ(
                check_pd(dnnl_augru_backward_primitive_desc_create(&pd, nullptr,
                                 bad_pk, dnnl_unidirectional_left2right,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, 0, nullptr, nullptr),
                        pd),
                dnnl_invalid_arguments);

        // lbr_augru
        EXPECT_EQ(check_pd(dnnl_lbr_augru_backward_primitive_desc_create(&pd,
                                   nullptr, bad_pk,
                                   dnnl_unidirectional_left2right, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr, nullptr,
                                   0, nullptr, nullptr),
                          pd),
                dnnl_invalid_arguments);
    }
}

// ---------------------------------------------------------------------------
// Backward functions must accept dnnl_backward
// ---------------------------------------------------------------------------
// Minimal f32 descriptors with format_tag_any satisfy every validation check
// after prop_kind.  The call will reach primitive_desc_create and return
// dnnl_success or dnnl_unimplemented — never dnnl_invalid_arguments.

TEST_F(rnn_prop_kind_test_t, BackwardAcceptsBackward) {
    const dnnl_rnn_direction_t dir = dnnl_unidirectional_left2right;

    // shapes: [T,N,SLC]==[T,N,DLC]==[1,1,1]
    //         [L,D,SLC,G,DHC] with SLC==SIC==DHC==DLC==1
    md_t io({1, 1, 1}); // src_layer / dst_layer / their diffs
    md_t wl_g1({1, 1, 1, 1, 1}); // weights for vanilla_rnn (G=1)
    md_t wi_g1({1, 1, 1, 1, 1});
    md_t wl_g3({1, 1, 1, 3, 1}); // weights for gru/augru (G=3)
    md_t wi_g3({1, 1, 1, 3, 1});
    md_t wl_g4({1, 1, 1, 4, 1}); // weights for lstm (G=4)
    md_t wi_g4({1, 1, 1, 4, 1});
    md_t attn({1, 1, 1}); // attention [T,N,1] for augru variants

    dnnl_primitive_desc_t pd = nullptr;

    // vanilla_rnn
    EXPECT_NE(check_pd(dnnl_vanilla_rnn_backward_primitive_desc_create(&pd,
                               engine_, dnnl_backward, dnnl_eltwise_relu, dir,
                               io, nullptr, wl_g1, wi_g1, nullptr, io, nullptr,
                               io, nullptr, wl_g1, wi_g1, nullptr, io, nullptr,
                               0, 0.0f, 0.0f, nullptr, nullptr),
                      pd),
            dnnl_invalid_arguments);

    // lstm
    EXPECT_NE(check_pd(dnnl_lstm_backward_primitive_desc_create(&pd, engine_,
                               dnnl_backward, dir, io, nullptr, nullptr, wl_g4,
                               wi_g4, nullptr, nullptr, nullptr, io, nullptr,
                               nullptr, io, nullptr, nullptr, wl_g4, wi_g4,
                               nullptr, nullptr, nullptr, io, nullptr, nullptr,
                               0, nullptr, nullptr),
                      pd),
            dnnl_invalid_arguments);

    // gru
    EXPECT_NE(check_pd(dnnl_gru_backward_primitive_desc_create(&pd, engine_,
                               dnnl_backward, dir, io, nullptr, wl_g3, wi_g3,
                               nullptr, io, nullptr, io, nullptr, wl_g3, wi_g3,
                               nullptr, io, nullptr, 0, nullptr, nullptr),
                      pd),
            dnnl_invalid_arguments);

    // lbr_gru
    EXPECT_NE(check_pd(dnnl_lbr_gru_backward_primitive_desc_create(&pd, engine_,
                               dnnl_backward, dir, io, nullptr, wl_g3, wi_g3,
                               nullptr, io, nullptr, io, nullptr, wl_g3, wi_g3,
                               nullptr, io, nullptr, 0, nullptr, nullptr),
                      pd),
            dnnl_invalid_arguments);

    // augru
    EXPECT_NE(check_pd(dnnl_augru_backward_primitive_desc_create(&pd, engine_,
                               dnnl_backward, dir, io, nullptr, attn, wl_g3,
                               wi_g3, nullptr, io, nullptr, io, nullptr, attn,
                               wl_g3, wi_g3, nullptr, io, nullptr, 0, nullptr,
                               nullptr),
                      pd),
            dnnl_invalid_arguments);

    // lbr_augru
    EXPECT_NE(check_pd(dnnl_lbr_augru_backward_primitive_desc_create(&pd,
                               engine_, dnnl_backward, dir, io, nullptr, attn,
                               wl_g3, wi_g3, nullptr, io, nullptr, io, nullptr,
                               attn, wl_g3, wi_g3, nullptr, io, nullptr, 0,
                               nullptr, nullptr),
                      pd),
            dnnl_invalid_arguments);
}

// ---------------------------------------------------------------------------
// Forward functions must reject dnnl_backward
// ---------------------------------------------------------------------------

TEST_F(rnn_prop_kind_test_t, ForwardRejectsBackwardProp) {
    dnnl_primitive_desc_t pd = nullptr;

    EXPECT_EQ(check_pd(dnnl_vanilla_rnn_forward_primitive_desc_create(&pd,
                               nullptr, dnnl_backward, dnnl_eltwise_relu,
                               dnnl_unidirectional_left2right, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                               0.0f, 0.0f, nullptr),
                      pd),
            dnnl_invalid_arguments);

    EXPECT_EQ(check_pd(dnnl_lstm_forward_primitive_desc_create(&pd, nullptr,
                               dnnl_backward, dnnl_unidirectional_left2right,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, 0, nullptr),
                      pd),
            dnnl_invalid_arguments);

    EXPECT_EQ(check_pd(dnnl_gru_forward_primitive_desc_create(&pd, nullptr,
                               dnnl_backward, dnnl_unidirectional_left2right,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, 0, nullptr),
                      pd),
            dnnl_invalid_arguments);

    EXPECT_EQ(check_pd(dnnl_lbr_gru_forward_primitive_desc_create(&pd, nullptr,
                               dnnl_backward, dnnl_unidirectional_left2right,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, 0, nullptr),
                      pd),
            dnnl_invalid_arguments);

    EXPECT_EQ(check_pd(dnnl_augru_forward_primitive_desc_create(&pd, nullptr,
                               dnnl_backward, dnnl_unidirectional_left2right,
                               nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, 0, nullptr),
                      pd),
            dnnl_invalid_arguments);

    EXPECT_EQ(
            check_pd(dnnl_lbr_augru_forward_primitive_desc_create(&pd, nullptr,
                             dnnl_backward, dnnl_unidirectional_left2right,
                             nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, 0, nullptr),
                    pd),
            dnnl_invalid_arguments);
}

// ---------------------------------------------------------------------------
// Forward functions must accept forward prop_kinds
// ---------------------------------------------------------------------------

TEST_F(rnn_prop_kind_test_t, ForwardAcceptsForwardProps) {
    const dnnl_rnn_direction_t dir = dnnl_unidirectional_left2right;

    md_t io({1, 1, 1});
    md_t wl_g1({1, 1, 1, 1, 1});
    md_t wi_g1({1, 1, 1, 1, 1});
    md_t wl_g3({1, 1, 1, 3, 1});
    md_t wi_g3({1, 1, 1, 3, 1});
    md_t wl_g4({1, 1, 1, 4, 1});
    md_t wi_g4({1, 1, 1, 4, 1});
    md_t attn({1, 1, 1});

    dnnl_primitive_desc_t pd = nullptr;

    for (auto good_pk : {dnnl_forward_training, dnnl_forward_inference}) {

        // vanilla_rnn
        EXPECT_NE(check_pd(dnnl_vanilla_rnn_forward_primitive_desc_create(&pd,
                                   engine_, good_pk, dnnl_eltwise_relu, dir, io,
                                   nullptr, wl_g1, wi_g1, nullptr, io, nullptr,
                                   0, 0.0f, 0.0f, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // lstm
        EXPECT_NE(check_pd(dnnl_lstm_forward_primitive_desc_create(&pd, engine_,
                                   good_pk, dir, io, nullptr, nullptr, wl_g4,
                                   wi_g4, nullptr, nullptr, nullptr, io,
                                   nullptr, nullptr, 0, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // gru
        EXPECT_NE(check_pd(dnnl_gru_forward_primitive_desc_create(&pd, engine_,
                                   good_pk, dir, io, nullptr, wl_g3, wi_g3,
                                   nullptr, io, nullptr, 0, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // lbr_gru
        EXPECT_NE(check_pd(dnnl_lbr_gru_forward_primitive_desc_create(&pd,
                                   engine_, good_pk, dir, io, nullptr, wl_g3,
                                   wi_g3, nullptr, io, nullptr, 0, nullptr),
                          pd),
                dnnl_invalid_arguments);

        // augru
        EXPECT_NE(
                check_pd(dnnl_augru_forward_primitive_desc_create(&pd, engine_,
                                 good_pk, dir, io, nullptr, attn, wl_g3, wi_g3,
                                 nullptr, io, nullptr, 0, nullptr),
                        pd),
                dnnl_invalid_arguments);

        // lbr_augru
        EXPECT_NE(check_pd(dnnl_lbr_augru_forward_primitive_desc_create(&pd,
                                   engine_, good_pk, dir, io, nullptr, attn,
                                   wl_g3, wi_g3, nullptr, io, nullptr, 0,
                                   nullptr),
                          pd),
                dnnl_invalid_arguments);
    }
}

} // namespace dnnl
