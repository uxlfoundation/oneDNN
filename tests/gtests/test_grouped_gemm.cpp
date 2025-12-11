
#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

class GroupedGEMM : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize DNNL engine and stream
        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        strm = dnnl::stream(eng);
    }

    void TearDown() override {
        // Cleanup code if needed
    }
    dnnl::engine eng;
    dnnl::stream strm;
};

using dnnl::matmul;
using std::unordered_map;

TEST_F(GroupedGEMM, BasicCreation) {
    const int num_groups = 2;
    const int num_experts = 4;
    const int K = 256;
    const int N = 256;
    const int M_all = 9; // total rows (tokens) across all experts
    const memory::data_type dt = memory::data_type::f32;

    memory::desc src_md = memory::desc::grouped(
            {M_all, K}, dt, num_groups, {DNNL_RUNTIME_DIM_VAL, K});

    auto weights_md
            = memory::desc({num_experts, K, N}, dt, memory::format_tag::abc);

    auto dst_md = memory::desc::grouped(
            {M_all, N}, dt, num_groups, {DNNL_RUNTIME_DIM_VAL, N});

    memory src_mem(src_md, eng);
    memory dst_mem(dst_md, eng);
    memory weights_mem(weights_md, eng);
    auto matmul_pd = matmul::primitive_desc(eng, src_md, weights_md, dst_md);
    auto matmul_prim = matmul(matmul_pd);

    unordered_map<int, memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    matmul_prim.execute(strm, args);
    strm.wait();
}
