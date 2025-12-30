
#include "dnnl_test_common.hpp"
#include "internals/test_utils.cpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

class GroupedGEMM : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize DNNL engine and stream
        eng = dnnl::engine(dnnl::engine::kind::gpu, 0);
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
    const int K = 128;
    const int N = 512;
    std::vector<int> M_group_sizes
            = {256, 128, 64, 48, 96, 512, 32, 32}; // rows per group
    const int num_groups = M_group_sizes.size();
    std::vector<int> M_group_offsets(num_groups + 1, 0);
    std::inclusive_scan(begin(M_group_sizes), end(M_group_sizes),
            begin(M_group_offsets) + 1);
    const int M_all = M_group_offsets.back();

    const memory::dims src_sz = {M_all, K};
    const memory::dims weights_sz = {num_groups, K, N};
    const memory::dims dst_sz = {M_all, N};

    const memory::data_type sdt = memory::data_type::f16;
    const memory::data_type wdt = memory::data_type::f16;
    const memory::data_type ddt = memory::data_type::f32;

    memory::desc src_md = memory::desc::grouped(src_sz, sdt, 0, num_groups);
    auto weights_md = memory::desc(weights_sz, wdt, memory::format_tag::abc);
    auto dst_md = memory::desc::grouped(dst_sz, ddt, 0, num_groups);

    memory src_mem(src_md, eng);
    memory dst_mem(dst_md, eng);
    memory weights_mem(weights_md, eng);

    std::vector<float> src_data(product(src_sz));
    std::vector<float> weights_data(product(weights_sz));

#if 0
    // Fill src and weights with sequential data for easier debugging
    for (int i = 0, group = 0; i < (int)src_data.size(); ++i) {
        if (i >= M_group_offsets[group + 1] * K) ++group;
        src_data[i] = i / K
                * 0.1f; //static_cast<float>(i % 13 + 1); // values from 1 to 13
    }
    for (int i = 0; i < (int)weights_data.size(); ++i) {
        //int group = i / (K * N);
        weights_data[i] = 1; //i / N * 0.1f; // * (group + 1);
                //= static_cast<float>((i % 7) - 3.f); // values from -3 to 3
    }

#else
    fill_random(src_data, src_md);
    fill_random(weights_data, weights_md);
#endif
    write_to_dnnl_memory(src_data.data(), src_mem, eng, strm);
    write_to_dnnl_memory(weights_data.data(), weights_mem, eng, strm);
    write_to_dnnl_memory(M_group_offsets.data(), src_mem, eng, strm, 1);
    write_to_dnnl_memory(M_group_offsets.data(), dst_mem, eng, strm, 1);
    std::vector<memory> src_groups;
    std::vector<memory> weights_groups;
    std::vector<memory> dst_groups;
    std::vector<matmul::primitive_desc> matmul_pds;
    std::vector<matmul> matmul_prims;

    //print_mem(src_mem, "src");
    //print_mem(weights_mem, "weights");
    //print_mem(dst_mem, "dst");

    for (int g = 0; g < num_groups; ++g) {
        src_groups.push_back(get_group_mem<float16_t>(src_mem, g));
        dst_groups.push_back(get_group_mem<float>(dst_mem, g));
        const memory::dims w_sz = {weights_sz[1], weights_sz[2]};
        weights_groups.push_back(
                memory(memory::desc(w_sz, wdt, memory::format_tag::ab), eng));
        // Copy weights for group
        size_t weights_group_offset = g * K * N;
        write_to_dnnl_memory(&weights_data[weights_group_offset],
                weights_groups[g], eng, strm);
    }

    auto matmul_pd = matmul::primitive_desc(eng, src_md, weights_md, dst_md);
    auto matmul_prim = matmul(matmul_pd);

    std::vector<matmul::primitive_desc> pds;
    std::vector<matmul> prims;
    for (int g = 0; g < num_groups; ++g) {
        auto pd = matmul::primitive_desc(eng, src_groups[g].get_desc(),
                weights_groups[g].get_desc(), dst_groups[g].get_desc());
        pds.push_back(pd);
        prims.push_back(matmul(pd));
        unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_groups[g]});
        args.insert({DNNL_ARG_WEIGHTS, weights_groups[g]});
        args.insert({DNNL_ARG_DST, dst_groups[g]});
        prims[g].execute(strm, args);
        strm.wait();
    }

    unordered_map<int, memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    matmul_prim.execute(strm, args);
    strm.wait();

    //print_mem(dst_mem, "dst");
    for (int g = 0; g < num_groups; ++g) {
        //print_mem(dst_groups[g], "gold");
        auto dst_test = get_group_mem<float>(dst_mem, g);
        auto dst_gold = dst_groups[g];
        float *mapped_ptr_test = (float *)dst_test.map_data();
        float *mapped_ptr_gold = (float *)dst_gold.map_data();
        strm.wait();
        for (int i = 0; i < M_group_sizes[g] * N; ++i) {
            float diff = std::abs(mapped_ptr_test[i] - mapped_ptr_gold[i]);
            ASSERT_NEAR(mapped_ptr_test[i], mapped_ptr_gold[i], 0.03f)
                    << "Mismatch at group" << g << " index " << i
                    << "test=" << mapped_ptr_test[i]
                    << ", gold=" << mapped_ptr_gold[i] << ", diff=" << diff;
        }
    }

    int iterations = 10;
    strm.wait();
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (int g = 0; g < num_groups; ++g) {
            unordered_map<int, memory> args2;
            args2.insert({DNNL_ARG_SRC, src_groups[g]});
            args2.insert({DNNL_ARG_WEIGHTS, weights_groups[g]});
            args2.insert({DNNL_ARG_DST, dst_groups[g]});
            prims[g].execute(strm, args2);
        }
    }
    strm.wait();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed
            = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Ref Grouped GEMM time: %.3f ms\n", elapsed / iterations);

    strm.wait();
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        matmul_prim.execute(strm, args);
    }
    strm.wait();
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    printf("DNNL Grouped GEMM time: %.3f ms\n", elapsed / iterations);
}
