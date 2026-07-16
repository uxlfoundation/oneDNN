/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_REORDER_JIT_BLK_REORDER_HPP
#define CPU_RV64_REORDER_JIT_BLK_REORDER_HPP

#include <cassert>
#include <memory>

#include "common/c_types_map.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/reorder/jit_uni_reorder_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace tr {

// Plain/channel-blocked f32 reorder kernel. 4c/8c use RVV segment
// load/store; 16c/32c use an RVV register transpose over the inner slice.
struct jit_single_blk_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)

    static bool applicable(const prb_t &p);

    jit_single_blk_kernel_t(const prb_t &prb);

    void operator()(
            const void *in, void *out, dim_t cols, dim_t real_block) const {
        jit_generator_t::operator()(in, out, cols, real_block);
    }

    void generate() override;

private:
    enum class kernel_kind_t { segment_4c8c, transpose_16c32c };

    static kernel_kind_t select_kernel_kind(const prb_t &prb);

    void emit_segment_kernel();
    void emit_transpose_kernel();

    const prb_t prb_;
    int itype_sz_;
    int otype_sz_;
    int block_sz_;
    int tile_cols_;
    bool plain_to_blocked_;
    kernel_kind_t kernel_kind_;
};

} // namespace tr

struct jit_blk_reorder_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("jit:blk", jit_blk_reorder_t);

        tr::prb_t prb_;

    private:
        static status_t create(reorder_pd_t **reorder_pd,
                const engine_t *engine, const primitive_attr_t *attr,
                const engine_t *src_engine, const memory_desc_t *src_md,
                const engine_t *dst_engine, const memory_desc_t *dst_md);

        // Put the 4/8/16/32 block node first.
        static void prb_tile_normalize(tr::prb_t &p);
        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    jit_blk_reorder_t(const pd_t *apd);
    ~jit_blk_reorder_t() override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::jit_single_blk_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
