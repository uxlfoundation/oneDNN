/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_REORDER_ZEN_REORDER_HPP
#define CPU_X64_ZEN64_REORDER_ZEN_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace reorder {

// ---------------------------------------------------------------------------
// Zen weight pre-pack reorder
// ---------------------------------------------------------------------------
// Dispatched only when the destination memory_desc uses the dedicated opaque
// `format_kind::zen_packed` format, set by the Zen matmul pd_t::init()
// when the weights layout is left open (format_any). See
// src/cpu/x64/zen64/common/zen_format_tag.hpp.
//
// The destination is an opaque buffer sized to exactly the Zen packer's
// output (zen_packed_desc.size); the bytes are produced via the Zen
// backend `reorder_direct` (prepack mode) instead of oneDNN's standard
// reorder. The matmul backend then treats the buffer as `mem_format_b='r'`
// so the backend skips its own packing pass.
struct zen_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("zen:reorder", zen_reorder_t);

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
    };

    zen_reorder_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace reorder
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
