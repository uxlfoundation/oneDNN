/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022, 2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_REORDER_JIT_UNI_REORDER_UTILS_HPP
#define CPU_AARCH64_REORDER_JIT_UNI_REORDER_UTILS_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {
struct node_t {
    static constexpr int64_t empty_field = -1;

    size_t n = 0;
    size_t tail_size = 0;
    int dim_id = empty_field;
    int parent_node_id = empty_field;
    bool is_zero_pad_needed = false;
    ptrdiff_t is = 0; // input stride
    ptrdiff_t os = 0; // output stride
    ptrdiff_t ss = 0; // scale stride
    ptrdiff_t cs = 0; // compensation stride

    bool is_dim_id_empty() const { return dim_id == empty_field; }
    bool is_parent_empty() const { return parent_node_id == empty_field; }
};

enum class scale_type_t { NONE, COMMON, MANY };

struct prb_t {
    /* The compensation mask value indicates how big an additional buffer should be.
     * Possible values for reorder:
     *     1) standard compensation = 1 = 0b01
     *     2) asymmetric compensation = 2 = 0b10
     *     3) compensation if tensor contains group = 3 = 0b11 */
    static constexpr int invalid_comp_mask = 0;
    static constexpr int standard_comp_mask = 0b1;
    static constexpr int asymmetric_comp_mask = 0b10;
    static constexpr int comp_mask_with_groups
            = standard_comp_mask + asymmetric_comp_mask;

    bool is_tail_in_one_of_child_nodes(int parent_node_id) const {
        for (int i = parent_node_id; i >= 0; i--) {
            if (nodes[i].parent_node_id == parent_node_id) {
                if (nodes[i].tail_size != 0)
                    return true;
                else
                    parent_node_id = i;
            }
        }

        return false;
    }

    int tail(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].tail_size);
    }

    int n(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].n);
    }
    int is(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].is);
    }
    int os(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].os);
    }
    int ss(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].ss);
    }

    int cs(int d) const {
        assert(d < ndims);
        return static_cast<int>(nodes[d].cs);
    }

    data_type_t itype;
    data_type_t otype;
    int ndims;
    node_t nodes[DNNL_MAX_NDIMS];
    ptrdiff_t ioff;
    ptrdiff_t ooff;
    scale_type_t src_scale_type;
    scale_type_t dst_scale_type;
    float beta;
    int full_ndims;
    bool is_tail_present = false;
    float scale_adjust = 1.f;
    int compensation_mask = invalid_comp_mask;
    bool req_s8s8_comp = false;
    bool req_asymmetric_comp = false;
    bool req_src_zp = false;
    bool req_dst_zp = false;
};

bool prb_has_small_strides(const prb_t &prb);

status_t prb_init(prb_t &prb, const memory_desc_t &imd,
        const memory_desc_t &omd, const primitive_attr_t *attr);

/** sorts the problem nodes so that output strides come in ascending order */
void prb_normalize(prb_t &p);

/** fill parent node info for blocked nodes */
void prb_node_dependency(prb_t &p);

/** folds nodes together if possible */
void prb_simplify(prb_t &p);

/** splits the node dim into two of sizes n1 and n / n1
 * @warning n must be multiple of n1 */
void prb_node_split(prb_t &p, int dim, size_t n1);

/** swaps d0 and d1 nodes */
void prb_node_swap(prb_t &p, int d0, int d1);

/** moves node d0 to the d1 position.
 * nodes (d0, d1] are shifted to the left if d0 < d1 or
 * to the right if d0 > d1 */
void prb_node_move(prb_t &p, int d0, int d1);

bool prb_has_small_strides(const prb_t &prb);

/** dumps the problem to a string */
std::string prb_dump(const prb_t &p);

void prb_block_for_cache(prb_t &prb);

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
void prb_thread_kernel_balance(prb_t &prb, int &ndims_ker_max, int nthr);

} // namespace tr

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
