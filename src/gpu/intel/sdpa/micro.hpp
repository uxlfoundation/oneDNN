/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_SDPA_MICRO_HPP
#define GPU_INTEL_SDPA_MICRO_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/gemm_utils.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"
#include "common/utils.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/sdpa/config.hpp"
#include "gpu/intel/sdpa/configs.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

struct micro_params_t : trivially_serializable_t<micro_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names = {"micro_sdpa"};
        return kernel_names;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;

    int ndims;
    data_type_t data_t;
    data_type_t dst_data_t, key_data_t, qry_data_t, val_data_t, msk_data_t;
    data_type_t key_scales_data_t, value_scales_data_t;
    data_type_t key_zp_data_t, value_zp_data_t;
    int kv_group_size;

    int q_align, k_align, v_align, a_align;
    bool transpose_k;
    uint8_t padding0[3] = {0};

    int kq_scale_mask, vs_scale_mask, kq_zp_mask, vs_zp_mask;
    int key_elements_per_byte, key_zp_elements_per_byte, val_elements_per_byte,
            val_zp_elements_per_byte;

    int key_group_size, val_group_size;
    data_type_t scale_data_t;

    int attn_mask_undef, attn_mask_buffer, attn_mask_top_left,
            attn_mask_bottom_right;
    bool invert_scale, with_attn_scale, with_attn_mask, broadcast_mask_q,
            with_causal_mask;
    uint8_t padding1[3] = {0};
    int subgroup_size, d_max;

    bool d_full, arch_gte_hpc;
    bool block_q, block_a, block_msk, block_2d_a;
    bool prefetch_mask, prefetch_k0, prefetch_k, prefetch_v, prefetch_remainder;
    uint8_t padding2[5] = {0};
    int prefetch_d_max;

    bool softmax_inf_as_zero;
    bool q_arrive_await_barrier;
    bool use_systolic_ukernel;
    uint8_t padding3[1] = {0};

    micro_ukernel_params_t ukernel_config;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_params_t);

struct micro_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public sdpa::pd_t {
        using sdpa::pd_t::pd_t;
        static constexpr int mask_mb_index = 0;
        static constexpr int mask_q_index = 2;
        static constexpr int mask_k_index = 3;
        static constexpr int ndims = 4;

        DECLARE_COMMON_PD_T("ocl:micro:reusable", micro_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            VCHECK_SDPA_COND(
                    utils::everyone_is(4, qry_md()->ndims, key_md()->ndims,
                            val_md()->ndims, dst_md()->ndims),
                    VERBOSE_UNSUPPORTED_TAG);

            memory_desc_wrapper qry_mdw(qry_md());
            memory_desc_wrapper key_mdw(key_md());
            memory_desc_wrapper val_mdw(val_md());
            memory_desc_wrapper dst_mdw(dst_md());
            VCHECK_SDPA_COND(utils::everyone_is(true, qry_mdw.is_plain(),
                                     key_mdw.is_plain(), val_mdw.is_plain(),
                                     dst_mdw.is_plain()),
                    VERBOSE_UNSUPPORTED_TAG);

            if (with_attn_mask()) {
                VCHECK_SDPA_COND(
                        attn_mask_md()->ndims == 4, VERBOSE_UNSUPPORTED_TAG);
                VCHECK_SDPA_COND(
                        utils::one_of(attn_mask_md()->dims[mask_q_index],
                                desc()->queries(), 1),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
                VCHECK_SDPA_COND(
                        attn_mask_md()->dims[mask_k_index] == desc()->keys(),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
                if (qry_md()->data_type == data_type::f32) {
                    VCHECK_SDPA_COND(
                            attn_mask_md()->data_type == qry_md()->data_type,
                            "Mask data type(%s) should match Qry/Dst data "
                            "type(%s).",
                            dnnl_dt2str(attn_mask_md()->data_type),
                            dnnl_dt2str(qry_md()->data_type));
                } else {
                    VCHECK_SDPA_COND(
                            (attn_mask_md()->data_type == qry_md()->data_type)
                                    || (attn_mask_md()->data_type
                                            == data_type::f32),
                            "Mask data type(%s) should be xf16 or f32 when "
                            "Qry/Dst(%s) is xf16.",
                            dnnl_dt2str(attn_mask_md()->data_type),
                            dnnl_dt2str(qry_md()->data_type));
                }
            }
            VCHECK_SDPA_COND(
                    (utils::everyone_is(data_type::f16, qry_md()->data_type,
                             dst_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    qry_md()->data_type, dst_md()->data_type)
                            || utils::everyone_is(data_type::f32,
                                    qry_md()->data_type, dst_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VCHECK_SDPA_COND(utils::one_of(key_md()->data_type, f32, bf16, f16,
                                     u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VCHECK_SDPA_COND(utils::one_of(val_md()->data_type, f32, bf16, f16,
                                     u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VCHECK_SDPA_COND(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_SDPA_COND(desc()->values() == desc()->head_size(),
                    "values does not match head size");

            if (utils::one_of(key_md()->data_type, u4, s4)) {
                VCHECK_SDPA_COND(desc()->keys() % 2 == 0,
                        "The number of keys must be an even size with the data "
                        "type is u4 or s4.");
            }

            if (utils::one_of(val_md()->data_type, u4, s4)) {
                VCHECK_SDPA_COND(desc()->values() % 2 == 0,
                        "The number of values must be an even size with the "
                        "data type is u4 or s4.");
            }

            VCHECK_SDPA_COND(qry_md()->dims[1] >= key_md()->dims[1]
                            && qry_md()->dims[1] >= val_md()->dims[1],
                    "number of heads in query tensor(%ld) must be greater "
                    "than the number of heads in the key(%ld) and value(%ld) "
                    "tensors",
                    static_cast<long int>(qry_md()->dims[1]),
                    static_cast<long int>(key_md()->dims[1]),
                    static_cast<long int>(val_md()->dims[1]));

            int kq_scales_mask = desc()->kq_scales.get_mask();
            int kq_zp_mask = desc()->kq_zero_points.get_mask();
            if (!desc()->kq_scales.has_default_values()
                    && !desc()->kq_zero_points.has_default_values())
                VCHECK_SDPA_COND(kq_scales_mask == kq_zp_mask,
                        "kq scales mask(%d) must equal kq zero point(%d) "
                        "mask",
                        kq_scales_mask, kq_zp_mask);
            if (!desc()->kq_scales.has_default_values())
                VCHECK_SDPA_COND(utils::one_of(kq_scales_mask, 0, 1, 3, 11, 15),
                        "unsupported mask for kq matmul(%d). must be 0, 1, 3, "
                        "11, or 15",
                        kq_scales_mask);
            if (!desc()->kq_zero_points.has_default_values())
                VCHECK_SDPA_COND(utils::one_of(kq_zp_mask, 0, 1, 3, 11, 15),
                        "unsupported mask for kq matmul(%d). must be 0, 1, 3, "
                        "11, or 15",
                        kq_zp_mask);

            int vs_scales_mask = desc()->vs_scales.get_mask();
            int vs_zp_mask = desc()->vs_zero_points.get_mask();
            if (!desc()->vs_scales.has_default_values()
                    && !desc()->vs_zero_points.has_default_values())
                VCHECK_SDPA_COND(vs_scales_mask == vs_zp_mask,
                        "vs scales mask(%d) must equal vs zero point(%d) "
                        "mask",
                        vs_scales_mask, vs_zp_mask);
            if (!desc()->vs_scales.has_default_values())
                VCHECK_SDPA_COND(utils::one_of(vs_scales_mask, 0, 1, 3, 7, 15),
                        "unsupported mask for vs matmul(%d). must be 0, 1, 3, "
                        "7, or 15",
                        vs_scales_mask);
            if (!desc()->vs_zero_points.has_default_values())
                VCHECK_SDPA_COND(utils::one_of(vs_zp_mask, 0, 1, 3, 7, 15),
                        "unsupported mask for vs matmul(%d). must be 0, 1, 3, "
                        "7, or 15",
                        vs_zp_mask);

            /// NOTE: Limitation of microkernels
            if (utils::one_of(desc()->vs_zero_points.get_data_type(), s4, u4)) {
                VCHECK_SDPA_COND(value_group_size() == 16,
                        "if vs zero points data type is s4 or u4 then the "
                        "group size(%d) must be 16.",
                        value_group_size());
            }

            if (!desc()->vs_scales.has_default_values()
                    || !desc()->vs_zero_points.has_default_values()) {
                int vgs = value_group_size();
                VCHECK_SDPA_COND(utils::one_of(vs_scales_mask, 0, 1, 3)
                                || (math::is_pow2<int>(vgs)
                                        || vgs == val_md()->dims[3]),
                        "the value group size(%d) must be a power of 2 or "
                        "equal to the number of values(%ld).",
                        vgs, static_cast<long int>(val_md()->dims[3]));
            }


            // ??? TODO ???

            CHECK(init_conf_microkernels(engine));
            CHECK(init_conf(engine));

            return status::success;
        }

        status_t set_default_format(memory_desc_t &md, bool allow_transpose) {
            using namespace format_tag;
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) return status::unimplemented;
            if (!is_md_gemm_compatible_plain_format(&md))
                return status::unimplemented;
            if (gemm_desc_t::get_trans(md) == dnnl_trans && !allow_transpose)
                return status::unimplemented;
            return status::success;
        }

        status_t set_default_formats() {
            CHECK(set_default_format(desc_.q_desc, false));
            CHECK(set_default_format(desc_.k_desc, true));
            CHECK(set_default_format(desc_.v_desc, false));
            CHECK(set_default_format(desc_.dst_desc, false));
            return status::success;
        }

        int sg_size() const { return sg_size_; }
        bool use_systolic_ukernel() const { return use_systolic_ukernel_; }

        // Block size for head_size, which must be hard-coded into the kernel.
        int d_max() const {
            int head_size = into<int>(desc()->head_size());
            for (int i = 32; i <= 1024; i *= 2)
                if (head_size <= i) return i;
            return head_size;
        }

        compute::gpu_arch_t arch() const { return arch_; }
        micro_params_t conf;

    private:
        int sg_size_ = 0;
        bool use_systolic_ukernel_ = true;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_conf_microkernels(impl::engine_t *engine);
        status_t init_conf(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute(const exec_ctx_t &ctx) const override;

    compute::kernel_t kernel_;
};

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
