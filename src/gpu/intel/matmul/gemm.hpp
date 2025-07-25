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

#ifndef GPU_INTEL_MATMUL_GEMM_HPP
#define GPU_INTEL_MATMUL_GEMM_HPP

#include "common/gemm_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct gemm_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T(gemm_pd_->name(), gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            primitive_attr_t gemm_attr;
            if (!attr()->scales_.has_default_values()) {
                gemm_attr.scales_ = attr()->scales_;
            }
            if (!attr()->dropout_.has_default_values()) {
                return status::unimplemented;
            }
            auto post_ops = attr()->post_ops_;
            auto a_md = src_md(), b_md = weights_md(), c_md = dst_md(),
                 bias_md = weights_md(1);
            const auto acc_dt = desc()->accum_data_type;
            memory_desc_t a_md_reshaped, b_md_reshaped, c_md_reshaped,
                    bia_md_reshaped;
            bool with_bia = bias_md->ndims > 0;
            auto orig_dims = a_md->ndims;

            const auto &scales = gemm_attr.scales_;
            const auto &zp = attr()->zero_points_;

            bool per_tensor_sc = false, per_tensor_zp = false;
            if (orig_dims > 2) {
                if (!scales.has_default_values()
                        && !scales.get(DNNL_ARG_SRC).has_default_groups()) {
                    per_tensor_sc = utils::one_of(
                            scales.get_mask(DNNL_ARG_SRC), full_tensor_mask());
                }
                if (!zp.has_default_values()
                        && !zp.get(DNNL_ARG_SRC).has_default_groups()) {
                    per_tensor_zp = utils::one_of(
                            zp.get_mask(DNNL_ARG_SRC), full_tensor_mask());
                }
            }

            auto map_gemm_zp = [&](int arg, bool reshape = false,
                                       int diff_dims = 0, dim_t g_dim = 0) {
                const auto &zp = attr()->zero_points_;
                if (zp.has_default_values(arg)) return status::success;

                int mask = zp.get_mask(arg);
                if (reshape) mask = mask >> diff_dims;
                data_type_t dt = zp.get_data_type(arg);
                int nd = 0;
                dims_t dims {};
                if (!zp.get(arg).has_default_groups()) {
                    if (per_tensor_zp && g_dim) mask = 3;
                    nd = 2; // Note: hardcoded so far.
                    dims[0] = ((arg == DNNL_ARG_SRC && g_dim && !per_tensor_zp)
                                    ? g_dim / attr()->scales_.get_group(arg, 0)
                                    : zp.get_group(arg, 0));
                    dims[1] = zp.get_group(arg, 1);
                }
                CHECK(gemm_attr.zero_points_.set(arg, mask, dt, nd, dims));
                return status::success;
            };

            // The function shrinks the mask for scales and updates it in
            // `scales` object.
            auto adjust_scales = [&](scales_t &scales, int arg, int diff_dims,
                                         dim_t g_dim = 0) {
                if (attr()->scales_.has_default_values(arg))
                    return status::success;

                int mask = attr()->scales_.get_mask(arg) >> diff_dims;
                data_type_t dt = attr()->scales_.get_data_type(arg);
                int nd = 0;
                dims_t dims {};
                if (!attr()->scales_.get(arg).has_default_groups()) {
                    if (per_tensor_sc && g_dim) mask = 3;
                    nd = 2; // Note: hardcoded so far.
                    dims[0] = ((arg == DNNL_ARG_SRC && g_dim && !per_tensor_sc)
                                    ? g_dim / attr()->scales_.get_group(arg, 0)
                                    : attr()->scales_.get_group(arg, 0));
                    dims[1] = attr()->scales_.get_group(arg, 1);
                }
                CHECK(scales.set(arg, mask, dt, nd, dims));
                return status::success;
            };
            if (!attr()->zero_points_.has_default_values()) {
                CHECK(map_gemm_zp(DNNL_ARG_SRC, false, orig_dims - 2));
                CHECK(map_gemm_zp(DNNL_ARG_WEIGHTS, false, orig_dims - 2));
                CHECK(map_gemm_zp(DNNL_ARG_DST));
            }

            auto maybe_reshape
                    = [&](dims_t &orig_a_dims, dims_t &orig_b_dims,
                              dims_t &orig_c_dims, dims_t &orig_bias_dims,
                              const int orig_dims) -> status_t {
                int batch_b_dims = 1;
                for (int i = 0; i < b_md->ndims - 2; i++) {
                    batch_b_dims *= b_md->dims[i];
                }
                for (int i = 0; i < orig_dims; i++) {
                    orig_a_dims[i] = a_md->dims[i];
                    orig_b_dims[i] = b_md->dims[i];
                    orig_c_dims[i] = c_md->dims[i];
                    orig_bias_dims[i] = bias_md->dims[i];
                }
                // for batch dim can map broadcast to 2d: eg. 4x1x4096:1x4096x16 -> 4x4096:4096x16
                bool reshape_2d = (batch_b_dims == 1 && b_md->ndims > 2);
                bool reshape_3d = (a_md->ndims > 3);
                bool allow_reshape
                        = gpu_utils::dev_getenv("GEMM_ALLOW_RESHAPE", true);

                // Early exit if not reshaping
                if (!allow_reshape || !(reshape_2d || reshape_3d))
                    return status::unimplemented;

                // memory_desc_reshape does not support strided matrices. In these cases, we want
                // to gracefully exit without reshaping
                auto md_ok = [](const memory_desc_wrapper &mdw) -> bool {
                    if (mdw.format_any()) return true;
                    if (mdw.is_dense()) return true;
                    return false;
                };
                bool ok = true;
                ok = ok && md_ok(a_md);
                ok = ok && md_ok(b_md);
                ok = ok && md_ok(c_md);

                int ndims = a_md->ndims;
                int reshape_size = reshape_2d ? 2 : 3;

                // Converts input dims_t to output dims_t after reshaping
                auto squash_dims = [](dims_t &out_dims, const dims_t &in_dims,
                                           int in_ndims, int out_ndims) {
                    int diff_ndims = in_ndims - out_ndims;
                    gpu_assert(diff_ndims > 0) << "Unexpected squashing dims";
                    gpu_assert(&out_dims != &in_dims)
                            << "Cannot squash in-place";
                    // Shift dims over
                    for (int i = 0; i < out_ndims; i++) {
                        out_dims[i] = in_dims[i + diff_ndims];
                    }
                    // Squash dims into 1st dim
                    for (int i = 0; i < diff_ndims; i++) {
                        out_dims[0] *= in_dims[i];
                    }
                };

                // Convert raw tensors to reshaped tensors
                dims_t a_dims, b_dims, c_dims, bia_dims;
                squash_dims(a_dims, a_md->dims, ndims, reshape_size);
                squash_dims(b_dims, b_md->dims, ndims, reshape_size);
                squash_dims(c_dims, c_md->dims, ndims, reshape_size);
                squash_dims(bia_dims, bias_md->dims, ndims, reshape_size);

                // Cannot reshape if bias is broadcast across a subset of squashed dimensions
                ok = ok
                        && IMPLICATION(with_bia,
                                utils::one_of(bia_dims[0], 1, c_dims[0]));

                // 3D reshaping is only possible if A and B batch sizes allow.
                // This means no reshaping with partial broadcasting
                // or when both tensors are broadcast in different dimensions.
                bool a_broadcast = false;
                bool b_broadcast = false;
                for (int i = 0; i < ndims - 2; i++) {
                    if (a_md->dims[i] == 1 && b_md->dims[i] > 1)
                        a_broadcast = true;
                    if (b_md->dims[i] == 1 && a_md->dims[i] > 1)
                        b_broadcast = true;
                }
                ok = ok && !(a_broadcast && b_broadcast);
                ok = ok
                        && IMPLICATION(reshape_size == 3,
                                a_dims[0] == b_dims[0]
                                        || utils::one_of(
                                                1, a_dims[0], b_dims[0]));
                if (!ok) return status::unimplemented;

                CHECK(memory_desc_reshape(
                        a_md_reshaped, *a_md, reshape_size, a_dims));
                CHECK(memory_desc_reshape(
                        b_md_reshaped, *b_md, reshape_size, b_dims));
                CHECK(memory_desc_reshape(
                        c_md_reshaped, *c_md, reshape_size, c_dims));
                if (with_bia) {
                    CHECK(memory_desc_reshape(
                            bia_md_reshaped, *bias_md, reshape_size, bia_dims));
                }
                auto tmp_post_ops = post_ops;
                for (int i = 0; i < attr()->post_ops_.len(); i++) {
                    auto &po = post_ops.entry_[i];
                    if (po.is_binary()) {
                        const auto &po_desc = po.binary.src1_desc;
                        auto a_dim = po_desc.dims[po_desc.ndims - reshape_size];
                        for (int i = po_desc.ndims; i > reshape_size; i--) {
                            a_dim *= po_desc.dims[po_desc.ndims - i];
                        }
                        //post ops cannot be applied if applied on only on a subset of batch dims
                        if (a_dim != c_dims[0] && a_dim > 1) {
                            return status::unimplemented;
                        }
                        auto has_dims = po_desc.ndims > 0;
                        dims_t po_dims;
                        if (reshape_2d) {
                            po_dims[0] = a_dim;
                            po_dims[1] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 1]
                                    : 1;
                        } else {
                            po_dims[0] = a_dim;
                            po_dims[1] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 2]
                                    : 1;
                            po_dims[2] = has_dims
                                    ? po_desc.dims[po_desc.ndims - 1]
                                    : 1;
                        }
                        memory_desc_t tmp_po_desc;
                        CHECK(memory_desc_reshape(
                                tmp_po_desc, po_desc, reshape_size, po_dims));
                        tmp_post_ops.entry_[i].binary.src1_desc = tmp_po_desc;
                    } else if (po.is_prelu()) {
                        auto mask = po.prelu.mask;
                        int new_mask = 0;
                        int batch_idx = reshape_size - 1;
                        int batch_dim = 1;
                        int mask_dim = 1;
                        //get mask for batch dim
                        for (int i = 0; i < c_md->ndims - batch_idx; i++) {
                            if (mask >> i & 1) {
                                //post ops cannot be applied if applied on only on a subset of batch dims
                                if (new_mask != 0) return status::unimplemented;
                                new_mask |= c_md->dims[i] == 1 ? 0 : 1;
                                mask_dim *= c_md->dims[i];
                            }
                            batch_dim *= c_md->dims[i];
                        }
                        //post ops cannot be applied if applied on only on a subset of batch dims
                        if (batch_dim != mask_dim) return status::unimplemented;
                        //get non-batch part of mask
                        auto shift = c_md->ndims - batch_idx;
                        auto non_batch_mask = mask >> shift;
                        //due to prelu being in axb format, if a reshape is done it
                        //implies layout is different e.g 1x30x20 -> 30 is innermost dimension
                        //but 30x20 -> 20 is innermost. Hence reshape does  not work if mask
                        //is applied across more than one dimension.
                        if (non_batch_mask > 2
                                || (non_batch_mask > 0 && new_mask > 0))
                            return status::unimplemented;
                        new_mask |= non_batch_mask << 1;
                        tmp_post_ops.entry_[i].prelu.mask = new_mask;
                    }
                }
                dim_t a_dim_ratio = a_dims[0] / a_md->dims[orig_dims - 2];
                auto new_scales = gemm_attr.scales_;
                if (!attr()->scales_.has_default_values()) {
                    CHECK(adjust_scales(new_scales, DNNL_ARG_A,
                            orig_dims - reshape_size, 0));
                    CHECK(adjust_scales(new_scales, DNNL_ARG_B,
                            orig_dims - reshape_size, a_dim_ratio));
                    CHECK(adjust_scales(new_scales, DNNL_ARG_C,
                            orig_dims - reshape_size, 0));
                }
                if (!attr()->zero_points_.has_default_values()) {
                    CHECK(map_gemm_zp(
                            DNNL_ARG_WEIGHTS, true, orig_dims - reshape_size));
                    CHECK(map_gemm_zp(DNNL_ARG_SRC, true,
                            orig_dims - reshape_size, a_dim_ratio));
                }
                post_ops = tmp_post_ops;
                gemm_attr.scales_ = std::move(new_scales);
                a_md = &a_md_reshaped;
                b_md = &b_md_reshaped;
                c_md = &c_md_reshaped;
                if (with_bia) bias_md = &bia_md_reshaped;
                return status::success;
            };

            CHECK(gemm_attr.set_fpmath_mode(
                    attr()->fpmath_.mode_, attr()->fpmath_.apply_to_int_));
            CHECK(gemm_attr.set_accumulation_mode(attr()->acc_mode_));
            gemm_attr.deterministic_ = attr()->deterministic_;

            dims_t orig_a_dims, orig_b_dims, orig_c_dims, orig_bias_dims;
            bool reshape = maybe_reshape(orig_a_dims, orig_b_dims, orig_c_dims,
                                   orig_bias_dims, orig_dims)
                    == status::success;

            if (!attr()->post_ops_.has_default_values()) {
                gemm_attr.post_ops_ = post_ops;
            }
            if (!attr()->rounding_mode_.has_default_values()) {
                gemm_attr.rounding_mode_ = attr()->rounding_mode_;
            }

            // We create a gemm_pd and resolve 'any' desc by querying gemm_pd
            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL_SC(create_gemm_pd(gemm_pd_, engine, a_md, b_md,
                                        c_md, bias_md, acc_dt, &gemm_attr),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "gemm");
            VDISPATCH_MATMUL_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            if (reshape) {
                CHECK(memory_desc_reshape(
                        src_md_, src_md_, orig_dims, orig_a_dims));
                CHECK(memory_desc_reshape(
                        weights_md_, weights_md_, orig_dims, orig_b_dims));
                CHECK(memory_desc_reshape(
                        dst_md_, dst_md_, orig_dims, orig_c_dims));
                if (with_bia)
                    CHECK(memory_desc_reshape(
                            bias_md_, bias_md_, orig_dims, orig_bias_dims));
            }
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;

    private:
        status_t set_default_params() {
            src_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_0);
            weights_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_1);
            bias_md_ = *gemm_pd_->arg_md(DNNL_ARG_BIAS);
            dst_md_ = *gemm_pd_->arg_md(DNNL_ARG_DST);
            return status::success;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(gemm_, pd()->gemm_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> gemm_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
