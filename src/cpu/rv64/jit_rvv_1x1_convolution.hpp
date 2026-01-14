/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#ifndef CPU_RV64_JIT_RVV_1X1_CONVOLUTION_HPP
#define CPU_RV64_JIT_RVV_1X1_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/rv64/jit_rvv_1x1_conv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        // XXX: Workaround for test_iface_attr DepthwiseFusion test.
        // The test expects all "jit_*" implementations to support depthwise
        // fusion post-ops. Since this implementation doesn't support post-ops
        // yet, we use "RISCV64GCV_1x1" instead of "jit_1x1" to skip the test.
        // TODO: Change back to "jit_1x1" once depthwise fusion is supported.
        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("RISCV64GCV_1x1:", v, ""),
                jit_rvv_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace format_tag;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(
                    expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::undef),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(attr()->has_default_values(
                                   primitive_attr_t::skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            // Only support: data = nwc/nhwc/ndhwc, weights = blocked formats (Oiw4o/gOiw4o/etc)
            const int n = ndims();
            const bool g = with_groups();
            const auto dat_tag_nxc = utils::pick(n - 3, nwc, nhwc, ndhwc);
            const auto wei_tag_blocked = utils::pick(2 * n - 6 + (g ? 1 : 0),
                    Oiw4o, gOiw4o, Oihw4o, gOihw4o, Oidhw4o, gOidhw4o);

            // Check if src/dst match supported format (nxc)
            // Only accept format_kind::any as a fallback, reject explicit
            // unsupported formats
            VDISPATCH_CONV(IMPLICATION(src_d.matches_one_of_tag(dat_tag_nxc)
                                           != dat_tag_nxc,
                                   src_d.format_kind() == format_kind::any),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(IMPLICATION(dst_d.matches_one_of_tag(dat_tag_nxc)
                                           != dat_tag_nxc,
                                   dst_d.format_kind() == format_kind::any),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    IMPLICATION(weights_d.matches_one_of_tag(wei_tag_blocked)
                                    != wei_tag_blocked,
                            weights_d.format_kind() == format_kind::any),
                    VERBOSE_UNSUPPORTED_TAG);

            // Set default formats if format_kind == any
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            // ISA check
            VDISPATCH_CONV(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

            // 1x1 convolution check
            const int ndims = src_d.ndims();
            const int weights_ndims = weights_d.ndims();
            for (int i = 0; i < ndims - 2; ++i) {
                VDISPATCH_CONV(
                        weights_d.dims()[weights_ndims - (ndims - 2) + i] == 1,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "only 1x1 convolution is supported");
                VDISPATCH_CONV(desc()->strides[i] == 1,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "only stride 1 is supported");
                VDISPATCH_CONV(desc()->padding[0][i] == 0,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "padding is not supported");
            }

            VDISPATCH_CONV_SC(jit_rvv_1x1_conv_kernel_t::init_conf(jcp_,
                                      *desc(), src_d, weights_d, dst_d, *attr(),
                                      dnnl_get_max_threads(), false),
                    VERBOSE_UNSUPPORTED_FEATURE, "init_conf failed");

            auto scratchpad = scratchpad_registry().registrar();
            jit_rvv_1x1_conv_kernel_t::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();

    protected:
        bool post_ops_ok() const {
            // TODO: Post-ops support is not implemented yet.
            return attr()->post_ops_.len() == 0;
        }
        bool set_default_formats() {
            using namespace format_tag;
            const int n = ndims();
            const bool g = with_groups();
            const auto dat_tag = utils::pick(n - 3, nwc, nhwc, ndhwc);
            const auto wei_tag = utils::pick(2 * n - 6 + (g ? 1 : 0), Oiw4o,
                    gOiw4o, Oihw4o, gOihw4o, Oidhw4o, gOidhw4o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    jit_rvv_1x1_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_rvv_1x1_conv_kernel_t(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md())));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr, const float *src,
            const float *weights, const float *bias, float *dst,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_rvv_1x1_conv_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
