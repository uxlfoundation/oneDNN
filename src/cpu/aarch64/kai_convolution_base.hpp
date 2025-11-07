/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_CONVOLUTION_BASE_HPP
#define CPU_AARCH64_KAI_CONVOLUTION_BASE_HPP

#include <cstddef>
#include <memory>
#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/aarch64/kai_utils.hpp"
#include "cpu/aarch64/post_ops_fallback.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace memory_tracking {
struct grantor_t;
struct registrar_t;
} // namespace memory_tracking
namespace cpu {
namespace aarch64 {

struct kai_convolution_fwd_base_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , name_("unknown:kai") {}

        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;
        const kai::ops::GemmArgs &gemm_args() const { return *args_; }

        static bool is_1x1(const cpu_convolution_fwd_pd_t &pd);
        static bool has_zero_top_left_padding(
                const cpu_convolution_fwd_pd_t &pd);
        // Assumes a 1x1 kernel and zero top and left padding. Checks that
        // the output is nonempty and the last required source pixel is
        // within the source image.
        static bool source_pixels_in_bounds(const cpu_convolution_fwd_pd_t &pd);
        data_type_t gemm_dst_dt() const;

        data_type_t gemm_src_dt_ = data_type::undef;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;
        post_ops_fallback_t post_ops;

    protected:
        // These blocks never call back into the derived descriptor. Each
        // implementation owns their order and its own source preparation.
        status_t init_common(const engine_t *engine);
        void init_gemm_args(unsigned int m, unsigned int k,
                unsigned int k_sections, unsigned int batches,
                unsigned int multi, bool indirect);
        status_t init_gemm(const engine_t *engine);

        std::string name_;

    private:
        std::shared_ptr<kai::ops::GemmArgs> args_;
    };

    kai_convolution_fwd_base_t(const pd_t *apd) : primitive_t(apd) {}

protected:
    status_t init_common(engine_t *engine);
    static status_t execute_reorder(const exec_ctx_t &ctx,
            const primitive_t &reorder, const memory_desc_t *src_md,
            const memory_desc_t *dst_md, const void *src, void *dst,
            memory_tracking::key_t nested_key);

    // We need this struct because multiple impl member functions use and mutate these
    // variables. These args cannot be primitive member variables because they are
    // specific to the execute function scope (primitives can be used concurrently)
    struct kernel_call_args_t {
        kernel_call_args_t(const exec_ctx_t &ctx, const pd_t &pd,
                kai::ops::IGemmCommon &kernel)
            : ctx(ctx), pd(pd), kernel(kernel) {}

        const exec_ctx_t &ctx;
        const pd_t &pd;
        kai::ops::IGemmCommon &kernel;
        kai_utils::thread_partition_t thread_partition {};
        const char *src_base = nullptr;
        void *wei_base = nullptr;
        void *kernel_dst_base = nullptr;
        const void *bias_base = nullptr;

        // set_arrays_generic argument values shared by every convolution method.
        // All strides are in elements
        const void *B() const { return wei_base; }
        int ldb() const {
            return static_cast<int>(pd.gemm_args()._fixed_format
                            ? pd.weights_md()->format_desc.blocking.strides[0]
                            : pd.OC());
        }
        int B_multi_stride() const { return 0; }
        void *C() const { return kernel_dst_base; }
        int ldc() const { return static_cast<int>(pd.OC()); }
        const void *bias() const { return bias_base; }
        int bias_multi_stride() const { return 0; }
    };

    status_t prepare(kernel_call_args_t &args) const;
    status_t execute_kernel(const kernel_call_args_t &args) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

private:
    std::shared_ptr<impl::primitive_t> dst_reorder_;
    post_ops_fallback_t post_ops_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
