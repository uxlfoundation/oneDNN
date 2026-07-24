/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_GENERIC_REF_SUM_MANY_INPUTS_HPP
#define GPU_GENERIC_REF_SUM_MANY_INPUTS_HPP

#include "common/primitive.hpp"
#include "common/sum.hpp"

#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

struct ref_sum_many_inputs_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_t("sycl:many_inputs:any", ref_sum_many_inputs_t);

        static constexpr int max_num_tensors = 8;

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper dst_d(dst_md());

            const int n = n_inputs();
            VDISPATCH_SUM_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SUM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            // prevent inf recursion
            VDISPATCH_SUM(n > max_num_tensors, VERBOSE_BAD_PARAM, "n_inputs");

            // the first kernel handles up to 8 inputs and remaining ones up to 7
            const int n_kernels
                    = n == 1 ? 1 : utils::div_up(n - 1, max_num_tensors - 1);
            base_pds_.resize(n_kernels);
            int in_arg_offset = 0;
            int n_remaining = n;
            for (auto i = 0; i < n_kernels; ++i) {
                bool pass_in_dst = i > 0;
                int max_n_child_inputs = max_num_tensors - pass_in_dst;
                int n_child_inputs = std::min(n_remaining, max_n_child_inputs);
                const memory_desc_t *src[max_num_tensors];
                if (pass_in_dst) { src[0] = dst_md(); }
                for (int j = 0; j < n_child_inputs; j++) {
                    src[j + pass_in_dst] = src_md(j + in_arg_offset);
                }
                in_arg_offset += n_child_inputs;
                n_remaining -= n_child_inputs;

                primitive_attr_t r_attr;
                CHECK(sum_primitive_desc_create(base_pds_[i], dst_md(),
                        n_child_inputs + pass_in_dst, scales(), src, &r_attr,
                        engine));
            }

            return status::success;
        }

        std::vector<std::shared_ptr<primitive_desc_t>> base_pds_;

    private:
        status_t init_conf();
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> base_prims_;
};

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
