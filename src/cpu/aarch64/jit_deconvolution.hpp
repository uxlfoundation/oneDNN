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

#ifndef CPU_AARCH64_JIT_DECONVOLUTION_HPP
#define CPU_AARCH64_JIT_DECONVOLUTION_HPP

#include <memory>
#include <string>

#include "common/primitive.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/cpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

/*
 * Forward-deconvolution implementation which executes an equivalent nested
 * forward-convolution primitive.
 *
 * The implementation currently supports:
 *
 *   - forward training and forward inference;
 *   - deconvolution_direct;
 *   - four-dimensional tensors;
 *   - supported f32, bf16, and f16 datatype combinations;
 *   - ungrouped deconvolution;
 *   - NHWC source and destination;
 *   - HWIO weights;
 *   - unit spatial strides;
 *   - supported padding and dilation;
 *   - attributes supported by the selected nested convolution.
 *
 * The nested convolution is selected by primitive_desc_iterator_t from the
 * normal CPU convolution implementation list. The first implementation whose
 * memory descriptors exactly match the outer deconvolution descriptors is
 * used.
 *
 * For non-1x1 kernels, the implementation creates a spatially inverted copy
 * of the weights before executing the nested convolution.
 */
struct jit_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                name_.c_str(),
                jit_deconvolution_fwd_t);

        status_t init(const engine_t *engine);

    private:
        /*
         * Builds the outer implementation name from the selected nested
         * convolution implementation.
         */
        void init_name();

        /*
         * Assigns the explicit NHWC/HWIO/NHWC formats required by the current
         * direct argument-forwarding and weight-inversion paths when
         * format_kind::any was supplied.
         */
        status_t set_default_formats();

        /*
         * Primitive descriptor for the selected nested forward-convolution
         * implementation.
         */
        std::shared_ptr<primitive_desc_t> conv_pd_;

        std::string name_;

        friend struct jit_deconvolution_fwd_t;
    };

    explicit jit_deconvolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    /*
     * Returns the derived primitive descriptor rather than the
     * primitive_desc_t base type returned by primitive_t::pd().
     */
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    /*
     * Nested forward-convolution primitive created from conv_pd_.
     */
    std::shared_ptr<primitive_t> conv_p_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif