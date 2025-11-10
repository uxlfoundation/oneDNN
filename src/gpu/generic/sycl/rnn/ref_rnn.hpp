/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_RNN_REF_RNN_HPP
#define GPU_GENERIC_SYCL_RNN_REF_RNN_HPP

#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/rnn/rnn_kernels.hpp"
#include "gpu/generic/sycl/rnn/rnn_utils.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/gpu_rnn_pd.hpp"

#include "gpu/generic/sycl/sycl_gpu_kernel.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

enum matmul_kind_t {
    matmul_iter_fwd,
    matmul_layer_fwd,
    matmul_iter_bwd,
    matmul_layer_bwd,
    matmul_diff_wei_iter,
    matmul_diff_wei_layer
};

struct cell_ctx_t {
    impl::engine_t *engine;
    const exec_ctx_t &ctx;
    dim_t dir;
    dim_t lay;
    dim_t iter;
    const rnn_utils::user_data_t &user_data;
    const rnn_utils::workspace_t &workspace;
    const rnn_utils::scratch_t &scratch;
    rnn_utils::conf_t rnn;
};

struct grid_ctx_t {
    impl::engine_t *engine;
    const exec_ctx_t &ctx;
    const rnn_utils::user_data_t &user_data;
    const rnn_utils::workspace_t &workspace;
    const rnn_utils::scratch_t &scratch;
    rnn_utils::conf_t rnn;
};

struct cpy_ctx_t {
    memory_storage_t *cpy_in_lay;
    memory_storage_t *cpy_out_lay;
    memory_storage_t *cpy_in_iter;
    memory_storage_t *cpy_out_iter;
};

struct ref_rnn_common_base_t : public primitive_t {
    using primitive_t::primitive_t;

    status_t init(impl::engine_t *engine) override {
        CHECK(init_(engine));
        return status::success;
    };

    status_t execute(const exec_ctx_t &ctx) const override {
        CHECK(execute_(ctx));
        return status::success;
    };

protected:
    bool create_nested_matmul(impl::engine_t *engine,
            const std::shared_ptr<primitive_desc_t> &prim_desc,
            std::shared_ptr<impl::primitive_t> &prim);

    virtual status_t init_(impl::engine_t *engine) = 0;

    void debug_print(const rnn_utils::conf_t &rnn, dim_t slc, dim_t sic,
            bool with_bias, bool with_dst_iter) const;
    void get_user_data(const exec_ctx_t &ctx, rnn_utils::user_data_t &user_data,
            cpy_ctx_t &cpy_ctx, bool is_fwd, const rnn_pd_t *pd) const;

    virtual status_t execute_(const exec_ctx_t &ctx) const = 0;

    virtual status_t linear_execution(const grid_ctx_t &grid_struct) = 0;

    virtual status_t cell_execution(const cell_ctx_t &cell_struct) = 0;

    virtual status_t matmul_primitive(impl::engine_t *engine,
            const exec_ctx_t &ctx, std::unique_ptr<memory_storage_t> &a,
            std::unique_ptr<memory_storage_t> &b,
            std::unique_ptr<memory_storage_t> &c,
            matmul_kind_t matmul_kind) const = 0;

    status_t launch_copy(bool fwd, const exec_ctx_t &ctx,
            const kernel_t &cpy_kernel, const sycl_rnn_copy_conf_t &copy_conf,
            ::sycl::range<3> global_range, ::sycl::range<3> local_range,
            const memory_storage_t &input,
            const memory_storage_t &output) const;

    status_t do_copy(bool fwd, const exec_ctx_t &ctx, size_t batch_range,
            size_t lay_iter_range, size_t channel_range,
            const sycl_rnn_copy_conf_t &copy_conf, const kernel_t &cpy_kernel,
            const memory_storage_t &input,
            const memory_storage_t &output) const;

    virtual status_t copy_init_layer(const exec_ctx_t &ctx, dim_t batch,
            dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
            dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const = 0;
    virtual status_t copy_init_iter(const exec_ctx_t &ctx, dim_t batch,
            dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
            dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const = 0;
    virtual status_t copy_res_layer(const exec_ctx_t &ctx, dim_t batch,
            dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
            dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const = 0;
    virtual status_t copy_res_iter(const exec_ctx_t &ctx, dim_t batch,
            dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
            dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const = 0;

    status_t execution_loop(const grid_ctx_t &grid_struct);

    // offset variables set in workspace and used in offset calculations for
    // grid & cell execution and fwd & bwd kernel macros
    dim_t ws_gates_offset_ = 0;
    dim_t ws_states_offset_ = 0;
    dim_t ws_c_states_offset_ = 0;
    dim_t ws_grid_comp_offset_ = 0;
    dim_t ws_bias_offset_ = 0;

    // ptrs for storing weight offsets which are pre-calculated in
    // in grid execution as weights_*_assing_func
    std::vector<dim_t> wei_layer_offsets;
    std::vector<dim_t> wei_iter_offsets;

    std::function<status_t(const cell_ctx_t &)> cell_func;
    std::function<status_t(const grid_ctx_t &)> grid_func;
};

struct ref_rnn_fwd_t : ref_rnn_common_base_t {
    using ref_rnn_common_base_t::ref_rnn_common_base_t;

    using base_pd_t = gpu_rnn_fwd_pd_t;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T("ref:any", ref_rnn_fwd_t);

        status_t init(impl::engine_t *engine);

        status_t set_default_params();

        rnn_utils::conf_t rnn_conf = {};
        data_type_t acc_data_t = data_type::undef;
        data_type_t src_type = data_type::undef;
        data_type_t weights_type = data_type::undef;

        std::shared_ptr<primitive_desc_t> vanilla_cell_act_pd_;
        std::shared_ptr<primitive_desc_t> matmul_iter_fwd_pd_;
        std::shared_ptr<primitive_desc_t> matmul_layer_fwd_pd_;

        sycl_rnn_copy_conf_t copy_init_layer_conf_;
        sycl_rnn_copy_conf_t copy_init_iter_conf_;
        sycl_rnn_copy_conf_t copy_res_layer_conf_;
        sycl_rnn_copy_conf_t copy_res_iter_conf_;
        sycl_rnn_bias_fwd_conf_t sycl_rnn_bias_fwd_conf_t_;

    private:
        void init_scratchpad(dim_t workspace_size) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, workspace_size, 1);

            rnn_utils::scratch_t::book_fwd(scratchpad, rnn_conf,
                    {matmul_iter_fwd_pd_.get(), matmul_layer_fwd_pd_.get()});
        }
    };

protected:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t init_(impl::engine_t *engine) override;
    status_t execute_(const exec_ctx_t &ctx) const override;

    status_t linear_execution(const grid_ctx_t &grid_struct) override;

    status_t cell_execution(const cell_ctx_t &cell_struct) override;

    status_t matmul_primitive(impl::engine_t *engine, const exec_ctx_t &ctx,
            std::unique_ptr<memory_storage_t> &a,
            std::unique_ptr<memory_storage_t> &b,
            std::unique_ptr<memory_storage_t> &c,
            matmul_kind_t matmul_kind) const override;

    status_t copy_init_layer(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_init_iter(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_res_layer(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_res_iter(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;

    status_t rnn_bias(const exec_ctx_t &ctx, dim_t batch, dim_t dhc, dim_t iter,
            dim_t lay, dim_t dir, const rnn_utils::workspace_t &ws,
            const rnn_utils::scratch_t &scratch,
            const rnn_utils ::user_data_t &user_data) const;

    std::shared_ptr<impl::primitive_t> matmul_layer_fwd_;
    std::shared_ptr<impl::primitive_t> matmul_iter_fwd_;
    kernel_t copy_fwd_kernel_;

    kernel_t bias_fwd_kernel_;
};

struct ref_rnn_bwd_t : ref_rnn_common_base_t {
    using ref_rnn_common_base_t::ref_rnn_common_base_t;

    using base_pd_t = gpu_rnn_bwd_pd_t;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) = default;

        DECLARE_COMMON_PD_T("ref:any", ref_rnn_bwd_t);

        status_t init(impl::engine_t *engine);

        status_t set_default_params();

        rnn_utils::conf_t rnn_conf = {};
        data_type_t acc_data_t = data_type::undef;
        data_type_t src_type = data_type::undef;
        data_type_t weights_type = data_type::undef;

        std::shared_ptr<primitive_desc_t> vanilla_cell_act_pd_;
        std::shared_ptr<primitive_desc_t> matmul_iter_bwd_pd_;
        std::shared_ptr<primitive_desc_t> matmul_layer_bwd_pd_;
        std::shared_ptr<primitive_desc_t> matmul_diff_wei_iter_pd_;
        std::shared_ptr<primitive_desc_t> matmul_diff_wei_layer_pd_;

        sycl_rnn_copy_conf_t copy_init_layer_conf_;
        sycl_rnn_copy_conf_t copy_init_iter_conf_;
        sycl_rnn_copy_conf_t copy_res_layer_conf_;
        sycl_rnn_copy_conf_t copy_res_iter_conf_;
        sycl_rnn_bias_bwd_conf_t sycl_rnn_bias_bwd_conf_t_;

    private:
        void init_scratchpad(dim_t workspace_size) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, workspace_size, 1);

            rnn_utils::scratch_t::book_bwd(scratchpad, rnn_conf,
                    {matmul_iter_bwd_pd_.get(), matmul_layer_bwd_pd_.get(),
                            matmul_diff_wei_iter_pd_.get(),
                            matmul_diff_wei_layer_pd_.get()});
        }
    };

protected:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t init_(impl::engine_t *engine) override;
    status_t execute_(const exec_ctx_t &ctx) const override;

    status_t linear_execution(const grid_ctx_t &grid_struct) override;

    status_t cell_execution(const cell_ctx_t &cell_struct) override;

    status_t matmul_primitive(impl::engine_t *engine, const exec_ctx_t &ctx,
            std::unique_ptr<memory_storage_t> &a,
            std::unique_ptr<memory_storage_t> &b,
            std::unique_ptr<memory_storage_t> &c,
            matmul_kind_t matmul_kind) const override;

    status_t copy_init_layer(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_init_iter(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_res_layer(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t copy_res_iter(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
            dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer, dim_t n_dir,
            dim_t states_ws_ld, const memory_storage_t &input,
            const memory_storage_t &output) const override;
    status_t rnn_bias(const exec_ctx_t &ctx, dim_t batch, dim_t dhc, dim_t iter,
            dim_t lay, dim_t dir, dim_t n_layer,
            const std::unique_ptr<dnnl::impl::memory_storage_t>
                    &diff_states_layer,
            const std::unique_ptr<dnnl::impl::memory_storage_t> &diff_cell_iter,
            const rnn_utils ::user_data_t &user_data,
            const std::unique_ptr<dnnl::impl::memory_storage_t> &scratch_gate,
            const std::unique_ptr<dnnl::impl::memory_storage_t> &diff_gates)
            const;

    std::shared_ptr<impl::primitive_t> matmul_layer_bwd_;
    std::shared_ptr<impl::primitive_t> matmul_iter_bwd_;
    std::shared_ptr<impl::primitive_t> matmul_diff_wei_layer_;
    std::shared_ptr<impl::primitive_t> matmul_diff_wei_iter_;
    kernel_t copy_bwd_kernel_;

    kernel_t bias_bwd_kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
