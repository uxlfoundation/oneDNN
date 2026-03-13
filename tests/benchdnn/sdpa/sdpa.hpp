/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef SDPA_HPP
#define SDPA_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

// Internal test interface for SDPA primitive.
#include "tests/gtests/internals/sdpa_internal.hpp"

namespace sdpa {

// Attention mask types matching dnnl_attn_mask_type_t from sdpa_types.hpp.
enum mask_type_t {
    MASK_NONE = 0,
    MASK_BUFFER = 1,
    MASK_CAUSAL_TOP_LEFT = 2,
    MASK_CAUSAL_BOTTOM_RIGHT = 3,
};
mask_type_t str2mask_type(const char *str);
const char *mask_type2str(mask_type_t mt);

// Scale type.
enum scale_type_t {
    SCALE_NONE = 0,
    SCALE_MUL = 1,
    SCALE_DIV = 2,
};
scale_type_t str2scale_type(const char *str);
const char *scale_type2str(scale_type_t st);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_vdims_t prb_vdims;

    // Data types: Q, K, V, DST (4 entries).
    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> qtag {tag::abx}, ktag {tag::abx},
            vtag {tag::abx}, dtag {tag::abx};
    std::vector<mask_type_t> mask_type {MASK_NONE};
    std::vector<scale_type_t> scale_type {SCALE_NONE};
    std::vector<dnnl_dim_t> kv_head_number {0};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%stag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dt.size() == 1 && qtag.size() == 1 && ktag.size() == 1
                && vtag.size() == 1 && dtag.size() == 1
                && mask_type.size() == 1 && scale_type.size() == 1
                && kv_head_number.size() == 1
                && base_settings_t::has_single_setup();
    }
};

// SDPA problem: Q * K^T -> scale -> [mask] -> softmax -> * V -> DST
//
// Dimensions (4D example: batch x heads x seq_len x head_size):
//   Q:   [batch..., queries, head_size]
//   K:   [batch..., head_size, keys]
//   V:   [batch..., keys, values]
//   DST: [batch..., queries, values]
//   Mask (optional): [batch..., queries, keys] (or broadcast)
//
// vdims encoding: Q_dims:K_dims:V_dims (3 sets, colon-separated)
// Example: 2x16x128x64:2x16x64x128:2x16x128x64
struct prb_t : public prb_vdims_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_vdims, s.dt[0], s.qtag[0], s.ktag[0], s.vtag[0],
                  s.dtag[0], s.mask_type[0], s.scale_type[0],
                  s.kv_head_number[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_vdims_t &prb_vdims,
            const std::vector<dnnl_data_type_t> &dt, const std::string &qtag,
            const std::string &ktag, const std::string &vtag,
            const std::string &dtag, mask_type_t mask_type,
            scale_type_t scale_type, dnnl_dim_t kv_head_number,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, const impl_filter_t &impl_filter)
        : prb_vdims_t(prb_vdims)
        , dt(dt)
        , qtag(qtag)
        , ktag(ktag)
        , vtag(vtag)
        , dtag(dtag)
        , mask_type(mask_type)
        , scale_type(scale_type)
        , kv_head_number(kv_head_number)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , impl_filter(impl_filter) {

        // Broadcast data types if needed: Q,K,V,DST
        if (this->dt.size() == 1) {
            const auto val = this->dt[0];
            this->dt.assign(4, val);
        }

        const auto &qdims = q_dims();
        const auto &kdims = k_dims();
        const auto &vdims_ref = v_dims();
        n_queries = qdims[ndims - 2];
        head_size = qdims[ndims - 1];
        n_keys = kdims[ndims - 1];
        n_values = vdims_ref[ndims - 1];

        // Compute dst_dims from Q and V dims.
        dst_dims.resize(ndims);
        for (int i = 0; i < ndims - 2; i++)
            dst_dims[i] = qdims[i];
        dst_dims[ndims - 2] = n_queries;
        dst_dims[ndims - 1] = n_values;

        // Compute mask dims if needed.
        if (with_mask()) {
            msk_dims.resize(ndims);
            for (int i = 0; i < ndims - 2; i++)
                msk_dims[i] = qdims[i];
            msk_dims[ndims - 2] = n_queries;
            msk_dims[ndims - 1] = n_keys;
        }

        mb = 1;
        for (int i = 0; i < ndims - 2; i++)
            mb *= qdims[i];

        // ops: QK^T (2*mb*queries*keys*head_size) + VS
        // (2*mb*queries*values*keys)
        ops = 2.0 * mb * n_queries * n_keys * head_size
                + 2.0 * mb * n_queries * n_values * n_keys;

        repro = set_repro_line();
    }

    int64_t n_queries, head_size, n_keys, n_values, mb;
    dir_t dir = FLAG_FWD; // Always forward.
    std::vector<dnnl_data_type_t> dt;
    std::string qtag, ktag, vtag, dtag;
    mask_type_t mask_type;
    scale_type_t scale_type;
    dnnl_dim_t kv_head_number;

    bool inplace = false;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    impl_filter_t impl_filter;

    double ops;
    dims_t dst_dims;
    dims_t msk_dims;

    const dims_t &q_dims() const { return vdims[0]; }
    const dims_t &k_dims() const { return vdims[1]; }
    const dims_t &v_dims() const { return vdims[2]; }

    bool with_mask() const {
        return mask_type == MASK_BUFFER;
    }
    bool with_causal_mask() const {
        return mask_type == MASK_CAUSAL_TOP_LEFT
                || mask_type == MASK_CAUSAL_BOTTOM_RIGHT;
    }
    bool with_scale() const { return scale_type != SCALE_NONE; }
    bool invert_scale() const { return scale_type == SCALE_DIV; }

    dnnl_data_type_t q_dt() const { return dt[0]; }
    dnnl_data_type_t k_dt() const { return dt[1]; }
    dnnl_data_type_t v_dt() const { return dt[2]; }
    dnnl_data_type_t dst_dt() const { return dt[3]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    // Required by init_memory_args template (for runtime dims support).
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const;

    const char *str() const { return repro.c_str(); }

private:
    std::string repro;
    std::string set_repro_line();
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({normalize_tag(p_->qtag, p_->ndims),
                  normalize_tag(p_->ktag, p_->ndims),
                  normalize_tag(p_->vtag, p_->ndims)})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    double ops() const override { return p_->ops; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;

    float get_density(const density_args_t &density_args) const override;
};

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);
std::vector<int> supported_exec_args(dir_t dir);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);

int bench(int argc, char **argv);

} // namespace sdpa

#endif
