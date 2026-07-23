/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GATED_MLP_HPP
#define GATED_MLP_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "utils/cfg.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

// GatedMLP primitive C API.
#include "src/common/gated_mlp_iface.hpp"

namespace gated_mlp {

// Argument aliases for gated MLP weight tensors.
#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2

dnnl_alg_kind_t str2activation(const char *str);
const char *activation2str(dnnl_alg_kind_t act);

struct settings_t : public base_settings_t {
    using base_settings_t::base_settings_t;

    prb_dims_t prb_dims;

    std::vector<std::vector<dnnl_data_type_t>> dt {{dnnl_f32}};
    std::vector<std::string> stag {tag::abx}, wtag {tag::abx}, dtag {tag::abx};
    std::vector<dnnl_alg_kind_t> activation {dnnl_eltwise_swish};

    const char *perf_template_csv() const {
        static const std::string args = "%sdt%,%stag%,%wtag%,%dtag%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return dt.size() == 1 && stag.size() == 1 && wtag.size() == 1
                && dtag.size() == 1 && activation.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public prb_dims_t, public base_prb_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.prb_dims, s.dt[0], s.stag[0], s.wtag[0], s.dtag[0],
                  s.activation[0], s.attributes.front(), s.ctx_init[0],
                  s.ctx_exe[0], s.impl_filter) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
    }

    prb_t(const prb_dims_t &prb_dims, const std::vector<dnnl_data_type_t> &dt,
            const std::string &stag, const std::string &wtag,
            const std::string &dtag, dnnl_alg_kind_t activation,
            const attr_t &attr, const thr_ctx_t &ctx_init,
            const thr_ctx_t &ctx_exe, const impl_filter_t &impl_filter)
        : prb_dims_t(prb_dims)
        , base_prb_t(FLAG_FWD, false, attr, impl_filter)
        , dt(dt)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , activation(activation)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe) {

        // Broadcast data types if needed: src, w_gate, w_up, w_down, dst.
        if (this->dt.size() == 1) {
            const auto val = this->dt[0];
            this->dt.assign(5, val);
        }

        // dims is [MB, IC, OC].
        mb = dims[0];
        ic = dims[1];
        oc = dims[2];

        // Determine ndims from tags. Explicit tags (e.g. "abc", "cab")
        // imply 3D; generic tags ("abx") default to 2D.
        auto tag_ndims = [](const std::string &t) -> int {
            if (t == tag::abx || t == tag::axb || t == tag::any
                    || t == tag::undef)
                return 0;
            int n = 0;
            for (char c : t)
                if (c >= 'a' && c <= 'l') n = std::max(n, c - 'a' + 1);
            return n;
        };
        ndims = std::max({2, tag_ndims(this->stag), tag_ndims(this->wtag),
                tag_ndims(this->dtag)});

        // Derive tensor shapes. For 3D ("fake batch"), prepend/insert a
        // unit dimension to match OV's layout: SRC/DST=[MB,1,IC],
        // W_GATE/W_UP=[1,IC,OC], W_DOWN=[1,OC,IC].
        if (ndims == 3) {
            src_dims = {mb, 1, ic};
            w_gate_dims = {1, ic, oc};
            w_up_dims = {1, ic, oc};
            w_down_dims = {1, oc, ic};
            dst_dims = {mb, 1, ic};
        } else {
            src_dims = {mb, ic};
            w_gate_dims = {ic, oc};
            w_up_dims = {ic, oc};
            w_down_dims = {oc, ic};
            dst_dims = {mb, ic};
        }

        // FLOPs: 3 matmuls + element-wise ops.
        // matmul(src, W_gate): 2*MB*IC*OC
        // matmul(src, W_up):   2*MB*IC*OC
        // matmul(gate, W_down): 2*MB*OC*IC
        ops = 6.0 * mb * ic * oc;

        repro = set_repro_line(); // must be last in ctor to collect right info
    }

    int64_t mb, ic, oc;
    int ndims;
    std::vector<dnnl_data_type_t> dt;
    std::string stag, wtag, dtag;
    dnnl_alg_kind_t activation;

    thr_ctx_t ctx_init, ctx_exe;

    double ops;
    dims_t src_dims, w_gate_dims, w_up_dims, w_down_dims, dst_dims;

    dnnl_data_type_t src_dt() const { return dt[0]; }
    dnnl_data_type_t w_gate_dt() const { return dt[1]; }
    dnnl_data_type_t w_up_dt() const { return dt[2]; }
    dnnl_data_type_t w_down_dt() const { return dt[3]; }
    dnnl_data_type_t dst_dt() const { return dt[4]; }
    dnnl_data_type_t get_dt(data_kind_t data_kind) const;

    // Required by init_memory_args (for runtime dims support).
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const override;

    static const prb_t *from(const base_prb_t *base_prb) {
        return downcast<const prb_t *>(base_prb);
    }

    void skip_unimplemented(res_t *res) const override;
    void skip_invalid(res_t *res) const override;
    std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const override;

private:
    std::string set_repro_line() override;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const base_prb_t *base_prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb_t::from(base_prb))
        , stag_({normalize_tag(p_->stag, p_->ndims)})
        , wtag_(normalize_tag(p_->wtag, p_->ndims))
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_dims_t &>(*p_);
    }

    double ops() const override { return p_->ops; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->dt;
    }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *wtag() const override { return &wtag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string wtag_, dtag_;
};

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res);
void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t);
void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args);

struct cfg_t : public base_cfg_t {
    cfg_t(const prb_t *prb, const std::vector<data_kind_t> &kinds);

    cfg_entry_t::cfg_map_t get_cfg_map(data_kind_t kind) const override;

    float get_density(const density_args_t &density_args) const override;
};

int bench(int argc, char **argv);

} // namespace gated_mlp

#endif
