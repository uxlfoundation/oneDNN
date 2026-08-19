/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "gpu/intel/conv/jit.hpp"

#include <utility>

#include "common/reorder.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/gpu_zero_points_conv.hpp"
#include "gpu/intel/conv/jit/config.hpp"
#include "gpu/intel/conv/jit/kernel.hpp"
#include "gpu/intel/conv/jit/tiler.hpp"
#include "gpu/intel/conv/jit/zero_out.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/logging.hpp"
#include "gpu/intel/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {

using namespace jit;

// Nested reorder between the user layout and the scratchpad compute layout.
struct reorder_data_t {
    std::shared_ptr<primitive_desc_t> pd;
    memory_desc_t compute_md;
    int arg_key = DNNL_ARG_UNDEF;
    uint32_t scratchpad_key = 0;
    uint32_t nested_key = 0;
    bool is_input = false;
};

struct pd_data_t {
    config_t pd_cfg;
    tensor_config_t tensor_cfg;
    std::vector<kernel_info_t> kernel_infos;
    // Parallel to kernel_infos, set for pre/post reorder entries only.
    std::vector<reorder_data_t> reorders;
    std::shared_ptr<primitive_desc_t> zp_pd;
};

struct tensor_ref_t {
    const layout_param_t *layout;
    const memory_desc_t *md;
};

static tensor_ref_t get_tensor_ref(const config_t &cfg,
        const convolution_pd_t *pd, const std::string &name) {
    if (name == "src") return {&cfg.src_layout(), pd->invariant_src_md()};
    if (name == "wei") return {&cfg.wei_layout(), pd->invariant_wei_md()};
    if (name == "bia") return {&cfg.bia_layout(), pd->invariant_bia_md()};
    if (name == "dst") return {&cfg.dst_layout(), pd->invariant_dst_md()};
    return {};
}

// GRF mode of the convolution kernel, see init_regs(). The pd-time config has
// no GRF mode set yet unless it's overridden from the environment.
static int conv_regs(const config_t &cfg) {
    if (cfg.options_param().is_overridden("regs")) return cfg.regs();
    return default_regs(cfg);
}

#define CONV_CHECK(_st) \
    do { \
        status_t st = (_st); \
        if (st == status::runtime_error) \
            VERROR(primitive, gpu, "%s,%s", pd->info(engine), \
                    "internal error"); \
        CHECK(st); \
    } while (false)

class gen_t {
public:
    static const int max_kernels = 16;

    template <typename T>
    static status_t init_pd(T *pd, const impl::engine_t *engine) {
        try {
            using intel::engine_t;
            auto *intel_engine = utils::downcast<const engine_t *>(engine);

            VDISPATCH_CONV_IC(intel_engine->mayiuse_ngen_kernels(),
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_CONV_IC(
                    pd->set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);

            problem_t prb;
            CONV_CHECK(prb.init(engine, pd));

            // The IR generator hard-codes s32 as the type for problem parameters.
            VDISPATCH_CONV_IC(
                    INT_MAX > std::max({prb.mb, prb.ic, prb.id, prb.ih, prb.iw,
                            prb.oc, prb.od, prb.oh, prb.ow, prb.kd, prb.kh,
                            prb.kw, prb.sd, prb.sh, prb.sw, prb.pd, prb.ph,
                            prb.pw, prb.dd, prb.dh, prb.dw}),
                    VERBOSE_SHAPE_RESTRICTION);

            pd->data = std::make_shared<pd_data_t>();
            CONV_CHECK(init_pd_time_cfg(
                    prb, pd->data->pd_cfg, engine, pd, &pd->attr_));

            if (pd->data->pd_cfg.zp_cfg().needs_src_reorder_precalc
                    || pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                primitive_attr_t attr;
                if (pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                    int mask = pd->attr_.zero_points_.get_mask(DNNL_ARG_SRC);
                    CHECK(attr.zero_points_.set(DNNL_ARG_SRC, mask));
                    CHECK(attr.post_ops_.append_eltwise(
                            1.f, alg_kind::eltwise_linear, -1.f, 0.f));
                }
                dim_t I[3], O[3], P[3], D[3];
                prepare_zp_precompute(prb, I, O, P, D);
                CONV_CHECK(create_zp_precompute_conv_pd(pd->data->zp_pd, engine,
                        attr, pd->weights_md(), I, O, P, D, data_type::f32,
                        pd->get_prop_kind(),
                        !pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc));
                if (pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                    auto scratchpad = pd->scratchpad_registry().registrar();
                    scratchpad.book(memory_tracking::names::key_nested_multiple,
                            pd->data->zp_pd->scratchpad_registry());
                }
            }

            pd->data->tensor_cfg
                    = get_tensor_config(pd->data->pd_cfg, zp_md_in(*pd->data));
            pd->data->kernel_infos.reserve(max_kernels);
            pd->data->reorders.reserve(max_kernels);
            CONV_CHECK(init_kernel_infos(pd, engine));

            return status::success;
        } catch (std::exception &err) {
            return report_runtime_error(pd, engine, err.what());
        }
    }

    gen_t() = default;

    template <typename T>
    status_t init(T *primitive, impl::engine_t *engine) {
        auto *pd = primitive->pd();
        auto &data = *pd->data;
        auto tiler = std::make_shared<tiler_t>(data.pd_cfg);

        if (primitive->cache_blob()) {
            int32_t version;
            CONV_CHECK(primitive->cache_blob().get_value(
                    (uint8_t *)&version, sizeof(version)));
            primitive->set_version(version);
        }

        bool ok = false;
        int max_tries = 100;
        config_t cfg;
        layout_t zp_dst;
        if (data.zp_pd) zp_dst = make_layout(*zp_md_out(data));

        if (primitive->cache_blob()) {
            tiler->set_cur_version(primitive->version());
        }

        int desc_idx = gpu_utils::dev_getenv("desc_idx", -1);
        for (int try_iter = 0; try_iter < max_tries; try_iter++) {
            if (try_iter < desc_idx) {
                tiler->move_next(cfg);
                continue;
            }
            if ((try_iter != 0 && !tiler->is_tuning_mode())) {
                tiler->move_next(cfg);
            }
            try {
                cfg = data.pd_cfg;
                cfg.set_pd(pd);
                cfg.set_tiler(tiler);
                CONV_CHECK(init_cfg(cfg, primitive));

                if (!tiler->is_grf_limit_ok(cfg)) continue;

                gpu_info() << "Configuration:";
                gpu_info() << cfg;

                init_nd_ranges(primitive, cfg);
                auto &kernel_infos = data.kernel_infos;

                // Nested primitives have to be created before the kernels
                // below: they register their cache blob blocks right away
                // while the kernels are registered only after the loop.
                // Does not depend on the tiler, create it once.
                for (int i = 0; i < int(kernel_infos.size()) && !zp_prim_;
                        i++) {
                    if (kernel_infos[i].id() != kernel_id_t::zp_precalc)
                        continue;
                    gpu_assert(data.zp_pd);
                    CONV_CHECK(primitive->create_nested_primitive(
                            zp_prim_, data.zp_pd, engine));
                }

                reorder_prims_.resize(kernel_infos.size());
                for (int i = 0; i < int(kernel_infos.size()); i++) {
                    auto &pd = data.reorders[i].pd;
                    if (!pd || reorder_prims_[i]) continue;
                    CONV_CHECK(primitive->create_nested_primitive(
                            reorder_prims_[i], pd, engine));
                }

                std::vector<compute::kernel_t> tmp_kernels;
                for (int i = 0; i < int(kernel_infos.size()); i++) {
                    auto &info = kernel_infos[i];
                    switch (info.id()) {
                        case kernel_id_t::convolution: {
                            tmp_kernels.push_back(make_kernel<conv_kernel_t>(
                                    primitive, /*register_kernel=*/false,
                                    engine, cfg, info,
                                    nd_ranges_[i].local_range(), zp_dst));
                            break;
                        }
                        case kernel_id_t::zero_out:
                            if (can_skip_zero_out(info, cfg)) {
                                tmp_kernels.emplace_back();
                                continue;
                            }
                            tmp_kernels.push_back(
                                    make_kernel<zero_out_kernel_t>(primitive,
                                            /*register_kernel=*/false, engine,
                                            cfg.options(), info, engine));
                            break;

                        case kernel_id_t::pre_reorder:
                        case kernel_id_t::post_reorder:
                        case kernel_id_t::zp_precalc:
                            tmp_kernels.emplace_back();
                            continue;

                        default: gpu_error_not_expected();
                    }
                    if (!tmp_kernels[i])
                        return report_runtime_error(pd, engine);
                }
                ok = true;
                primitive->set_version(tiler->cur_version());
                kernels_ = std::move(tmp_kernels);
                break;
            } catch (ngen::out_of_registers_exception &err) {
                if (handle_exception(try_iter, max_tries))
                    return report_runtime_error(pd, engine, err.what());
                tiler->notify_out_of_registers(cfg);
                continue;
            } catch (std::exception &err) {
                if (handle_exception(try_iter, max_tries))
                    return report_runtime_error(pd, engine, err.what());
                continue;
            }
        }
        if (!ok) return report_runtime_error(pd, engine);
        gpu_assert(kernels_.size() == data.kernel_infos.size());
        CONV_CHECK(primitive->register_kernels(kernels_));

        tiler_t::after_create_hook(cfg, primitive);
        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto *pd = primitive->pd();
        auto *engine = ctx.stream()->engine();
        auto &data = *pd->data;
        auto &kernel_infos = data.kernel_infos;

        tiler_t::before_exec_hook(primitive, ctx.stream());

        int max_stage = 100;
        int nsubmitted = 0;
        int nkernels = int(kernel_infos.size());
        for (int stage = 0; stage < max_stage; stage++) {
            for (int i = 0; i < nkernels; i++) {
                auto &info = kernel_infos[i];
                if (info.stage_id() != stage) continue;

                if (kernels_[i]) {
                    std::vector<memory_storage_wrapper_t> storage_list;
                    info.init_memory_storage_list(storage_list, ctx, primitive);

                    compute::kernel_arg_list_t arg_list;
                    info.set_args(arg_list, storage_list);

                    CONV_CHECK(primitive->parallel_for(
                            ctx, nd_ranges_[i], kernels_[i], arg_list));
                } else if (info.id() == kernel_id_t::zp_precalc) {
                    auto scratchpad_arg
                            = [&](std::unique_ptr<memory_t, memory_deleter_t>
                                              &retn,
                                      const std::string &name,
                                      const memory_desc_t *md) {
                        auto s = ctx.get_scratchpad_grantor()
                                         .get_memory_storage(info.key(name));
                        return safe_ptr_assign(
                                retn, new memory_t(engine, md, std::move(s)));
                    };
                    gpu_assert(zp_prim_);
                    std::unique_ptr<memory_t, memory_deleter_t> zp_src, zp_dst;
                    CONV_CHECK(scratchpad_arg(
                            zp_src, "src_zero_points", zp_md_in(data)));
                    CONV_CHECK(scratchpad_arg(zp_dst, "dst", zp_md_out(data)));

                    exec_args_t e_args;
                    auto src_zp_idx = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
                    e_args[src_zp_idx] = ctx.args().at(src_zp_idx);
                    e_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_WEIGHTS);
                    e_args[DNNL_ARG_SRC] = memory_arg_t {zp_src.get(), true};
                    e_args[DNNL_ARG_DST] = memory_arg_t {zp_dst.get(), false};
                    exec_ctx_t e_ctx(ctx, std::move(e_args));
                    const auto nm = memory_tracking::names::key_nested_multiple;
                    auto *nested_grantor = create_nested_grantor(
                            ctx.get_scratchpad_grantor(), nm,
                            zp_prim_->pd()->scratchpad_registry());
                    e_ctx.set_scratchpad_grantor(nested_grantor);
                    CONV_CHECK(zp_prim_->execute(e_ctx));
                } else if (data.reorders[i].pd) {
                    CONV_CHECK(execute_reorder(
                            ctx, data.reorders[i], *reorder_prims_[i]));
                }
                nsubmitted++;
                if (nsubmitted == nkernels) break;
            }
        }

        return status::success;
    }

private:
    static status_t execute_reorder(const exec_ctx_t &ctx,
            const reorder_data_t &r, const impl::primitive_t &prim) {
        auto storage = ctx.get_scratchpad_grantor().get_memory_storage(
                r.scratchpad_key);
        std::unique_ptr<memory_t, memory_deleter_t> compute_mem;
        CHECK(safe_ptr_assign(compute_mem,
                new memory_t(ctx.stream()->engine(), &r.compute_md,
                        std::move(storage))));

        exec_args_t r_args;
        auto compute_arg = memory_arg_t {compute_mem.get(), !r.is_input};
        r_args[DNNL_ARG_FROM]
                = r.is_input ? ctx.args().at(r.arg_key) : compute_arg;
        r_args[DNNL_ARG_TO]
                = r.is_input ? compute_arg : ctx.args().at(r.arg_key);
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        auto *nested_grantor
                = create_nested_grantor(ctx.get_scratchpad_grantor(),
                        r.nested_key, prim.pd()->scratchpad_registry());
        r_ctx.set_scratchpad_grantor(nested_grantor);
        return prim.execute(r_ctx);
    }

    static const memory_desc_t *zp_md_in(const pd_data_t &data) {
        if (!data.zp_pd) return nullptr;
        const bool is_bwd_d
                = (data.zp_pd->get_prop_kind() == prop_kind::backward_data);
        return (is_bwd_d) ? data.zp_pd->diff_dst_md() : data.zp_pd->src_md();
    }

    static const memory_desc_t *zp_md_out(const pd_data_t &data) {
        if (!data.zp_pd) return nullptr;
        const bool is_bwd_d
                = (data.zp_pd->get_prop_kind() == prop_kind::backward_data);
        return (is_bwd_d) ? data.zp_pd->diff_src_md() : data.zp_pd->dst_md();
    }

    template <typename T>
    static kernel_info_t &create_kernel_info(T *pd, kernel_id_t kernel_id) {
        auto &infos = pd->data->kernel_infos;
        gpu_assert((int)infos.size() + 1 <= max_kernels);
        infos.emplace_back();
        pd->data->reorders.emplace_back();
        auto &ret = infos.back();
        ret.set_id(kernel_id);
        return ret;
    }

    template <typename T>
    static status_t init_reorder(T *pd, const impl::engine_t *engine,
            kernel_id_t id, const std::string &name, int arg_key,
            int scratchpad_key, size_t scratchpad_size) {
        auto &data = *pd->data;
        auto t = get_tensor_ref(data.pd_cfg, pd, name);
        VDISPATCH_CONV_IC(
                t.layout, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, name.c_str());
        // Unnormalized layout matches the user memory descriptor ndims.
        auto &compute_layout = t.layout->compute_unnormalized();
        // Compensation is handled by the conv, reorder the plain data only.
        auto user_md = *t.md;
        user_md.extra = {};
        auto compute_md = to_md(compute_layout, user_md);
        VDISPATCH_CONV_IC(
                memory_desc_wrapper(compute_md).size() <= scratchpad_size,
                VERBOSE_SCRATCHPAD_LIMIT);

        create_kernel_info(pd, id);
        auto &r = data.reorders.back();
        r.compute_md = compute_md;
        r.arg_key = arg_key;
        r.scratchpad_key = into<uint32_t>(scratchpad_key);
        r.nested_key
                = into<uint32_t>(memory_tracking::names::key_nested_multiple
                        + data.reorders.size());
        r.is_input = (id == kernel_id_t::pre_reorder);
        // Match the conv kernel's GRF and systolic mode to avoid a GPU state
        // switch between the interleaved reorder and conv dispatches.
        gpu_primitive_attr_t gpu_attr(
                conv_regs(data.pd_cfg), data.pd_cfg.options().require_dpas());
        primitive_attr_t attr;
        CHECK(attr.set_gpu_attr(gpu_attr));
        CHECK(reorder_primitive_desc_create(r.pd, engine,
                r.is_input ? &user_md : &r.compute_md,
                r.is_input ? &r.compute_md : &user_md, &attr));
        pd->scratchpad_registry().registrar().book(
                r.nested_key, r.pd->scratchpad_registry());
        return status::success;
    }

    template <typename T>
    static status_t init_kernel_infos(T *pd, const impl::engine_t *engine) {
        auto &data = *pd->data;
        auto &cfg = data.pd_cfg;
        auto &conv_info = create_kernel_info(pd, kernel_id_t::convolution);
        auto &zp_precalc_info = (cfg.zp_cfg().needs_src_conv_precalc)
                ? create_kernel_info(pd, kernel_id_t::zp_precalc)
                : conv_info;

        static_assert(DNNL_ARG_UNDEF == memory_tracking::names::key_none,
                "Undefined argument and empty scratchpad key are out of sync!");

        // Initialize kernel arguments.
        int scratchpad_key = memory_tracking::names::key_none;
        for (auto &t : data.tensor_cfg.tensors()) {
            auto buf_type = [pd](int key) {
                if (key & DNNL_ARG_ATTR_ZERO_POINTS) {
                    key &= ~DNNL_ARG_ATTR_ZERO_POINTS;
                    auto &zp = pd->attr()->zero_points_;
                    if (zp.get(key).is_host_scalar()) switch (key) {
                            case DNNL_ARG_SRC:
                                return (zp.has_default_values(DNNL_ARG_WEIGHTS))
                                        ? dsl::type_t::f32()
                                        : dsl::type_t::s32();
                            case DNNL_ARG_DST: return dsl::type_t::f32();
                            default: return to_ir(zp.get(key).get_data_type());
                        }
                } else if (key & DNNL_ARG_ATTR_SCALES) {
                    key &= ~DNNL_ARG_ATTR_SCALES;
                    auto &sc = pd->attr()->scales_;
                    if (sc.get(key).is_host_scalar())
                        return to_ir(sc.get(key).get_data_type());
                }
                return dsl::type_t::byte(dsl::type::attr_t::ptr);
            };
            const bool wei_reorder_precalc = (t.name == "wei")
                    && cfg.zp_cfg().needs_src_reorder_precalc;
            const bool src_conv_precalc = (t.name == "src_zero_points")
                    && cfg.zp_cfg().needs_src_conv_precalc;

            size_t compute_size = size_bytes(t.compute_layout);
            int compute_arg_key = t.arg_key;
            const auto compute_buf
                    = var_t::make(buf_type(compute_arg_key), t.name);

            if (compute_arg_key == DNNL_ARG_UNDEF) {
                gpu_assert(!t.needs_reorder);
                gpu_assert(!t.needs_zero_out);
                gpu_error_not_expected();
                continue;
            }

            auto add_compute_arg
                    = [&](kernel_info_t &ki, const expr_t &buf, bool is_input) {
                if (t.needs_reorder || src_conv_precalc)
                    ki.register_scratchpad_arg(
                            buf, compute_arg_key, is_input, compute_size);
                else if (buf.type().is_ptr())
                    ki.register_user_arg(buf, compute_arg_key, is_input);
                else
                    ki.register_immediate_arg(buf, expr_t(), compute_arg_key);
            };
            auto scratchpad_book = [&](int key) {
                pd->scratchpad_registry().registrar().book(into<uint32_t>(key),
                        compute_size, 1, OCL_BUFFER_ALIGNMENT);
            };
            auto create_zero_out_info = [&]() -> kernel_info_t & {
                auto &zero_out_info
                        = create_kernel_info(pd, kernel_id_t::zero_out);
                auto size_var = var_t::make(dsl::type_t::u32(), "size");
                zero_out_info.register_immediate_arg(
                        size_var, into<uint32_t>(compute_size));
                zero_out_info.set_nd_range(zero_out_kernel_desc_t::nd_range(
                        cfg.simd(), compute_size));
                return zero_out_info;
            };

            if (t.needs_reorder || src_conv_precalc) {
                int user_arg_key = compute_arg_key;
                auto user_buf = make_buffer(t.name + "_user");
                compute_arg_key = ++scratchpad_key;

                if (!src_conv_precalc && t.is_input)
                    CHECK(init_reorder(pd, engine, kernel_id_t::pre_reorder,
                            t.name, user_arg_key, compute_arg_key,
                            compute_size));
                if (!src_conv_precalc && t.is_output)
                    CHECK(init_reorder(pd, engine, kernel_id_t::post_reorder,
                            t.name, user_arg_key, compute_arg_key,
                            compute_size));
                if (src_conv_precalc) {
                    scratchpad_book(++scratchpad_key);
                    create_zero_out_info().register_scratchpad_arg(compute_buf,
                            scratchpad_key, /*is_input=*/false, compute_size);

                    zp_precalc_info.register_scratchpad_arg(compute_buf,
                            scratchpad_key, /*is_input=*/true, compute_size);
                    const auto &prb = cfg.prb();
                    compute_size = int64_t(compute_size) * sizeof(int32_t)
                            * ir_utils::max_unique_pad_states(prb.od, prb.id,
                                    prb.kd, prb.dd, prb.pd, prb.sd, true)
                            * ir_utils::max_unique_pad_states(prb.oh, prb.ih,
                                    prb.kh, prb.dh, prb.ph, prb.sh, true)
                            * ir_utils::max_unique_pad_states(prb.ow, prb.iw,
                                    prb.kw, prb.dw, prb.pw, prb.sw, true)
                            * utils::rnd_up(prb.g * prb.oc, cfg.simd())
                            / std::min((prb.kd - 1) * (prb.dd + 1) + 1, prb.id)
                            / std::min((prb.kh - 1) * (prb.dh + 1) + 1, prb.ih)
                            / std::min((prb.kw - 1) * (prb.dw + 1) + 1, prb.iw)
                            / (prb.g * prb.ic);
                    add_compute_arg(zp_precalc_info, make_buffer("dst"), false);
                }
                scratchpad_book(compute_arg_key);
                if (wei_reorder_precalc) {
                    // user-supplied weights contain precomputed ZP values, so
                    // the buffer is to be passed to the conv alongside weights
                    conv_info.register_user_arg(
                            user_buf, user_arg_key, t.is_input && !t.is_output);
                }
            }
            if (t.needs_zero_out) {
                add_compute_arg(create_zero_out_info(), compute_buf, false);
            }
            add_compute_arg(conv_info, compute_buf, t.is_input && !t.is_output);
        }

        return status::success;
    }

    template <typename T>
    void init_nd_ranges(T *primitive, const config_t &cfg) {
        auto *pd = primitive->pd();
        auto &data = *pd->data;
        int nkernels = int(data.kernel_infos.size());
        nd_ranges_.resize(nkernels);
        for (int i = 0; i < nkernels; i++) {
            auto &info = data.kernel_infos[i];
            switch (info.id()) {
                case kernel_id_t::convolution:
                    // Convolution kernel info is initialized at PD creation
                    // time when ND range/grid information are not known yet so
                    // we need to directly query config here.
                    nd_ranges_[i] = cfg.nd_range();
                    break;
                case kernel_id_t::zero_out:
                    nd_ranges_[i] = info.nd_range();
                    break;
                case kernel_id_t::pre_reorder:
                case kernel_id_t::post_reorder:
                case kernel_id_t::zp_precalc: break;
                default: gpu_error_not_expected();
            }
        }
    }

    static bool can_skip_zero_out(
            const kernel_info_t &info, const config_t &cfg) {
        gpu_assert(info.id() == kernel_id_t::zero_out);
        auto &buf_name = info.arg_var(1).as<var_t>().name;
        if (buf_name == "wei") return cfg.can_skip_wei_zero_out();
        if (buf_name == "bia") return cfg.can_skip_bia_zero_out();
        return false;
    }

    static bool handle_exception(int iter, int max_iters) {
        return iter + 1 >= max_iters;
    }

    static status_t report_runtime_error(const convolution_pd_t *pd,
            const impl::engine_t *engine, const char *msg = "internal error") {
        VERROR(primitive, gpu, "%s,%s", pd->info(engine), msg);
        return status::runtime_error;
    }

    std::vector<compute::kernel_t> kernels_;
    std::vector<compute::nd_range_t> nd_ranges_;
    std::shared_ptr<impl::primitive_t> zp_prim_;
    // Parallel to kernel_infos.
    std::vector<std::shared_ptr<impl::primitive_t>> reorder_prims_;
};

status_t gen_fwd_t::pd_t::init(const impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_fwd(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_t::init_pd(this, engine));
    return status::success;
}

status_t gen_fwd_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_bwd_data_t::pd_t::init(const impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_bwd_d(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_t::init_pd(this, engine));
    return status::success;
}

status_t gen_bwd_weights_t::pd_t::init(const impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_bwd_w(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_t::init_pd(this, engine));
    return status::success;
}

status_t gen_bwd_data_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_bwd_weights_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
