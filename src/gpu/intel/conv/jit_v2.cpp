/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/conv/jit_v2.hpp"

#include "common/utils.hpp"
#include "gpu/intel/conv/jit/v2/bridge.hpp"
#include "gpu/intel/conv/jit/v2/debug.hpp"
#include "gpu/intel/conv/jit/v2/plan_registry.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"
#include "gpu/intel/logging.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace v2 {

using namespace jit;
using namespace jit::v2;

void maybe_init_layout(
        memory_desc_t &md, const layout_tag_t &_tag, bool remove_a_dim) {
    if (md.format_kind != format_kind::any) return;
    // This code sets the any memory descriptor as follows:
    // - Blocking is set according to the tag from the kernel descriptor
    // - Type is copied from the user descriptor
    // XXX: When internal blocked layouts are added, this needs an adjustment.
    // The user layout should be set to a plain layout for consistency with
    // other layers.
    auto layout = to_layout(_tag, md, remove_a_dim);
    md = layout.to_dnnl(md.dims);
}

status_t init_default_layouts(pd_t *pd) {
    auto &src_md = *const_cast<memory_desc_t *>(pd->invariant_src_md());
    auto &dst_md = *const_cast<memory_desc_t *>(pd->invariant_dst_md());
    if (src_md.format_kind == format_kind::any)
        set_default_format(src_md, "axb");
    if (dst_md.format_kind == format_kind::any)
        set_default_format(dst_md, "axb");
    return status::success;
}

status_t init_layouts(const kernel_desc_t &desc, pd_t *pd) {
    auto &src_md = *const_cast<memory_desc_t *>(pd->invariant_src_md());
    auto &wei_md = *const_cast<memory_desc_t *>(pd->invariant_wei_md());
    auto &dst_md = *const_cast<memory_desc_t *>(pd->invariant_dst_md());
    auto &bia_md = *const_cast<memory_desc_t *>(pd->invariant_bia_md());
    maybe_init_layout(src_md, desc.src_tag, false);
    maybe_init_layout(wei_md, desc.wei_tag, !pd->with_groups());
    maybe_init_layout(dst_md, desc.dst_tag, false);
    maybe_init_layout(bia_md, make_layout_tag(tensor_kind_t::bias, "a"), false);
    return status::success;
}

void iter_md(const pd_t *pd,
        const std::function<void(const memory_desc_t &)> &func) {
    func(*pd->invariant_src_md());
    func(*pd->invariant_wei_md());
    func(*pd->invariant_dst_md());
    func(*pd->invariant_bia_md());
    auto &post_ops = pd->attr()->post_ops_;
    for (int i = 0; i < post_ops.len(); i++) {
        auto &e = post_ops.entry_[i];
        if (e.is_binary()) func(e.binary.src1_desc);
    }
}

bool has_large_buffers(const pd_t *pd) {
    auto is_large = [](const memory_desc_t &md) {
        memory_desc_wrapper mdw(md);
        gpu_assert(!mdw.format_any());
        return mdw.size() > size_t(std::numeric_limits<int32_t>::max());
    };
    bool has = false;
    iter_md(pd, [&](const memory_desc_t &md) {
        if (is_large(md)) has = true;
    });
    return has;
}

bool has_shifted_mds(const pd_t *pd) {
    bool has = false;
    iter_md(pd, [&](const memory_desc_t &md) {
        if (md.offset0 != 0) has = true;
    });
    return has;
}

class gen_t {
public:
    template <typename T>
    static bool is_supported(T *pd, prop_kind_t prop) {
        // Non-trivial strides produces non-linear offset arithmetic which is
        // not yet supported.
        if (pd->is_bwd_d()
                && !(pd->KSW() == 1 && pd->KSH() == 1 && pd->KSD() == 1))
            return false;
        // Mixed types are not supported for backward by data.
        if (pd->is_bwd_d()
                && pd->diff_dst_md()->data_type
                        != pd->diff_src_md()->data_type) {
            return false;
        }
        using sm = primitive_attr_t::skip_mask_t;
        auto skip_mask
                = sm::post_ops | sm::sum_dt | sm::scales | sm::scales_data_type;
        if (!pd->attr()->has_default_values(skip_mask)) return false;
        return true;
    }

    template <typename T>
    static status_t init_pd(T *pd, impl::engine_t *engine, prop_kind_t prop) {
        if (!is_supported(pd, prop)) return status::unimplemented;

        using intel::engine_t;
        auto *intel_engine = utils::downcast<engine_t *>(engine);
        if (!intel_engine->mayiuse_ngen_kernels()) return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;

        CHECK(init_default_layouts(pd));

        auto prb = to_problem(pd, engine);
        kernel_desc_t _desc;
        if (debug_t::init_kernel_desc(_desc)) {
            _desc.set_missing();
            _desc.spec.mode = specialization_mode_t::_default;
        } else {
            auto &registry = const_plan_registry();
            _desc = registry.find_best(prb, specialization_mode_t::_default);
            if (_desc.is_empty()) {
                gpu_info() << "Cannot find kernels that can fit the problem.";
                return status::unimplemented;
            }
        }
        _desc.fit_to(prb);
        CHECK(init_layouts(_desc, pd));
        CHECK(pd->attr_.set_default_formats(out_md(pd)));

        // Large buffer support is unimplemented.
        if (has_large_buffers(pd)) return status::unimplemented;
        // Shifted memory descriptors are not supported.
        if (has_shifted_mds(pd)) return status::unimplemented;

        CHECK(_desc.set_attr(pd, pd->attr(), out_md(pd)));
        if (!create_plan(_desc, prb)) {
            gpu_info() << "Cannot create kernel descriptor.\n";
            return status::runtime_error;
        }
        pd->init_plan = std::make_shared<primitive_init_plan_t>();
        CHECK(_desc.init_primitive_plan(*pd->init_plan, prb, pd));
        return status::success;
    }

    gen_t() = default;

    template <typename T>
    status_t init(T *primitive, impl::engine_t *engine) {
        auto &init_plan = *primitive->pd()->init_plan;
        CHECK(init_plan.create_exec_plan(exec_plan_, primitive, engine));
        return status::success;
    }

    status_t execute(
            const primitive_t *primitive, const exec_ctx_t &ctx) const {
        return exec_plan_.execute(primitive, ctx);
    }

private:
    static const memory_desc_t *out_md(const pd_t *pd) {
        if (pd->is_fwd()) return pd->dst_md();
        if (pd->is_bwd_d()) return pd->diff_src_md();
        if (pd->is_bwd_w()) return pd->diff_weights_md();
        gpu_error_not_expected();
        return nullptr;
    }

    primitive_exec_plan_t exec_plan_;
};

status_t gen_fwd_t::pd_t::init(impl::engine_t *engine) {
    return gen_t::init_pd(this, engine, prop_kind::forward);
}

status_t gen_fwd_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_bwd_data_t::pd_t::init(impl::engine_t *engine) {
    return gen_t::init_pd(this, engine, prop_kind::backward_data);
}

status_t gen_bwd_data_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_bwd_weights_t::pd_t::init(impl::engine_t *engine) {
    return gen_t::init_pd(this, engine, prop_kind::backward_weights);
}

status_t gen_bwd_weights_t::init(impl::engine_t *engine) {
    impl_ = std::make_shared<gen_t>();
    return impl_->init(this, engine);
}

status_t gen_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace v2
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
