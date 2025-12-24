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

#include "gemmstone/../../dsl/ir/codegen/extensions.hpp"
#include "gemmstone/../../dsl/ir/codegen/reorder.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

template <typename ngen_generator_t>
void eltwise(ngen_generator_t &host, ir::ngen_register_scope_t &scope,
        const dsl::hw_t &hw, const eltwise_t &func,
        const std::vector<ir::ngen_operand_t> &args) {
    int elems = to_cpp<int>(hw, eltwise_t::arg_elems(args));
    auto &data_op = eltwise_t::arg_data(args);
    const auto &data_rd = data_op.reg_buf_data();

    eltwise_injector_f32_t<typename ngen_generator_t::RootCodeGenerator> inj(
            &host, func.alg_kind, func.alpha, func.beta, func.scale);
    auto scratch = scope.alloc_range(inj.preferred_scratch_regs());
    inj.set_scratch(scratch);
    inj.prepare();

    int grf_size = ngen::GRF::bytes(hw);
    int f_size = sizeof(float);
    int step = 2 * grf_size / f_size;

    auto do_eltwise = [&](const ir::reg_buf_data_t &r, const int count) {
        if (func.alg_kind == alg_kind::eltwise_stochastic_round) {
            gpu_assert(args.size() == 3);
            const auto &seed = args[2].reg_buf_data();
            inj.compute(ngen::GRFRange(r.base(), count),
                    seed.reg_data().getBase(), seed.reg_data().getOffset(),
                    func.dst_dt);
        } else if (func.alg_kind == alg_kind::eltwise_mx_scale) {
            gpu_assert(args.size() == 3);
            const auto &scales_dst = args[2].reg_buf_data();
            inj.compute(ngen::GRFRange(r.base(), count),
                    scales_dst.reg_data().getBase(),
                    scales_dst.reg_data().getOffset(), func.dst_dt);
        } else {
            inj.compute(ngen::GRFRange(r.base(), count));
        }
    };
    for (int i = 0; i < elems; i += step) {
        ir::ngen_register_scope_t i_scope(scope.register_allocator());
        step = std::min(step, elems - i);
        step = utils::rnd_down_pow2(step);
        int cur_elems = step;
        auto rd = data_rd.format(i, ngen::DataType::f);
        // Use temporary storage when needed to ensure:
        // - Eltwise is applied to full register
        // - Data is aligned to GRF boundary
        if ((cur_elems * f_size) % grf_size != 0 || rd.byte_offset() != 0) {
            int full_elems
                    = utils::rnd_up(cur_elems * f_size, grf_size) / f_size;
            auto tmp = i_scope.alloc_reg_data(dsl::type_t::f32(full_elems));
            emit_reorder_1d_tile(&host, hw, i_scope, cur_elems, rd, 1, tmp, 1);
            do_eltwise(tmp, full_elems * f_size / grf_size);
            emit_reorder_1d_tile(&host, hw, i_scope, cur_elems, tmp, 1, rd, 1);
        } else {
            do_eltwise(rd, cur_elems * f_size / grf_size);
        }
    }
}

template <typename ngen_generator_t>
void handler(ngen_generator_t &host, const object_t &obj,
        ir::codegen_extension_iface_t &ext_iface) {
    auto &options = ext_iface.options();
    if (obj.is<func_call_t>()) {
        auto &call = obj.as<func_call_t>();
        if (call.func.is<eltwise_t>()) {
            ir::ngen_register_scope_t scope(ext_iface.allocator());
            auto args = ext_iface.evaluate(call.args, scope);
            eltwise(host, scope, options.hw(), call.func.as<eltwise_t>(), args);
            return;
        }
    }

    gpu_error_not_expected() << "Unknown object " << obj;
}

void extension_handler(
        const object_t &obj, ir::codegen_extension_iface_t &ext_iface) {
    auto host = ext_iface.root_code_generator();

#ifdef NGEN_ASM
    if (host.info == typeid(ngen::AsmCodeGenerator))
        return handler(*static_cast<ngen::AsmCodeGenerator *>(host.ptr), obj,
                ext_iface);
#endif

#define GPU_HW_CASE(hw) \
    return handler(*static_cast<ngen::BinaryCodeGenerator<(hw)> *>(host.ptr), \
            obj, ext_iface);

    GPU_HW_SWITCH(host.hw);
#undef GPU_HW_CASE

    gpu_error_not_expected() << "Unknown nGEN code generator";
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace gemmstone {
namespace dsl {
namespace kernel {
codegen_extension_handler_t default_extension_handler
        = &dnnl::impl::gpu::intel::jit::extension_handler;
}
} // namespace dsl
} // namespace gemmstone
