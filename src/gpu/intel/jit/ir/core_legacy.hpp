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

#ifndef GPU_INTEL_JIT_IR_CORE_HPP
#define GPU_INTEL_JIT_IR_CORE_HPP

#include "gemmstone/../../dsl/ir/core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Replace with using dsl = gemmstone::dsl once migration is complete
namespace dsl {
namespace type {
using attr_t = gemmstone::dsl::type::attr_t;
}
using type_t = gemmstone::dsl::type_t;
} // namespace dsl

namespace ir = gemmstone::dsl::ir;

template <typename KeyT>
using object_set_t = gemmstone::dsl::ir::object_set_t<KeyT>;
template <typename KeyT>
using object_eq_set_t = gemmstone::dsl::ir::object_eq_set_t<KeyT>;
template <typename KeyT, typename ValueT>
using object_map_t = gemmstone::dsl::ir::object_map_t<KeyT, ValueT>;
template <typename KeyT, typename ValueT>
using object_eq_map_t = gemmstone::dsl::ir::object_eq_map_t<KeyT, ValueT>;

using ir_mutator_t = gemmstone::dsl::ir::ir_mutator_t;
using ir_visitor_t = gemmstone::dsl::ir::ir_visitor_t;

namespace object {
template <typename T>
using info_t = gemmstone::dsl::ir::object::info_t<T>;
using impl_t = gemmstone::dsl::ir::object::impl_t;
} // namespace object

using object_t = gemmstone::dsl::ir::object_t;
using expr_t = gemmstone::dsl::ir::expr_t;
using stmt_t = gemmstone::dsl::ir::stmt_t;

using op_kind_t = gemmstone::dsl::ir::op_kind_t;
using binary_op_t = gemmstone::dsl::ir::binary_op_t;
using bool_imm_t = gemmstone::dsl::ir::bool_imm_t;
using cast_t = gemmstone::dsl::ir::cast_t;
using const_var_t = gemmstone::dsl::ir::const_var_t;
using float_imm_t = gemmstone::dsl::ir::float_imm_t;
using int_imm_t = gemmstone::dsl::ir::int_imm_t;
using iif_t = gemmstone::dsl::ir::iif_t;
using linear_t = gemmstone::dsl::ir::linear_t;
using load_t = gemmstone::dsl::ir::load_t;
using ptr_t = gemmstone::dsl::ir::ptr_t;
using shuffle_t = gemmstone::dsl::ir::shuffle_t;
using ternary_op_t = gemmstone::dsl::ir::ternary_op_t;
using unary_op_t = gemmstone::dsl::ir::unary_op_t;
using var_t = gemmstone::dsl::ir::var_t;
using ref_t = gemmstone::dsl::ir::ref_t;
using var_t = gemmstone::dsl::ir::var_t;

using alloc_kind_t = gemmstone::dsl::ir::alloc_kind_t;
using alloc_attr_impl_t = gemmstone::dsl::ir::alloc_attr_impl_t;
using alloc_attr_t = gemmstone::dsl::ir::alloc_attr_t;
using bank_conflict_attr_t = gemmstone::dsl::ir::bank_conflict_attr_t;
using alloc_t = gemmstone::dsl::ir::alloc_t;
using assign_t = gemmstone::dsl::ir::assign_t;
using store_t = gemmstone::dsl::ir::store_t;
using for_t = gemmstone::dsl::ir::for_t;
using if_t = gemmstone::dsl::ir::if_t;
using let_t = gemmstone::dsl::ir::let_t;
using stmt_label_t = gemmstone::dsl::ir::stmt_label_t;
using stmt_group_t = gemmstone::dsl::ir::stmt_group_t;
using stmt_seq_t = gemmstone::dsl::ir::stmt_seq_t;
using while_t = gemmstone::dsl::ir::while_t;
using func_call_attr_t = gemmstone::dsl::ir::func_call_attr_t;
using instruction_modifier_attr_t
        = gemmstone::dsl::ir::instruction_modifier_attr_t;
using func_impl_t = gemmstone::dsl::ir::func_impl_t;
using func_t = gemmstone::dsl::ir::func_t;
using func_call_t = gemmstone::dsl::ir::func_call_t;
using builtin_t = gemmstone::dsl::ir::builtin_t;

using gemmstone::dsl::ir::is_const;
using gemmstone::dsl::ir::is_cpp;
using gemmstone::dsl::ir::to_cpp;
using gemmstone::dsl::ir::to_expr;

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
