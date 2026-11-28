/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_STORE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_STORE_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace memory {

/// Scalar store expression
struct store_t : public rvjit::memory::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::memory";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit store_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op.data` has a scalar store width
    ///
    /// @param op The operation checked
    bool is_valid(const rvjit::memory::op_t &op) const override {
        if (!op.data.is_defined()) return false;
        const int w = op.data.width;
        return op.data.is_int() ? (w == 1 || w == 2 || w == 4 || w == 8)
                                : (w == 2 || w == 4 || w == 8);
    }

    /// General scalar memory store
    ///
    /// @param op The operation, in the scalar form
    /// @param args The data register and the address
    ///
    /// @details Emits an integer or float store based on `op.data`
    void emit(const rvjit::memory::op_t &op,
            const rvjit::memory::args_t &args) const override {
        CHECK_OR_RET(op.data.is_valid() && args.data != no_name, ,
                "invalid operand descriptor");
        CHECK_OR_RET(store_t::is_valid(op), , "unsupported datatype width");
        if (op.data.is_int())
            xstore(reg_t(args.data), args.addr, op.data.width);
        else
            fstore(freg_t(args.data), args.addr, op.data.width);
    }

    /// General scalar float store
    ///
    /// @param fs The source register
    /// @param addr The memory address stored to
    /// @param width The access size in bytes
    ///
    /// @details Emits fsh, fsw, or fsd based on `width`
    ///
    /// @note Emits nothing if `width` has no supported float store
    ///
    /// @pre `width` must be 2, 4, or 8 (bytes)
    void fstore(const FReg &fs, const rvjit::pointer_t &addr, int width) const {
        switch (width) {
            case 2: e_->fsh(fs, addr.base, addr.offset); break;
            case 4: e_->fsw(fs, addr.base, addr.offset); break;
            case 8: e_->fsd(fs, addr.base, addr.offset); break;
            default: LOG_FAIL("store", "unsupported datatype width"); break;
        }
    }

    /// General scalar integer store
    ///
    /// @param rs The source register
    /// @param addr The memory address stored to
    /// @param width The access size in bytes
    ///
    /// @details Emits sb, sh, sw, or sd based on `width`
    ///
    /// @note Emits nothing if `width` is unsupported
    ///
    /// @pre `width` must be 1, 2, 4, or 8 (bytes)
    void xstore(const Reg &rs, const rvjit::pointer_t &addr, int width) const {
        switch (width) {
            case 1: e_->sb(rs, addr.base, addr.offset); break;
            case 2: e_->sh(rs, addr.base, addr.offset); break;
            case 4: e_->sw(rs, addr.base, addr.offset); break;
            case 8: e_->sd(rs, addr.base, addr.offset); break;
            default: LOG_FAIL("store", "unsupported datatype width"); break;
        }
    }

protected:
    emitter_t e_;
};

} // namespace memory
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
