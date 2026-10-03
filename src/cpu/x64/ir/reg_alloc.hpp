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

#ifndef CPU_X64_IR_REG_ALLOC_HPP
#define CPU_X64_IR_REG_ALLOC_HPP

#include <cstdint>
#include <vector>

#include "common/utils.hpp"

#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Describes where a virtual register was assigned by the allocator.
// The location is determined by `spilled`:
//
// - If `spilled == false`, the value is stored in the physical register
//   `phys`. The number is the register index within its group
//   (for example: gpr 0=rax, ..., 15=r15 and vec 0=ymm0, ..., 15=ymm15).
//   In this case, `slot` is not used.
//
// - If `spilled == true`, the value is stored on the stack at byte offset
//   `slot` in the spill area. In this case, `phys` is not used.
//   Whenever the value is needed, the emitter loads it into a scratch
//   register before using it.
struct assignment_t {
    bool spilled = false;
    int phys = -1;
    size_t slot = 0;
};

// The final allocation result.
//
// - `assignments` contains one `assignment_t` for each virtual register,
//   indexed by virtual register id.
// - `frame_bytes` is the total amount of stack space needed for spilled
//   values. The kernel reserves this space with a single `sub rsp`.
// - `any_spill` is true if any virtual register was spilled to the stack.
struct reg_alloc_result_t {
    std::vector<assignment_t> assignments;
    size_t frame_bytes = 0;
    bool any_spill = false;
};

// A register file contains physical registers the allocator may assign, and the
// stack-slot size used when a spill is needed.
//
// `regs` holds the register indices available for allocation (for example, all
// general-purpose registers except reserved ones such as `rsp`, the argument
// pointer, and scratch registers).
//
// `slot_size` is how many bytes a spilled value needs on the stack
// (8 for a GPR, 32 for a YMM, 64 for a ZMM).
struct reg_file_t {
    std::vector<int> regs;
    size_t slot_size = 0;
};

// The register files plus a map from each register kind to the file it
// allocates from. Two kinds may share one file, e.g. on AVX2* a mask is a
// vector register, so `vec` and `mask` kinds map to the same file and compete
// for the same pool. On AVX-512 a mask is a k-register and has its own file.
//
// `kind_to_file` is indexed by `(int)reg_kind_t`.
//
// This structure is created and filled by make_reg_config() based on the target
// ISA and later used by allocate_registers() during allocation.
struct reg_pools_t {
    std::vector<reg_file_t> files;
    std::vector<int> kind_to_file;
};

// Statistics of one allocate_registers() call, for the debug output (see
// `ir/dump.hpp`). The allocator computes these values during the allocation,
// and it keeps them only when the caller asks for them.
//
//   start, end      - live interval [start, end] of each virtual register,
//                     indexed by virtual register id. `end < 0` means that no
//                     operation references the virtual register.
//   weight          - spill weight of each virtual register, as the scan uses
//                     it
//   loop_depth      - loop nesting depth of each operation, as the spill
//                     weights use it
//   pressure        - [file][op]: the number of virtual registers of the file
//                     that are live on entry to the operation or that the
//                     operation writes. The file needs that many registers at
//                     the operation.
//   overlap         - [file][op]: the number of virtual registers of the file
//                     whose interval contains the operation. The scan spills
//                     in a file if and only if the largest overlap is larger
//                     than the pool of the file.
//   liveness_passes - the number of passes of the liveness analysis, including
//                     the last pass, which changes nothing
//
// `pressure` is never larger than `overlap`. It is smaller where a virtual
// register is dead inside its interval.
struct reg_alloc_stats_t {
    std::vector<int> start, end;
    std::vector<int64_t> weight;
    std::vector<int> loop_depth;
    std::vector<std::vector<int>> pressure;
    std::vector<std::vector<int>> overlap;
    int liveness_passes = 0;
};

// Allocates registers for `ir` from `pools`. When `stats` is not null, it also
// fills `stats`. The statistics do not change the allocation.
//
// Export for testing.
reg_alloc_result_t DNNL_API allocate_registers(const ir_t &ir,
        const reg_pools_t &pools, reg_alloc_stats_t *stats = nullptr);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
