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

#ifndef CPU_X64_IR_DUMP_HPP
#define CPU_X64_IR_DUMP_HPP

// Debug output of the IR pipeline.
//
// A kernel built from the IR calls `print_kernel_dump()` at the end of its
// `generate()`. The call prints a text description of the kernel. It prints
// only in dev-mode builds, and only when `ONEDNN_VERBOSE` holds
// `x64ir=<level>`. Each level prints everything that the lower levels print,
// and adds more:
//   1 - summary: kernel name, ISA, IR counts, and code size
//   2 - nothing beyond level 1 yet
//   3 - the IR dump, one line per operation
//
// The output uses only what the shared pipeline knows: the generator, the IR,
// and the static data. It does not depend on a particular builder.
//
// The level does not depend on `debuginfo=` or on `all`. `all` enables the
// standard verbose output, and this backend-specific dump should not mix with
// it.
//
// The output for one kernel is formatted into one string and printed with a
// single stdio call, so the output of kernels created at the same time does
// not mix. It starts and ends with a marker line:
//
//   begin x64ir #<seq> <kernel name> isa=<isa>
//   ...
//   end x64ir #<seq>
//
// The formatting functions are compiled in every build, so the unit tests run
// without dev mode. Only `verbose_level()` depends on dev mode.

#include <string>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Returns the level of the last `x64ir=<level>` token in `verbose_value`, a
// comma-separated `ONEDNN_VERBOSE` value. Returns 0 when there is no such
// token. The level is parsed with `atoi()`, so non-numeric text reads as 0.
// Negative levels are clamped to 0.
//
// Export for testing.
int DNNL_API parse_x64ir_level(const std::string &verbose_value);

// Returns the level set by `ONEDNN_VERBOSE`. The variable is read once per
// process. Returns 0 in builds without dev mode.
int verbose_level();

// Kernel facts that the IR does not hold. `print_kernel_dump()` takes them
// from the generator and the static data.
//
//   name      - kernel name, as returned by `name()`
//   isa       - ISA the kernel is generated for (`max_cpu_isa()`)
//   code_size - total code size in bytes at the end of `generate()`,
//               including static data
//   data_size - static data in bytes, written after the postamble (see
//               `data_section_t`)
struct kernel_info_t {
    const char *name = "";
    cpu_isa_t isa = isa_undef;
    size_t code_size = 0;
    size_t data_size = 0;
};

// Returns the IR dump of `ir`, one line per operation. Each line starts with
// the operation index and is indented by loop depth.
//
// Operands print as follows:
//   r<id>, m<id>     - gpr and mask vreg with id `<id>`
//   <dt>:v<id>       - vec vreg with id `<id>` that holds data type `<dt>`
//   [r<id>+<disp>]   - memory operand with a decimal byte offset.
//                      `[param+<disp>]` reads the kernel argument struct. A
//                      vector access is prefixed with the data type in memory
//                      (`bf16:[r3+0]`).
//   L<id>            - IR label with id `<id>`
//
// Export for testing.
std::string DNNL_API to_string(const ir_t &ir);

// Returns the output for one kernel at `level`, framed with sequence number
// `seq`.
//
// Export for testing.
std::string DNNL_API format_kernel_dump(
        int level, int seq, const kernel_info_t &info, const ir_t &ir);

// Prints the output for the kernel that `gen` generated from `ir`, with the
// static data in `data`. Prints nothing when `verbose_level()` is 0.
//
// The kernel calls it at the end of `generate()`, after the static data is
// written, so the code size includes the static data.
void print_kernel_dump(
        const jit_generator_t &gen, const ir_t &ir, const data_section_t &data);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
