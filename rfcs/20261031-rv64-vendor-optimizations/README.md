# RFC: Vendor-specific ISA and microarchitecture support for RV64

## Introduction

We would like to give contributors a clear way to bring RISC-V optimizations
into oneDNN. The design follows the discussion in
[#6320](https://github.com/uxlfoundation/oneDNN/issues/6320).

We propose two routes:

- **Route 1: Native JIT integration.** Add instruction support to Xbyak_riscv,
  then implement optimized kernels in oneDNN. This is our preferred long-term
  route because multiple primitives can share the same kernel code.
- **Route 2: Vendor-library integration.** Connect an existing vendor library
  to oneDNN through an adapter. Each new library needs its own RFC and must be
  introduced as a Git submodule.

SpacemiT K3 is a useful example of why this matters. Its X100 cores have
256-bit vectors, while its A100 cores have 1024-bit vectors and specialized
matrix and dot-product instructions. In our tests, longer vectors alone did
not always make A100 faster. Kernels also need suitable blocking, register
usage, memory access, and scheduling.

We want applications to benefit from such work through the existing oneDNN
primitives. Support for standard RISC-V extensions and selected vendor
extensions can develop in parallel.

## Proposal

### Route 1: Native JIT integration

A contribution taking this route should complete these steps in order:

1. **Make the hardware detectable through upstream Linux.** Linux must have
   merged the support needed to identify the target CPUs with
   `riscv_hwprobe()`. At minimum, we need to identify the microarchitecture.
   We recommend directly reporting the supported instruction extensions.
2. **Submit the Xbyak_riscv changes.** These include instruction encodings,
   operand checks, detection support, and tests. If Linux only reports CPU
   identifiers, propose an identifier-to-extension lookup table there first.
3. **Wait for those changes to merge, then update oneDNN's Xbyak_riscv
   version.** oneDNN can then use the accepted encodings and detection APIs.
4. **Submit the optimized oneDNN kernels.** Include correctness results,
   performance measurements and supported configurations.

Direct ISA reporting can take time to reach upstream Linux and deployed
systems. We therefore accept CPU identification plus a lookup table as a
starting point. We will first ask Xbyak_riscv to maintain that table. If they
decline, we can put it in oneDNN's RV64 capability-detection code.

#### What does “detectable” mean?

There are two ways to obtain the information we need.

**Example A: Linux tells us which CPU it is; a table tells us what it supports.**

Our K3 measurements in #6320 returned these architecture identifiers:

| CPU group on the tested machine | `MARCHID` returned by Linux | Core type |
| --- | --- | --- |
| CPUs 0-7 | `0x8000000058000002` | X100 |
| CPUs 8-15 | `0x8000000041000002` | A100 |

Suppose we want to run an IME kernel on A100. We ask `riscv_hwprobe()` for the
vendor, architecture, and implementation identifiers of the selected CPUs:
`MVENDORID`, `MARCHID`, and `MIMPID`. We then look up those identifiers in a
table based on the vendor's documented specifications.

```text
Query the selected CPUs with riscv_hwprobe()
    -> identify an A100 implementation
Look up that implementation in the vendor table
    -> find the supported extensions and their versions
Select a oneDNN kernel that uses those extensions
```

In this example, Linux answers “which CPU?”, and the table answers “which
instructions?”. Each table entry documents the mapping, including CPU
revisions, OS requirements, and its specification references.

**Example B: Linux directly tells us whether an extension is supported.**

SiFive's `Xsfvqmaccdod` extension provides an existing vendor-specific example.
Linux defines the query key `RISCV_HWPROBE_KEY_VENDOR_EXT_SIFIVE_0` and the
capability bit `RISCV_HWPROBE_VENDOR_EXT_XSFVQMACCDOD`.

We query that key for the CPUs that will execute the kernel and check the
returned bit. When it is set, Linux reports that those CPUs support the
extension:

```text
hwprobe(selected CPU set, RISCV_HWPROBE_KEY_VENDOR_EXT_SIFIVE_0)
    -> value contains RISCV_HWPROBE_VENDOR_EXT_XSFVQMACCDOD
The extension is available on the queried CPU set
    -> a kernel using Xsfvqmaccdod is eligible for ISA-based selection
```

The key, bit, and detection implementation are already present in upstream
Linux: see the [hwprobe keys](https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/hwprobe.h),
[SiFive capability bits](https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/vendor/sifive.h),
and [SiFive hwprobe implementation](https://github.com/torvalds/linux/blob/master/arch/riscv/kernel/vendor_extensions/sifive_hwprobe.c).

This is the model we recommend for other vendors.
With direct reporting, a new CPU can advertise the same extension without
needing another model entry in our table.

**Both examples use `riscv_hwprobe()`. We accept Example A as the minimum and
recommend Example B.** When direct reporting is available, it takes priority.
An explicit unsupported result must not be overridden by the lookup table.
If neither method confirms support, oneDNN uses its existing fallback.
Extensions that add OS-managed state also need the corresponding OS support.

The [Linux hwprobe documentation](https://docs.kernel.org/arch/riscv/hwprobe.html)
explains CPU identifiers, extension bits, and CPU-set queries.

#### Where should the work live?

| Work | Preferred location |
| --- | --- |
| Instruction encodings and operand checks | Xbyak_riscv |
| CPU-set detection and vendor lookup table | Xbyak_riscv, proposed there first |
| Lookup table if Xbyak_riscv declines it | Centralized oneDNN RV64 detection code |
| Primitive selection and workload thresholds | oneDNN |
| Blocking and scheduling | oneDNN or a shared kernel-generation library |

If oneDNN owns the table, primitives should use the shared detection result.
We can move entries to Xbyak_riscv or direct kernel reporting as support
becomes available.

A reusable GEMM/BRGEMM kernel is a useful first target because it can benefit
matmul, convolution, and inner product. The same CPU information can help tune
standard RVV kernels, including LMUL, packing, and blocking choices.

### Route 2: Vendor-library integration

A vendor may already have an optimized operator library. This route connects
it to oneDNN through an adapter and can make those kernels available sooner.

**Each new library needs its own approved RFC before integration.** The RFC
should cover the supported operators and configurations, performance results,
binary size, licensing, threading, hardware adoption, and maintenance. It
should also explain hardware detection and any OS or resource requirements,
including how the library handles heterogeneous CPUs.

**Use a Git submodule pinned to a reviewed commit.** Users initialize the
selected submodule when enabling the library's optional build configuration.
Library updates change that recorded commit through code review. Document
both submodule initialization and builds from source packages.

The adapter translates oneDNN descriptors and arguments into library calls.
It joins the existing primitive-selection process and checks that the library
supports the requested operation and execution CPU set. Other configurations
continue through oneDNN's fallback implementations.

