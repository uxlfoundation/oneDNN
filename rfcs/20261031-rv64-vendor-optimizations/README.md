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

#### Heterogeneous CPU support

We separate three responsibilities: the OS reports usable hardware features,
Xbyak_riscv detects them, and the application or threading runtime controls
where workers run. oneDNN uses that information to select and tune kernels.

This follows the division already visible in other oneDNN backends:

- **x64:** ISA selection uses cached Xbyak CPU information. Hybrid-core topology
  is also available, and oneDNN uses it to choose conservative cache sizes
  across P/E core types. See [ISA detection](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/x64/cpu_isa_traits.hpp)
  and [hybrid cache handling](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/x64/platform.cpp).
- **AArch64:** the native JIT backend uses HWCAP information and caches the SVE
  length obtained through `PR_SVE_GET_VL`. Linux exposes system-safe feature
  information and manages SVE vector length as thread state. This differs from
  reading the physical RVV VLEN on whichever core runs the detection code.
  See [AArch64 detection](https://github.com/uxlfoundation/oneDNN/blob/cc575246d0682887b1bc461588836f1466cca826/src/cpu/aarch64/cpu_isa_traits.hpp),
  [Linux feature reporting](https://docs.kernel.org/arch/arm64/cpu-feature-registers.html),
  and [SVE thread state](https://docs.kernel.org/arch/arm64/sve.html).

For RV64, we use the following execution policy.

**Default execution uses common capabilities.** On platforms where the OS
allows workers to run across different core types, detection covers every CPU on
which the workload may run. If the runtime cannot establish a narrower set,
use all online CPUs. Vendor extensions can be enabled only when supported
throughout that set. For a mixed-microarchitecture set, use generic tuning.
RVV execution requires OS support for vector state and thread migration.
Enable fixed-VLEN kernels when the execution set has a known, compatible VLEN;
otherwise select a compatible fallback.

**Specialized execution uses a fixed, homogeneous CPU group.** The application
or launcher establishes the group before oneDNN capability initialization and
before creating OpenMP or thread-pool workers. The application must keep every
worker and later primitive execution within that group. Platform-specific
steps, are performed by the launcher or runtime.
A process-wide cpuset restriction is one way to enforce the CPU boundary.

The initial implementation supports one fixed execution group per process.
The backend records its CPU set and capabilities once and uses them for
primitive creation, JIT generation, and execution. Keep that set fixed for the
lifetime of the generated kernels.
To use X100 and A100 concurrently in this initial design, applications use
separate processes, each configured for its core group.

#### Changes required in Xbyak_riscv

The current [Xbyak_riscv detector](https://github.com/herumi/xbyak_riscv/blob/main/xbyak_riscv/xbyak_riscv_util.hpp)
queries all online CPUs with `cpusetsize=0` and `cpus=NULL`, then reads `vlenb`
on the calling CPU. We need an explicit CPU-set query so both results describe
the same target CPUs.

1. **Accept the target CPU set.** Add an explicit-set detection API alongside
   the existing API. oneDNN supplies its established execution set.
2. **Query and group CPUs.** Read vendor, architecture, implementation, and
   extension information per CPU. Group matching results and return the
   common extension set as well. `RISCV_HWPROBE_WHICH_CPUS` can accelerate
   matching; intersect its results with the requested set. Per-CPU queries
   provide a fallback when that flag is unavailable.
3. **Resolve vendor capabilities.** Prefer direct hwprobe extension reporting,
   then use the documented identity table where needed. Keep an unknown
   identity distinct from a confirmed unsupported capability.
4. **Obtain VLEN for the selected group.** Prefer an OS interface where one is
   available. Otherwise use a dedicated probe thread bound to a target CPU
   before its first vector instruction. Check vector availability first and
   create fresh probe threads for other CPUs rather than migrating live vector
   state across unlike cores. The launcher must already have completed any
   required core enrollment. Failed probes leave VLEN unknown.
5. **Return an immutable detection result.** Include the queried CPU set,
   identifiers, extensions, VLEN information, and the source of the capability
   information. Each explicit query returns a separate object.

oneDNN uses this result during initialization and kernel selection. Target
properties that affect generated code, including VLEN, must be represented in
kernel-cache keys or isolated by the fixed execution configuration. Capability
checks stay outside the kernel's hot loops. Tests must cover restricted CPU
sets, mixed groups, unknown VLEN, and rejection of incompatible reuse.

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

