# Improving Functional Validation Coverage

## Introduction

oneDNN functional validation is built on benchdnn input files. Each driver
has a set of hand-written files that iterate over options, such as data types,
attributes, and memory formats, for a fixed set of shapes. The files are
combined into test launchers for the smoke, CI, nightly, and weekly test sets,
often split by backend and by feature.

This RFC proposes to rebuild the content of functional validation and the way
it is updated. The core coverage of each driver becomes a set of test cases
produced by a test case generator, sized to fit a time budget per test set.
To make generated coverage useful, the RFC also proposes to reduce the skip
logic in benchdnn: missing functionality is implemented in reference
implementations, and data types that are not supported on the system are
reported by the library with a new status.

## Motivation

The request comes from both external and internal parties that rely on oneDNN
quality. The library gains new functionality quickly, and the coverage must keep
up with it to prevent bug escapes.

Validation exists to prove that a change is safe to promote into the
repository. This RFC focuses on functional validation. The more complete the
coverage is, the higher the chance that a change is safe. However, the answer
must come in reasonable time: a process that takes a day to verify a change
gives less value to development than one that responds quickly. More test
cases mean more testing time, so the goal is to reach the best coverage
possible within a fixed time budget.

The goals are:

* **Time budget** per testing job, assuming unlimited hardware resources:
  * CI: 2 hours.
  * Nightly: 8 hours.
* **Coverage of the most useful API combinations**, the ones customers rely on
  day to day. The coverage should be expressed in a generalized way, not tied
  to specific shapes or a fixed set of API parameters.
* **Coverage of corner cases**, to make sure that leaving the main path
  doesn't lead to undesired behavior.
* **A high hit rate of optimized implementations**, around 80% of cases.
  Reference implementations bring little value to customers, so testing them
  should not dominate the time budget.

The current approach doesn't scale towards these goals:

* Iterating over options in input files multiplies the number of cases with
  every new option, so options are tested one at a time, with default values
  for the rest. Combinations of non-default values are rarely tested.
* Multiple test launchers per driver, split by backend or feature, lead to
  repeated validation, more load on the testing infrastructure, and little
  transparency about what is validated.
* Skip logic in benchdnn hides the reasons why cases don't run. This is
  covered in [Skipped Cases](#skipped-cases).

## Proposal

### Generated Coverage

A single test case may combine many options with non-default values, so a
generated case covers more functionality than a hand-written one. Given the
size of the combination space, the proposal is to sample it randomly. Broad
sampling covers features that are usually independent implementation-wise
together, instead of validating one feature at a time.

The generator produces a list of single benchdnn test cases, with no
iteration over options. It is expected to:

* **Follow the library API.** It produces only valid and supported
  combinations, not arbitrary ones. For example, quantization parameters are
  applied to integer tensors, fp8 tensors, or in the weights-only
  quantization mode, and masks follow the dimensions used and reflect
  real-world applications.
* **Weigh common cases against corner cases.** A probability mechanism makes
  common functionality appear more often than corner cases.
* **Control value ranges**, such as shapes and memory formats, to reach the
  desired hit rate of optimized implementations.
* **Scale across drivers.** Modules are reused across drivers, and patterns
  express more with less. For example, `x4` stands for `s4` or `u4`, `xf16`
  stands for `bf16` or `f16`, use cases are combined into policies such as
  MXFP8 quantization, and `*` stands for any value.

An example of the generated coverage is
[harness_matmul_generated_ci](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/inputs/matmul/harness_matmul_generated_ci).
Each list starts with the generator options used, so it can be reproduced.

### Coverage Layout

Each test set contains the following:

* **Smoke**: the smallest set, used in public CI and in simulation
  validation. It contains about 250 small generated cases (up to 1e6 ops for
  compute primitives). It covers the same features as CI on a smaller scale.
* **CI**: the standard set for commit-to-commit development. It contains:
  * The smoke set, to verify that the public set is functionally correct.
  * About 5,000 medium-sized generated cases (up to 1e9 ops for compute
    primitives). All features with an optimized path in both CPU and GPU
    backends are covered.
  * The regression list, usually issues reported externally.
  * Cases that the generator doesn't support yet.
* **Nightly**: the largest set for day-to-day development. It contains:
  * The CI set, so that CI procedures don't break while the nightly set
    passes.
  * About 25,000 medium and large generated cases (up to 1e10 ops for compute
    primitives).
  * Features relevant to a single backend, such as a data type combination
    or a device feature.
  * Rarely used features, such as non-strict accumulation modes or
    precomputed reductions.
* **Weekly**: a set for week-to-week development. It contains:
  * The nightly set.
  * Problems obtained from external parties through verbose logs, known as
    "fwks" lists.
  * Extremely large cases, to verify integer overflows.

Problem sizes of compute-bound primitives are limited by the number of
operations, as listed above, so the memory used by a case may exceed 1 GB.
Problem sizes of memory-bound primitives are limited to 1 GB of memory. The
rough proportion of smoke, CI, and nightly generated cases is
1:20:100. The exact numbers depend on the importance of the primitive and on
its testing time.

Lists are named `harness_<driver>_generated_<set>`, where `<driver>` is
`matmul`, `reorder`, and so on, and `<set>` is `smoke`, `ci`, `nightly`, or
`large`. Each driver has a single launcher per test set, named
`test_<driver>_<set>`, that includes the launcher of the previous set. Launchers
are no longer split by backend: the coverage is shared.

Filtering by hardware, such as running a case only on some platforms, must not
be expressed through launchers, and neither must load balancing. Balancing can
be done by CMake, if needed, by splitting a list into chunks with a test target
for each chunk in the build directory.

The matmul driver already follows this layout:

| Set     | Launcher              | Generated cases                         |
|---------|-----------------------|-----------------------------------------|
| Smoke   | `test_matmul_smoke`   | 250, `harness_matmul_generated_smoke`   |
| CI      | `test_matmul_ci`      | ~5,000, `harness_matmul_generated_ci`   |
| Nightly | `test_matmul_nightly` | 25,000, `harness_matmul_generated_nightly` |
| Weekly  | `test_matmul_weekly`  | `harness_matmul_generated_large`        |

Grouped matmul is tested through separate `test_matmul_grouped_<set>`
launchers. Other drivers will move to the layout one by one.

### Coverage Update

Most changes in the generator lead to completely different cases, since it
relies on random sampling. Regenerating the lists frequently would be
disruptive, as each regeneration may expose new bugs that block development.
The proposal is to regenerate the lists once per release, at the beginning of
its development cycle. Regeneration is also triggered when:

* The time budget is exhausted, and jobs no longer fit the time goals.
* The scope changes in a major way, for example when the proportions of cases
  are adjusted to follow industry trends.

Between regenerations, new functionality is added without regenerating the
lists:

1. The generator is extended to cover the new functionality with the share it
   should have in the core coverage.
2. Cases are generated for the new functionality only and appended to the
   existing generated lists.

Functionality that the generator doesn't support yet is added to launchers
manually, below a dedicated note, together with a TODO to extend the
generator.

Developers are responsible for extending the generator to cover the
functionality they add, whether it is a new feature or an important set of
settings. The generator also lets developers tailor local coverage to the
feature or bug they work on.

### Tooling

The generated coverage is produced by fuzzdnn, an internal generator built
around the requirements in [Generated Coverage](#generated-coverage). It will
be published in the repository once its architecture proves stable and needs
only minor adjustments for future extensions. It becomes the primary tool for
the generated CI and nightly coverage.

Alternatives considered:

* **Extend synthdnn.** [synthdnn](https://github.com/uxlfoundation/oneDNN/tree/main/scripts/synthdnn)
  already generates matmul problems with various data types, math modes,
  memory formats, and shapes, and it collects and analyzes correctness and
  performance data.
  * Pros: an existing tool in the repository.
  * Cons: it lacks a probability mechanism, so its output is less
    controlled, and it doesn't adapt to supported scenarios, which becomes a
    problem as more features are added.

synthdnn keeps its other capabilities, such as data collection and analysis,
which fuzzdnn doesn't cover. Until fuzzdnn supports them, the extremely large
cases in `harness_matmul_generated_large` remain generated by synthdnn.

### Skipped Cases

benchdnn skips cases that are known to be unsupported: when primitive
descriptor creation returns `dnnl_unimplemented`, each driver checks whether
the case matches its list of unsupported configurations. Generated coverage
revealed several issues with this:

* The reason for a skip is hard to establish. Verbose output may report a
  generic message, or a single spot for a function with many return points.
* Each driver has lengthy logic about what is supported. Adapting the
  generator to each driver and introducing a long list of skip reasons doesn't
  scale.
* The logic may hide bugs. For example, grouped matmul with post-ops on GPU
  was skipped with a message that GPU supports only the source sparse
  descriptor.
* The library returns `dnnl_unimplemented` for valid configurations that are
  not documented as unsupported.

The proposal is to drop most of the skip logic in benchdnn:

* Missing functionality is implemented in reference implementations, so the
  library provides results for valid configurations. This is not possible for
  every part of the API, but it is for most of them. Once a skip is removed,
  `dnnl_unimplemented` is reported as a failure.
* The following exceptions remain skipped in benchdnn, as implementing them
  brings no benefit to end users:
  * **Convolution with the Winograd algorithm.** It provides an advantage on
    few platforms with low compute capabilities.
  * **Convolution with the depthwise post-op.** It is supported on older CPU
    platforms only.
  * **RNN.** Support is sparse and may exclude an algorithm, or a
    combination of an algorithm and data types, for a whole backend. RNN is
    rarely requested and outdated.
  * **Matmul with the `packed` sparse encoding.** It is supported on CPU
    with AMX for weights only.
  * **Reorder on GPU with runtime dimensions or compensation.** These are not
    supported by design.
* Unsupported data types are reported by the library, as described below.

Alternatives considered:

* **Restrict unsupported configurations in the API.** Return an error from
  the common API layer for a given backend.
  * Pros: no reference implementation work, and a clear error for users.
  * Cons: CPU and GPU behave differently for the same settings, so input
    files can't be shared between backends, which is the opposite of the
    goal.
* **Keep the skip logic with itemized reasons.** Report a dedicated reason
  for each kind of skip and collect statistics.
  * Pros: no library changes.
  * Cons: the long list of reasons and per-driver logic remain, and the logic
    may still hide bugs.

### Data Type Not Supported Status

Support for some data types depends on the hardware. Reference
implementations don't support a data type that the hardware doesn't support
natively, and primitive descriptor creation returns `dnnl_unimplemented`. To
tell such cases apart from missing functionality, benchdnn checks data type
support on its side through internal library functions exported for testing.

The proposal is a new status that reports a data type not supported on the
system:

```c
typedef enum {
    // ...
    /// The data type cannot be inferred or compiled
    dnnl_invalid_data_type = 10,
    /// The data type is not supported on the system
    dnnl_data_type_not_supported = 11,
} dnnl_status_t;
```

```cpp
namespace dnnl {
enum class status {
    // ...
    /// @copydoc dnnl_data_type_not_supported
    data_type_not_supported = dnnl_data_type_not_supported,
};
} // namespace dnnl
```

Primitive descriptor creation returns the status when no implementation is
created and at least one implementation reports an unsupported data type,
including through nested primitives. On CPU, this happens when a data type
requires instructions that the processor doesn't support, for example `bf16`
on Intel AVX2, and for data types that the library doesn't support on CPU,
such as `f64`. Reorder, concat, and sum don't depend on hardware support for
data types, so they report the status only for data types that the library
doesn't support. In the C++ API, the status is reported through a
`dnnl::error` exception.

benchdnn skips cases that return the status, without checking data type
support itself. This moves the responsibility to the library, which also gives
users a clearer reason for the failure.

Alternatives considered:

* **Reuse `dnnl_invalid_data_type`.**
  * Pros: no new status.
  * Cons: the status is used by the graph API when a data type can't be
    inferred or compiled, which is an error in the user input. A data type
    not supported on the system is a valid request that the system can't run.
* **Keep checking data type support in benchdnn.**
  * Pros: no API changes.
  * Cons: benchdnn relies on internal library functions, duplicates the
    library logic, and users don't get a clear reason for the failure.

The CPU part is implemented in
[PR #6520](https://github.com/uxlfoundation/oneDNN/pull/6520). GPU support
will follow separately.

#### Compatibility

The new status is an extension of the API and keeps binary and source
compatibility. The behavior changes for applications that check for
`dnnl_unimplemented` specifically to fall back to another path: they need to
check the new status as well. Applications that fall back on any failure, or
that use `allow_empty` primitive descriptors, are not affected.

The integrations of oneDNN in PyTorch, TensorFlow, OpenVINO, and XLA don't
check for `dnnl_unimplemented` when creating primitive descriptors, so their
behavior doesn't change. Code that handles every status value in a `switch`
without a `default` case gets a `-Wswitch` warning, which fails builds with
`-Werror`. The XLA CPU backend does this for `dnnl::graph::status`, so it
needs a `default` case to be added, which works with current oneDNN versions
as well.

The behavior change will be called out in the release notes.
