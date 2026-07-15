# RFC: Interface and Unit Tests Separation

## Introduction

oneDNN library provides a set of public interfaces, often referred to as API,
for the users. Those interfaces are high level and implement complex
functionality underneath. There's a growing need to validate parts of the
library that make those complex implementations functionally correct.
The industry commonly knows this validation as unit testing.

While the APIs must be validated, which validation is commonly known as
interface testing, the purpose of this document is to propose a new routine for
better establishment of unit testing in oneDNN.

### Dictionary

This section describes terms used throughout this document.
* Public API - a set of oneDNN interfaces declared in
  include/oneapi/dnnl/*.h{pp} files.
* Exported (exposed) symbols - a set of functions that can be used by an
  application and successfully linked against oneDNN. When a symbol is not
  exported, the linker would fail to find it. Not all exported symbols are
  declared in public headers, which is the current state. The ideal state is
  when only public APIs are exported.
* Visibility - a property of a symbol with external linkage that controls
  whether a shared object exports it. Doesn't affect static linking.
* Linkage - a property of a symbol that controls whether other translation
  units can refer to it. `static` functions and functions in an unnamed
  namespace have internal linkage and can't be referred to from outside of
  their translation unit regardless of visibility.
* Translation unit - a source file for compiler to generate a single object
  file. Most of the time it means a single .cpp file.
* The shipped library - the final binary product that is released. If built
  from source, it means it doesn't have tests (and sometimes examples, too) in
  it.
* Interface test - a test that depends only on public API, thus, expected to
  link against the shipped library successfully.
* Unit test - a test that depends on at least one non-public symbol from the
  library.


## Motivation

The default oneDNN compilation mode contains `-fvisibility=internal` and
`-fvisibility-inlines-hidden` flags which require explicit exporting of symbols
to make them linkable. In the code such explicit declaration is implemented via
`DNNL_API` macro which sets `visibility("default")`.

Adding a new unit test as of now requires careful management of exporting
desired symbols to test and sometimes code shuffling to keep definitions in
header files instead of translation units. Such scenarios pin function bodies
to specific locations and they can't be moved, which triggers other
code-structural problems.

Unconditional symbol exporting is not an active bug today, but there's always a
chance to somehow exploit exported symbols, especially if they work with
pointers. Additionally, the shipped package is preferred to export exactly the
public API.

Today the test suites mix both kinds of tests:
* `tests/gtests/internals` and several API-level gtests (e.g.,
  `test_iface_primitive_cache.cpp`, `test_malloc.cpp`, `test_isa_hints.cpp`)
  include library headers and rely on internal symbols exported for them.
* `tests/gtests/graph/unit` already links library object files directly into a
  single `test_graph_unit` binary and registers a separate ctest target for
  each test suite via `--gtest_filter`. The single binary was chosen after
  per-file binaries, each close to the library size, exhausted disk space in
  nightly debug builds.
* benchdnn relies on non-public symbols to control and inspect the library
  (primitive cache, profiling, GPU fill, zero padding, kernel overrides, etc.).


## Proposal

The proposal splits gtests into two classes with different linking models and
keeps benchdnn on the shipped library.

### Test Classes

* Interface tests - most of existing gtests. They rely exclusively on the
  shipped shared library and use only public API. They don't include library
  headers outside of `include/` and use no internal symbols.
* Unit tests - tests of internal functions for specific scenarios, systems that
  their parts respond to state and controls correctly, infrastructure pieces
  play together well. They link a new static archive of the library and may call
  any function with external linkage, no matter its visibility.

Interface tests are kept clean implicitly: once internal symbols stop being
exported, an interface test using one fails to link against the shipped
library.

All symbols exported for unit-kind tests get hidden back, so the shipped library
doesn't grow its export list for testing purposes anymore.

### Static Archive for Unit Tests

The build composes a static archive from the same object files that form the
shipped library. Unit test binaries link against it. Static linking ignores
visibility, so the shipped library is untouched and no internal symbol leaks
into any distributed artifact. The archive is built only when a CMake target
involving unit tests is requested.

To avoid disk space issues, each component gets a single unit test binary
(e.g., `common`, `cpu`, `gpu`, `graph`), and every logical test case gets a
separate ctest target that runs the binary with a corresponding
`--gtest_filter`, the same way `test_graph_unit` works today. A crash in one
case doesn't stop validation of other cases, and the number of binaries
remains small.

The cost of the archive is that each unit test binary pulls most of the library
in. The engine contains the list of all implementations for each primitive,
which pulls all auxiliary functions to support them. Any test that touches a
function that depends internally on an engine or a primitive descriptor pulls
object files transitively, approaching the full library size.

TODO: unit binary size and link time numbers below are pending.

| Build (x64, SYCL GPU runtime) | Archive update | Unit binary link | Unit binary size |
|:------------------------------|:---------------|:-----------------|:-----------------|
| Debug                         | 28.8 s         | TBD              | TBD              |
| Release                       | 2.5 s          | TBD              | TBD              |

The archive takes 3028 MiB in debug and 232 MiB in release. benchdnn linked
against the archive, which pulls nearly the whole library, gives an upper bound
for a unit binary: 192.8 s link and 696 MiB size in debug, 104.2 s link and
85 MiB size in release.

With SYCL, device code is linked at the final executable link, so each unit
binary link includes the device code link step of all pulled object files.

Developer tools that rely on internal symbols, may choose between linking
against the archive, or linking against shared object if symbols they expose are
guarded by #DNNL_DEV_MODE macro. In that case, it can benefit from faster
re-link and doesn't expose internal symbols in a production build. Unit tests of
components with their own test suites, such as Gemmstone, fit the same model as
a part of the corresponding component binary.

#### Linkage Limitation

The archive resolves symbols with external linkage regardless of their
visibility. Functions declared `static` or placed in an unnamed namespace have
internal linkage: they get into a test binary together with their translation
unit and are validated through their external callers, but a test can't call
them directly. To test such a function directly, its linkage must change to
external (e.g., by moving it into a named internal namespace). Visibility stays
hidden, so the shipped library doesn't export it.

#### Test Doubles

Some subsystems are best validated with their dependencies replaced by test
doubles. For example, a JIT emitter that depends on the platform vector length
can be validated for all supported lengths on a single machine when the length
comes from a test-controlled source instead of the hardware query.

Unit tests may use injection-style doubles: the subsystem receives its
dependencies through parameters or a small interface, and a test passes
test-controlled implementations. Replacing library symbols at link time is not
supported, as such a replacement conflicts with the definitions the archive
brings in. Doubles belong to unit test utilities, the auxiliary supportive code
shared by unit tests.

### Tests Placement

Unit tests sources are tentatively placed under `tests/gtests/unit` mirroring
the `src` structure, e.g.:
* `tests/gtests/unit/common`
* `tests/gtests/unit/cpu/x64`
* `tests/gtests/unit/gpu/intel/gemm`
* `tests/gtests/unit/graph`

Platform-specific tests reside in their platform directory and are built only
when the platform is enabled, which avoids `#ifdef`-guarded test files.
Existing `tests/gtests/internals` and `tests/gtests/graph/unit` move to the new
location, and API-level gtests relying on internal symbols either switch to
public API or move to unit tests.

Placing tests next to sources (e.g., `src/cpu/x64/tests`) was considered.
It requires a massive overhaul of CMake files to extract test sources from
library source globs and to form test targets separately, which is not quite
convenient.

### benchdnn

benchdnn keeps linking against the shipped shared library.

Moving benchdnn to the archive is impractical due to link times. With the shared
library, a change in a single library source file requires to re-link the
library and, by default, to re-link benchdnn without recompiling its objects.
The benchdnn re-link is skipped completely with
`CMAKE_LINK_DEPENDS_NO_SHARED=ON`. With the archive, every library change
requires to re-link benchdnn, and with SYCL the device code link moves into the
benchdnn link.

| Build (x64, SYCL GPU runtime)  | Library update | benchdnn update |
|:-------------------------------|:---------------|:----------------|
| Debug, shared library          | 108.5 s        | 11.7 s          |
| Release, shared library        | 19.7 s         | 2.5 s           |
| Debug, static archive          | 28.8 s         | 192.8 s         |
| Release, static archive        | 2.5 s          | 104.2 s         |

The benchdnn binary takes 64 MiB in debug and 4.8 MiB in release when linked
against the shared library, which itself takes 664 MiB and 83 MiB
respectively. When linked against the archive, benchdnn takes 696 MiB and
85 MiB.

With the archive, the benchdnn link time doesn't depend on the amount of
changes: in release, it takes 104.1 s after a clean build and 104.2 s after a
single source file change, which is ~72% of the 145.5 s incremental build.

The debug shared library link alone takes ~99 s: offloading device code from
each of 932 object files takes ~47 s, the host link takes ~35 s, and the device
code link takes the rest.

The numbers are collected with both GPU support and Graph API enabled. A
library built without GPU support (no SYCL device code link) and/or without
Graph API is smaller and links faster, but linking against the archive still
costs considerably more than linking against the shared library.

Internal symbols benchdnn relies on stay exported and follow the
`dnnl_impl_<xxx>_` naming prefix denoting their origin, e.g.,
`dnnl_impl_zero_pad` or `dnnl_impl_gpu_fill_random`. benchdnn declares them via
`extern` directly in its sources without any library headers.

The following symbols used by benchdnn don't comply with the rule. The
suggested names are given for reference; the renaming is a part of the proposal
implementation.

| Current symbol                                                | Suggested name                                          |
|:--------------------------------------------------------------|:--------------------------------------------------------|
| `dnnl_memory_desc_create_with_string_tag`                     | `dnnl_impl_memory_desc_create_with_string_tag`          |
| `dnnl_memory_desc_set_data_type`                              | `dnnl_impl_memory_desc_set_data_type`                   |
| `dnnl_primitive_attr_set_rnn_tparams`                         | `dnnl_impl_primitive_attr_set_rnn_tparams`              |
| `dnnl_test_is_pd_in_cache`                                    | `dnnl_impl_primitive_cache_is_pd_in_cache`              |
| `dnnl_test_is_primitive_in_cache`                             | `dnnl_impl_primitive_cache_is_primitive_in_cache`       |
| `dnnl_test_set_primitive_cache_capacity_without_clearing`     | `dnnl_impl_primitive_cache_set_capacity_without_clearing` |
| `sdpa_primitive_desc_create` (forward and backward overloads) | `dnnl_impl_sdpa_forward_primitive_desc_create`, `dnnl_impl_sdpa_backward_primitive_desc_create` |

The list is collected from an x64 CPU build; GPU-specific symbols already
follow the rule.

C++ symbols in `dnnl::impl` namespaces (e.g.,
`dnnl::impl::cpu::platform::get_num_cores` or `dnnl::impl::bfloat16_t`
conversions from and to `float`) already denote their origin through the
namespace and stay exported as is.

The brgemm driver is an exception. It validates the internal x64 and AArch64
BRGEMM C++ interfaces, which can't be reasonably expressed as `extern "C"`
entry points, so those symbols stay exported as is.

A cheap hook to keep the rule enforced is a CI check comparing the dynamic
symbol table of the shipped library (`nm -D --defined-only`) against the public
API, the `dnnl_impl_` prefix, the `dnnl::impl` namespaces, and the brgemm
exception.

### Alternatives Considered

#### `DNNL_TEST_API` conditional annotation

A macro that exports annotated internals only when tests are enabled.
Test-enabled and shipped artifacts are different binaries built from different
preprocessor configurations, and test dependencies are still driven manually.

#### Dedicated all-visible shared object

A second copy of every object file compiled with `-fvisibility=default` and
assembled into a test shared object. Test binaries stay small, but every
source is compiled twice, CMake targets are duplicated, the tested objects are
not the shipped ones, and tests get a new runtime dependency.

#### Visibility manipulation via export map

All objects are compiled with default visibility, and a linker version script
localizes everything except the public API in the shipped library. Objects are
compiled once, but the export map becomes manually driven and responsible for
the product ABI, with a higher impact of a mistake in the list.

#### benchdnn on the static archive

Rejected due to re-link times covered in the benchdnn section.
