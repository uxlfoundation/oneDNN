# Introduce `trace` Level for Verbose Mode

## Introduction
This RFC proposes the addition of a new verbose level
`ONEDNN_VERBOSE=trace` to the oneDNN verbose mode infrastructure. The
introduced `trace` level provides fine-grained diagnostic output bracketing
primitive creation and execution. Like `dispatch`, it is an independent flag
that can be combined with other levels. In standard logging terminology
[[link](https://github.com/gabime/spdlog#log-levels)], the `trace` level is the
most granular logging level sitting below `debug`. While `debug` captures
high-level diagnostic information about system state and decisions, `trace`
captures the flow of execution itself - entry and exit points of operations,
fine-grained runtime checks and transient state that is only meaningful in the
context of a specific execution. In oneDNN terms, `debug` maps to the existing
`debuginfo` level (capturing internal state during primitive creation), while
`trace` captures what is created and executed, when, and in what order.

## Motivation and Objectives
The case for adding a `trace` level rests on four practical needs:

- **Crash identification during primitive execution**: Currently,
	`ONEDNN_VERBOSE=profile` prints primitive descriptor info (`pd_info`) only
	after an enqueued primitive finishes execution. If a segfault or crash occurs
	during primitive execution, no verbose output is produced and the crashing
	primitive cannot be identified without rebuilding oneDNN with
	`DNNL_DEV_MODE=ON` and rerunning with debug instrumentation. The trace level
	introduces `trace:start` and `trace:end` markers printed before and after
	execution respectively, so the last `trace:start` line on `stdout` before a
	crash unambiguously identifies the culprit primitive without requiring a
	special build.

- **Runtime execution checks**: The existing `VCHECK` and `VDISPATCH` macros
	provide guarded diagnostic output during primitive creation. There is
	currently no equivalent mechanism for primitive execution. The `VTRACE`
	macro fills that gap, allowing `execute()` bodies to emit guarded runtime
	checks and diagnostics as `trace:info` records, analogous to the
	creation-time output from `VCHECK` and `VDISPATCH`.

- **Alignment with spdlog logging levels**: oneDNN uses `spdlog` as its
	logging backend when `DNNL_EXPERIMENTAL_LOGGING` is enabled. `spdlog` defines
	the following level hierarchy: `trace > debug > info > warn > error >
	critical > off`. A future task to align oneDNN verbose levels with spdlog
	levels is made cleaner by the existence of a `trace` level; without it, the
	mapping between oneDNN and `spdlog` levels will require workarounds.

- **Consolidation of `debuginfo` levels**: Some execution-time diagnostics,
	currently available only through developer-mode `debuginfo`, such as
	execution-flow or scratchpad details, can be more useful to users as
	`trace:info` records in standard builds.

With these goals in mind, the implementation should meet the following
objectives:

1. Extend the verbose infrastructure with an independently selectable `trace`
	flag, `VTRACE` support, and start/end record subtypes; preserve compatibility
	with existing levels, filters, timestamps, and runtime controls.
2. Instrument primitive creation and execution with trace markers.
3. Implement trace-ID assignment to correlate records from concurrent
	primitive executions; attach each invocation's ID to its trace records and
	asynchronous profiler entry.
4. Integrate trace with synchronous and asynchronous profiling, and ensure the
	verbose converter does not treat the new records as timing samples.
5. Preserve existing behavior and overhead when trace is disabled.

## Proposals 

### Proposal 1 — Introduce a Dedicated trace Verbose Level (Recommended)
This proposal is straightforward as it introduces `trace` as a first-class
verbose flag in oneDNN. Like `dispatch`, it is selected independently and can
be combined with other flags, e.g. `ONEDNN_VERBOSE=trace,profile`.

The main implementation steps are:

1. Add an `exec_trace` flag and `ONEDNN_VERBOSE=trace` parsing; preserve
	existing numeric levels and profiling API behavior.
2. Define `VTRACE` and the (`trace:start`, `trace:info`, `trace:end`) record
	subtypes.
3. Assign a trace ID to each primitive invocation and propagate it through
	asynchronous profiler entries so concurrent invocations can be correlated.
4. Instrument primitive creation and execution boundaries with trace markers,
	reusing descriptor information where available.
5. Integrate trace with synchronous/asynchronous profiling; the verbose
	converter skips trace records, so they are not treated as timing samples.
6. Specify coverage for different components and update the API, documentation,
	and tests.

**Example**: The following example log, produced with
`ONEDNN_VERBOSE=trace,profile`, illustrates how the verbose trace markers
should bracket a successful primitive execution and retain the start marker if
execution crashes:

```text
# Normal execution: start, execution profile, end
onednn_verbose,v1,primitive,trace:start,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,exec_id:148901263163
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,18.0879
onednn_verbose,v1,primitive,trace:end,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,exec_id:148901263163

# Crash during execution: no profile or end record
onednn_verbose,v1,primitive,trace:start,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,exec_id:148901263163
# Illustrative diagnostic: the kernel is about to access beyond the scratchpad allocation
onednn_verbose,v1,primitive,trace:info,scratchpad access offset=4096 exceeds allocated size=2048,cpu/x64/matmul/brgemm_matmul.cpp:123,exec_id:148901263163
Segmentation fault (core dumped)
```

### Proposal 2 — profile:trace Field Within Existing Profile Logic (Alternative)
This alternative approach emits a primitive start record in the existing
`exec_profile` path, immediately before `enqueue_primitive()`, then retains the
normal `VPROF` record after successful execution. This avoids adding a
separately selectable verbose flag and identifies the primitive when execution
fails before the profile record is emitted.
([reference](https://github.com/uxlfoundation/oneDNN/pull/5089))

```text
# Normal execution
onednn_verbose,v1,primitive,exec:trace_start,cpu,matmul,<implementation>,<descriptor>
onednn_verbose,v1,primitive,exec,cpu,matmul,<implementation>,<descriptor>,18.0879

# Crash during execution
onednn_verbose,v1,primitive,exec:trace_start,cpu,matmul,<implementation>,<descriptor>
Segmentation fault (core dumped)
```

This alternative is limited to workloads with `profile_exec` enabled. It
provides no independent `VTRACE` / `trace:info` path or creation markers, and
duplicates descriptor information when profiling succeeds. It also offers no
direct mapping to a distinct `spdlog` `trace` level and requires consumers to
handle the additional record subtype.

## Implementation (Proposal 1):
Proposal 1 is recommended because its dedicated trace level supports a broader
scope than the `profile`-only alternative, including `trace:info` diagnostics
and lifecycle markers independent of execution profiling. The following
subsections expand on its implementation steps.

### Introduction of `trace` flag and runtime controls
The dedicated `exec_trace` flag represents the opt-in `trace` category selected
by `ONEDNN_VERBOSE=trace`. It is independent of legacy `level1` and `level2`,
whose meanings remain unchanged, and follows the component-filtering behavior
of execution profiling. The corresponding enum and parser updates are shown
below.

```cpp
struct verbose_t {
	enum flag_kind : uint32_t {
		// we reserve the 24 lower bits for user info
		none = 0,
		// We reserve bits 0,1 to maintain backward compatibility support
		// of VERBOSE={1,2}
		error = 1 << 2,
		create_check = 1 << 3,
		create_dispatch = 1 << 4,
		create_profile = 1 << 5,
		exec_check = 1 << 6,
		exec_profile = 1 << 7,
		profile_externals = 1 << 8,
		warn = 1 << 9,
		exec_trace = 1 << 10,
		// the upper 8 bits are reserved for devinfo levels
		debuginfo = 1 << 24,
		//
		level1 = error | exec_profile | warn,
		level2 = error | exec_profile | warn | create_profile,
		all = (uint32_t)-1,
	};
};

// ONEDNN_VERBOSE option parsing
auto update_kind = [&](const std::string &s, int &k) {
    // ... Exisiting options
	if (s == "profile_create") k |= verbose_t::create_profile;
	if (s == "profile_exec") k |= verbose_t::exec_profile;
	if (s == "trace") k |= verbose_t::exec_trace;
	if (s == "profile_externals") k |= verbose_t::profile_externals;
	if (s == "warn") k |= verbose_t::warn;
	if (s.rfind("debuginfo=", 0) == 0)
		k |= verbose_t::make_debuginfo(
				std::strtol(s.c_str() + 10, nullptr, 10));
};
```

The current C and C++ runtime APIs accept numeric levels 0, 1, and 2. Runtime
selection of trace therefore requires an additive, backward-compatible API
choice; existing values remain unchanged. `verbose_profiling_enabled()`
continues to describe timing profiling, not trace output.

### Definition of `VTRACE` and record subtypes
The trace logging interface comprises a guarded `VTRACE` helper, enabled by
`exec_trace`, and three record subtypes. `trace:start` and `trace:end` mark
lifecycle boundaries; `trace:info` carries optional diagnostics followed by the
source location. Every record ends with the trace ID as `<phase>_id:<id>`,
where `phase` is `exec` or `create`.

```cpp
// Subtypes are appended to the `trace` log type by VFORMAT.
#define VERBOSE_trace "trace"
#define VERBOSE_trace_start ":start"
#define VERBOSE_trace_info ":info"
#define VERBOSE_trace_end ":end"

// Arguments: subtype, API type, phase, trace ID, format, ...
#define VTRACE(subtype, apitype, phase, trace_id, msg, ...) \
	do { \
		if (dnnl::impl::get_verbose(dnnl::impl::verbose_t::exec_trace)) { \
			const bool is_info_ = strcmp(#subtype, "info") == 0; \
			VFORMAT(dnnl::impl::get_msec(), \
					dnnl::impl::verbose_t::exec_trace, apitype, trace, \
					VERBOSE_trace_##subtype, \
					msg "%s%s%s," CONCAT2(VERBOSE_, phase) "_id:%llu", \
					##__VA_ARGS__, is_info_ ? "," : "", \
					is_info_ ? __FILENAME__ : "", \
					is_info_ ? ":" STRINGIFY(__LINE__) : "", \
					static_cast<unsigned long long>(trace_id)); \
		} \
	} while (0)

// Mark entry to primitive execution.
VTRACE(start, primitive, exec, trace_id, "%s", pd_info.c_str());

// Emit optional context at a useful point in execution.
VTRACE(info, primitive, exec, trace_id,
		"scratchpad access offset=%zu exceeds allocated size=%zu", offset,
		scratchpad_size);

// Mark exit from primitive execution.
VTRACE(end, primitive, exec, trace_id, "%s", pd_info.c_str());
```

`trace:info` supplies optional diagnostic context; validation failures continue
through their existing check/error paths.

### Lifecycle instrumentation
In `primitive_iface.cpp`, `trace:start` and `trace:end` bracket
`stream->enqueue_primitive()` at primitive execution and
`create_primitive_iface()` at primitive creation. The existing profile
descriptor string, including runtime dimensions and strides, is shared when
either tracing or profiling needs it. Similar markers can potentially cover
primitive descriptor and memory descriptor boundaries.

```cpp
const bool trace_enabled = get_verbose(verbose_t::exec_trace, component);
const bool profile_enabled
		= get_verbose(verbose_t::exec_profile, component);

if (trace_enabled || profile_enabled) {
	// Generate the pd_info string, using runtime descriptors when required.

    // Block around execution to measure host-side timing, unless the runtime
    // supports non-blocking device-measured profiling (GPU runtimes and
    // async threadpool CPU). Tracing alone does not block, so trace:end
    // then marks the return from enqueue rather than completion.
    const bool block_on_wait = profile_enabled
            && !stream->is_verbose_profiler_enabled()
            && stream->is_syncable();

    if (block_on_wait) stream->wait();
    double start_ms = get_msec();
    // The trace ID correlates all trace records of this execution call.
    // Hashing pd_info with the start time separates concurrent calls with
    // the same descriptor.
    const size_t trace_id = trace_enabled
            ? hash_combine(hash_combine(0, pd_info), start_ms)
            : 0;
    // Printed before enqueue so that a crash during execution leaves an
    // unmatched trace:start identifying the primitive.
    if (trace_enabled)
        VTRACE(start, primitive, exec, trace_id, "%s", pd_info.c_str());

    status = stream->enqueue_primitive(primitive_iface, ctx);
    // A failed enqueue is reported between trace:start and trace:end.
    if (trace_enabled && status != status::success)
        VTRACE(info, primitive, exec, trace_id, "%s,enqueue failed: %s",
                pd_info.c_str(), dnnl_status2str(status));
    if (block_on_wait) stream->wait();

    if (stream->is_verbose_profiler_enabled()) {
        CHECK(stream->run_verbose_profiler(pd_info, start_ms, 0));
    } else {
        if (profile_enabled) {
            const double duration_ms = get_msec() - start_ms;
            VPROF(start_ms, primitive, exec, VERBOSE_profile,
                    pd_info.c_str(), duration_ms);
        }
        if (trace_enabled)
            VTRACE(end, primitive, exec, trace_id, "%s", pd_info.c_str());
    }
} else {
	status = stream->enqueue_primitive(primitive_iface, ctx);
}
```

The start trace marker is always emitted before enqueue. With asynchronous
profiling, the verbose profiler emits the profile record and the end trace
marker in enqueue order after the associated device events complete. With
blocked profiling, this is done after the stream is waited after enqueuing.

Primitive creation follows the same pattern with a different phase (`create`).

### Updates to verbose profiling and conversion
For asynchronous profiling, the profiler later processes completed entries in
enqueue order and emits the corresponding trace end marker beside the profile
record. Synchronous profiling emits  the trace marker after its blocking wait.
Neither path adds synchronization solely for tracing.

```cpp
void verbose_profiler_t::add_to_pending_primitive_list(
		double start_ms, const std::string &pd_info, uint64_t component) {
	// ... existing metadata
	last_entry.trace_enabled_ = get_verbose(verbose_t::exec_trace);
	if (last_entry.trace_enabled_)
		last_entry.trace_id_ = hash_combine(hash_combine(0, pd_info), start_ms);
}

void verbose_profiler_t::log_profile(
		const prim_profile_data_t &prof_data, double duration_ms) {
	const char *info = prof_data.pd_info_.c_str();
	const uint64_t id = prof_data.trace_id_;
	switch (static_cast<component_t::flag_kind>(prof_data.component_kind_)) {
		case component_t::graph:
			VPROF(prof_data.start_ms_, graph, exec, VERBOSE_profile, info,
					duration_ms);
			if (prof_data.trace_enabled_)
				VTRACE(end, graph, exec, id, "%s", info);
			break;
		// ... ukernel and primitive cases follow the same pattern
	}
}
```

Each pending entry stores `trace_enabled_` and `trace_id_`. The ID is derived
from the same `pd_info` and `start_ms` used at enqueue, so `trace:end` matches
the `trace:start` emitted there. `log_profile()` emits `trace:end` only for
entries with trace enabled; `VPROF` remains independent and reports timing
whenever profiling is enabled. 

### Trace ID generation
Each primitive invocation needs one ID shared by its `trace:start`,
`trace:info`, and `trace:end` records. Worker threads executing the same
invocation share the ID; separate concurrent invocations receive distinct
IDs. `pd_info` alone is not unique because concurrent invocations can use the
same primitive descriptor.

**Option 1: derive an ID from `pd_info` and `start_ms`.** This is the simplest
approach and follows the pattern used for ITT IDs. A hash can be appended to
each trace record:

```cpp
const size_t trace_id = dnnl::impl::hash_combine(
		dnnl::impl::hash_combine(0, pd_info), start_ms);

VTRACE(start, primitive, exec, trace_id, "%s", pd_info.c_str());
VTRACE(end, primitive, exec, trace_id, "%s", pd_info.c_str());
```

This is a best-effort key, not a uniqueness guarantee - but involves the least
change in code behavior.

**Option 2: allocate IDs from an atomic counter.** A counter owned by the
verbose infrastructure assigns a monotonically increasing ID to each traced
invocation:

```cpp
struct verbose_t {
	static uint64_t next_trace_id() {
		static std::atomic<uint64_t> counter {1};
		return counter.fetch_add(1, std::memory_order_relaxed);
	}
};

const uint64_t trace_id = trace_enabled ? verbose_t::next_trace_id() : 0;
```

This avoids collisions between concurrent invocations within the process. The
same ID is included in each marker, and stored in the pending profiler entry
so asynchronous `log_profile()` can attach it to `trace:end`.

The example [PoC](https://github.com/uxlfoundation/oneDNN/pull/6527)
implementing the `trace` verbose mode uses the first option to add trace IDs
to the logged primitives.

### Coverage Considerations
Trace IDs identify primitive invocations, not worker threads: threads executing
one invocation share an ID, while concurrent or nested invocations receive
distinct IDs. For graph and ukernel work routed through the verbose profiler,
the pending profile entry carries the component, `pd_info`, and trace ID;
`log_profile()` then emits the matching `trace:end` with the profile record in
enqueue order after completion.

## Usage and PoC:

A PoC for the `trace` verbose mode is provided here: [link](https://github.com/uxlfoundation/oneDNN/pull/6527)

Example usage with this implemented mode is shown below:

### Example 1: Successful trace logging

```sh
# Standard exec profiling with trace:
ONEDNN_VERBOSE=profile,trace ./benchdnn --conv mb1_ic64oc64_ih56oh56kh3sh1dh0ph1
```

Expected output (shortened; reorders created by benchdnn are omitted):

```text
onednn_verbose,v1,primitive,trace:start,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,create_id:13591497206375581441
onednn_verbose,v1,primitive,create:cache_miss,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,0.709961
onednn_verbose,v1,primitive,trace:end,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,create_id:13591497206375581441
onednn_verbose,v1,primitive,trace:start,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,exec_id:7023598612480132245
onednn_verbose,v1,primitive,exec,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,0.342041
onednn_verbose,v1,primitive,trace:end,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,exec_id:7023598612480132245
```

The profile records are printed between the corresponding start and end trace
markers, which share the same `create_id` or `exec_id`.

### Example 2: Crash Reproducer

```sh
# Demonstrates a convolution crashing during execution due to a corrupted dst pointer.
ONEDNN_VERBOSE=trace ./benchdnn --conv mb1_ic64oc64_ih56oh56kh3sh1dh0ph1 
```
Example output:

```text
onednn_verbose,v1,primitive,trace:start,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,create_id:13591497206375581441
onednn_verbose,v1,primitive,trace:end,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,create_id:13591497206375581441
onednn_verbose,v1,primitive,trace:start,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,exec_id:7023598612480132245
Segmentation fault (core dumped)
```
The unmatched start trace marker identifies the crashing primitive. Its problem
description is in benchdnn format and can be used to reproduce the crash in
isolation.

## Limitations/Drawbacks
- Start/end records add output and can perturb short workloads; each
	`trace:info` record adds further formatting and I/O overhead.
- Initial coverage is limited to instrumented primitive creation, execution
	and profiler paths. Primitive-descriptor, memory-descriptor, memory
	creation, and other component boundaries do not emit markers.
- The trace ID is not available inside primitive implementations, so
	`trace:info` records emitted from `execute()` cannot be correlated with the
	start/end trace markers of the invocation - this can be resolved though 
    using Option 2 for trace ID generation.
- With trace alone, no synchronization is added, so on asynchronous streams the
	end trace marker is printed when enqueue returns. A crash during device
	execution may then occur after the end trace marker, and the unmatched start
	trace marker no longer identifies the crashing primitive.

## Future Work
- Add start/end markers to primitive-descriptor, memory-descriptor,
	memory-creation, and other selected lifecycle boundaries.
- Extend trace coverage across remaining components and direct ukernel paths
	where useful.
- Propagate trace IDs to `VTRACE` diagnostics inside primitive implementations
	and to nested invocations; consider parent IDs and process IDs for
	cross-process log correlation.
- Migrate selected `debuginfo` diagnostics to `trace:info`. Execution-time
	diagnostics useful in standard builds are viable candidates for
	`VTRACE(info, ...)`. For example, the x64 JIT softmax forward primitive
	(`jit_uni_softmax_fwd_t`) currently uses `VDEBUGINFO` during execution to
	report the selected implementation and runtime dimensions and strides, along
	with raw source/destination pointers. The useful execution context can be
	migrated to `trace:info` without logging the addresses:

```cpp
VTRACE(info, primitive, exec, trace_id,
	"%s,outer_size=%" PRId64 " outer_stride=%" PRId64
	" inner_size=%" PRId64 " inner_stride=%" PRId64
	" axis_stride=%" PRId64,
	pd()->impl_name(), outer_size, outer_stride, inner_size,
	inner_stride, axis_stride);
```

## References:
- [oneDNN verbose mode](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)
- [Verbose message catalogue](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose_table.html)
- [Verbose runtime API](https://github.com/uxlfoundation/oneDNN/blob/main/include/oneapi/dnnl/dnnl_common.h)
- [spdlog log levels](https://github.com/gabime/spdlog#log-levels)

(EOD)
