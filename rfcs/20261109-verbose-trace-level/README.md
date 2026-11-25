# Introduce `trace` Level for Verbose Mode

## Introduction
This RFC proposes the addition of a new verbose level
`ONEDNN_VERBOSE=trace` to the oneDNN verbose mode infrastructure. The
introduced `trace` level prints start and end trace markers around primitive
creation and execution, and tags each pair with a trace ID that identifies the
call. The end marker is printed once the work has completed, following the same
synchronous or asynchronous mechanism as execution profiling. This records what
is created and executed, in what order, and which call was running when a
crash or hang occurred.

Like `dispatch`, `trace` is an independent flag that can be combined with other
levels. It serves a purpose not covered by existing levels: `dispatch` explains
implementation selection during creation, `check` reports failed validations,
and `profile` reports timings only after a call completes.

## Motivation and Objectives
The case for adding a `trace` level rests on three practical needs:

- **Crash and hang identification during primitive execution**: Currently,
	`ONEDNN_VERBOSE=profile` prints primitive descriptor info (`pd_info`) only
	after an enqueued primitive finishes execution. If a segfault, page fault or
	hang occurs during primitive execution, no verbose output is produced and the
	failing primitive cannot be identified without rebuilding oneDNN with
	`DNNL_DEV_MODE=ON` and rerunning with debug instrumentation. The trace level
	introduces start and end trace markers printed before execution and after it
	completes, so a start marker without a matching end marker identifies the
	culprit primitive without requiring a special build. On asynchronous
	runtimes, where several calls may be in flight, the culprit is the earliest
	unmatched start marker.

- **Correlation of creation and execution calls**: The existing levels report
	what happened within a call (`dispatch` and `check` on failures, `profile`
	after completion), but do not bracket each call. Trace markers record the
	order in which primitives are created and executed, and a trace ID shared by
	the start and end markers of a call allows matching them when calls with the
	same descriptor run concurrently.

- **Invocation-wise filtering and processing of verbose data**: With each call
	bracketed by start and end markers that share a trace ID, tools such as the
	verbose converter can select the records of a particular primitive
	invocation, e.g. the creation, nested primitive and profiling records printed
	between its markers, and filter or process verbose logs per invocation
	rather than per line.

With these goals in mind, the implementation should meet the following
objectives:

1. Extend the verbose infrastructure with an independently selectable `trace`
	flag, `VTRACE` support, and start/end record subtypes; preserve compatibility
	with existing levels, filters, timestamps, and runtime controls.
2. Instrument primitive creation and execution with trace markers.
3. Implement trace-ID assignment to correlate the start and end markers of
	concurrent calls; attach each call's ID to its markers and asynchronous
	profiler entry.
4. Print the end marker only after the work completes, following execution
	profiling: through the asynchronous verbose profiler where it is enabled,
	and through a blocking stream wait otherwise.
5. Update the verbose converter to recognize trace markers without treating
	them as timing samples, and to use their trace IDs to filter and group
	verbose records by invocation.
6. Preserve existing behavior and overhead when trace is disabled.

## Proposals 

### Proposal 1 — Introduce a Dedicated trace Verbose Level (Recommended)
This proposal is straightforward as it introduces `trace` as a first-class
verbose flag in oneDNN. Like `dispatch`, it is selected independently and can
be combined with other flags, e.g. `ONEDNN_VERBOSE=trace,profile`.

The main implementation steps are:

1. Add `create_trace` and `exec_trace` flags, following the existing
	`create_*`/`exec_*` naming (e.g. `create_profile`/`exec_profile`), and
	`ONEDNN_VERBOSE=trace` parsing that enables both; preserve existing numeric
	levels and profiling API behavior.
2. Define `VTRACE` and the trace record format: trace markers use the existing
	`<operation>:<subtype>` layout with `trace` as the subtype (`create:trace`,
	`exec:trace`), followed by the descriptor and a `start_id:<id>` or
	`end_id:<id>` field identifying the marker and the call.
3. Assign a trace ID to each primitive invocation and propagate it through
	asynchronous profiler entries so concurrent invocations can be correlated.
4. Instrument primitive creation and execution boundaries with trace markers,
	reusing descriptor information where available.
5. Integrate trace with synchronous/asynchronous profiling so the end marker
	is printed after the work completes, and update the verbose converter to
	recognize trace markers and group records by trace ID.
6. Specify coverage for different components and update the API, documentation,
	and tests.

**Example**: The following example log, produced with
`ONEDNN_VERBOSE=trace,profile`, illustrates how the verbose trace markers
should bracket a successful primitive execution and retain the start marker if
execution crashes:

```text
# Normal execution: start, execution profile, end
onednn_verbose,v1,primitive,exec:trace,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,start_id:1
onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,18.0879
onednn_verbose,v1,primitive,exec:trace,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,end_id:1

# Crash during execution: no profile or end marker
onednn_verbose,v1,primitive,exec:trace,cpu,matmul,brg_matmul:avx512_core,undef,src:f32::blocked:abc::f0 wei:f32::blocked:abc::f0 bia:f32::blocked:abc::f0_mask4 dst:f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,start_id:2
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

This alternative is limited to workloads with `profile_exec` enabled, so the
markers cannot be enabled without timing output. It provides no creation
markers and no end marker or trace ID to match the start record with its
completion, and duplicates descriptor information when profiling succeeds.
It also requires consumers to handle the additional record subtype.

## Implementation (Proposal 1):
Proposal 1 is recommended because its dedicated trace level supports a broader
scope than the `profile`-only alternative, including creation markers, end
markers with trace IDs, and markers independent of execution profiling. The following
subsections expand on its implementation steps.

### Introduction of `trace` flags and runtime controls
The dedicated `create_trace` and `exec_trace` flags represent the opt-in
`trace` category selected by `ONEDNN_VERBOSE=trace`, which enables both, in the
same way `check` enables `create_check` and `exec_check`. They are independent
of legacy `level1` and `level2`, whose meanings remain unchanged, and follow
the component-filtering behavior of profiling. 
The corresponding enum and parser updates are shown below.

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
		create_trace = 1 << 11,
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
	if (s == "trace")
		k |= verbose_t::create_trace | verbose_t::exec_trace;
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
continues to describe timing profiling, not trace output. With experimental
logging enabled, both trace flags map to the `spdlog` `trace` level.

### Definition of `VTRACE` and record format
The trace logging interface comprises a guarded `VTRACE` helper and a single
`trace` record subtype. Trace markers follow the existing
`<operation>:<subtype>` layout used by `create:check` and `create:dispatch`:
the operation is the phase being traced (`create` or `exec`) and the subtype is
`trace`. Each marker ends with a `start_id:<id>` or `end_id:<id>` field, which
identifies both the marker and the call. `VTRACE` derives its flag from the
phase, so creation markers are gated by `create_trace` and execution markers by
`exec_trace`.

```cpp
// Subtype appended to the operation by VFORMAT, e.g. "exec:trace".
#define VERBOSE_trace ":trace"

// Arguments: marker (start|end), API type, phase (create|exec), trace ID,
// descriptor string
#define VTRACE(marker, apitype, phase, trace_id, info) \
	do { \
		if (dnnl::impl::get_verbose(dnnl::impl::verbose_t::phase##_trace)) { \
			VFORMAT(dnnl::impl::get_msec(), \
					dnnl::impl::verbose_t::phase##_trace, apitype, phase, \
					VERBOSE_trace, "%s," #marker "_id:%llu", (info), \
					static_cast<unsigned long long>(trace_id)); \
		} \
	} while (0)

// Mark entry to primitive execution.
VTRACE(start, primitive, exec, trace_id, pd_info.c_str());

// Mark completion of primitive execution.
VTRACE(end, primitive, exec, trace_id, pd_info.c_str());
```

This produces:

```text
onednn_verbose,v1,primitive,exec:trace,<pd_info>,start_id:1
onednn_verbose,v1,primitive,exec:trace,<pd_info>,end_id:1
```

Formatting and argument evaluation occur only when the corresponding trace
flag is enabled.

### Lifecycle instrumentation
In `primitive_iface.cpp`, the start and end trace markers bracket
`stream->enqueue_primitive()` at primitive execution and
`create_primitive_iface()` at primitive creation. The existing profile
descriptor string, including runtime dimensions and strides, is shared when
either tracing or profiling needs it. Similar markers can potentially cover
primitive descriptor and memory descriptor boundaries during creation.

```cpp
const bool trace_enabled = get_verbose(verbose_t::exec_trace, component);
const bool profile_enabled
		= get_verbose(verbose_t::exec_profile, component);

if (trace_enabled || profile_enabled) {
	// Generate the pd_info string, using runtime descriptors when required.

	// Block around execution unless the runtime supports non-blocking
	// device-measured profiling (GPU runtimes and async threadpool CPU), so
	// that the end trace marker and the profile record follow completion.
	const bool block_on_wait = !stream->is_verbose_profiler_enabled()
			&& stream->is_syncable();

	if (block_on_wait) stream->wait();
	double start_ms = get_msec();
	// The trace ID correlates the start and end trace markers of this call.
	const uint64_t trace_id = trace_enabled ? verbose_t::next_trace_id() : 0;
	// Printed before enqueue so that a crash during execution leaves an
	// unmatched start trace marker identifying the primitive.
	if (trace_enabled) VTRACE(start, primitive, exec, trace_id, pd_info.c_str());

	status = stream->enqueue_primitive(primitive_iface, ctx);
	if (block_on_wait) stream->wait();

	if (stream->is_verbose_profiler_enabled()) {
		// The async profiler emits the profile record and the end trace
		// marker once the device events complete.
		CHECK(stream->run_verbose_profiler(pd_info, start_ms, 0, trace_id));
	} else {
		if (profile_enabled) {
			const double duration_ms = get_msec() - start_ms;
			VPROF(start_ms, primitive, exec, VERBOSE_profile,
					pd_info.c_str(), duration_ms);
		}
		if (trace_enabled) VTRACE(end, primitive, exec, trace_id, pd_info.c_str());
	}
} else {
	status = stream->enqueue_primitive(primitive_iface, ctx);
}
```

The start trace marker is always emitted before enqueue, and the end trace
marker only after the work completes, following execution profiling:

- When the asynchronous verbose profiler is enabled (execution profiling on a
	runtime that supports it), the profiler emits the profile record and the end
	trace marker in enqueue order after the associated device events complete.
- Otherwise, including `ONEDNN_VERBOSE=trace` without `profile`, the stream is
	waited before and after enqueue, and the end trace marker is emitted after
	the wait. On synchronous CPU runtimes the work is already complete when
	enqueue returns, so the wait adds no cost.

On asynchronous runtimes several calls may be in flight, so after a crash or
hang the culprit is the earliest start trace marker without a matching end
trace marker.

Primitive creation follows the same pattern with the `create` phase, gated by
`create_trace`. Creation is synchronous, so no wait is needed. Graph compiled
partition execution follows the same pattern as primitive execution.

### Updates to verbose profiling and conversion
For asynchronous profiling, the profiler later processes completed entries in
enqueue order and emits the corresponding end trace marker beside the profile
record. Synchronous profiling emits the end trace marker after its blocking
wait. Since the asynchronous verbose profiler is only enabled with execution
profiling, `ONEDNN_VERBOSE=trace` alone uses the blocking path.

```cpp
void verbose_profiler_t::add_to_pending_primitive_list(double start_ms,
		const std::string &pd_info, uint64_t component, uint64_t trace_id) {
	// ... existing metadata
	last_entry.trace_id_ = trace_id; // 0 when trace is disabled
}

void verbose_profiler_t::log_profile(
		const prim_profile_data_t &prof_data, double duration_ms) {
	const char *info = prof_data.pd_info_.c_str();
	const uint64_t id = prof_data.trace_id_;
	switch (static_cast<component_t::flag_kind>(prof_data.component_kind_)) {
		case component_t::graph:
			VPROF(prof_data.start_ms_, graph, exec, VERBOSE_profile, info,
					duration_ms);
			if (id != 0) VTRACE(end, graph, exec, id, info);
			break;
		// ... ukernel and primitive cases follow the same pattern
	}
}
```

The trace ID assigned at enqueue is passed through `run_verbose_profiler()` to
`add_to_pending_primitive_list()` and stored in the pending entry, so the end
trace marker carries the same ID as the start trace marker emitted at enqueue.
Since IDs start at 1, an ID of 0 marks entries with trace disabled, and
`log_profile()` emits the end trace marker only for entries with a non-zero
ID; `VPROF` remains independent and reports timing whenever profiling is
enabled.

The verbose converter is updated to recognize `create:trace` and `exec:trace`
records. They are excluded from the `create` and `exec` events used as timing
samples, since their last field is a trace ID, not a time. The `start_id` and
`end_id` fields are used to filter and group verbose records by invocation.

### Trace ID generation
Each call needs one ID shared by its start and end trace markers. Worker
threads executing the same call share the ID; separate concurrent calls
receive distinct IDs. `pd_info` alone is not unique because concurrent calls
can use the same primitive descriptor.

**Option 1: derive an ID from `pd_info` and `start_ms`.** This follows the
pattern used for ITT IDs:

```cpp
const size_t trace_id = dnnl::impl::hash_combine(
		dnnl::impl::hash_combine(0, pd_info), start_ms);
```

This is a best-effort key, not a uniqueness guarantee: concurrent calls with
the same descriptor and the same millisecond timestamp produce the same ID.
The resulting IDs are also long and carry no ordering information.

**Option 2: allocate IDs from an atomic counter (preferred).** A counter owned
by the verbose infrastructure in `verbose.hpp` assigns a monotonically
increasing ID to each traced call:

```cpp
struct verbose_t {
	static uint64_t next_trace_id() {
		static std::atomic<uint64_t> counter {1};
		return counter.fetch_add(1, std::memory_order_relaxed);
	}
};

const uint64_t trace_id = trace_enabled ? verbose_t::next_trace_id() : 0;
```

This guarantees unique IDs within the process, keeps them short, and reflects
the order in which calls were made. Creation and execution share the counter,
so an ID is unique across both phases. Since the ID cannot be recomputed later,
it is passed to the asynchronous verbose profiler with the pending entry, as
described above.

### Coverage Considerations
Trace IDs identify primitive invocations, not worker threads: threads executing
one invocation share an ID, while concurrent or nested invocations receive
distinct IDs. Graph compiled partition execution emits the start trace marker
at submission like primitive execution. For graph work routed through the
verbose profiler, the pending profile entry carries the component, `pd_info`,
and trace ID; `log_profile()` then emits the matching end trace marker with the
profile record in enqueue order after completion. Ukernel entries routed
through the profiler are submitted without a trace ID and emit no trace
markers.

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
onednn_verbose,v1,primitive,create:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,start_id:1
onednn_verbose,v1,primitive,create:cache_miss,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,0.709961
onednn_verbose,v1,primitive,create:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,end_id:1
onednn_verbose,v1,primitive,exec:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,start_id:2
onednn_verbose,v1,primitive,exec,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,0.342041
onednn_verbose,v1,primitive,exec:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,end_id:2
```

The profile records are printed between the corresponding start and end trace
markers, which share the same trace ID in their `start_id` and `end_id`
fields.

### Example 2: Crash Reproducer

```sh
# Demonstrates a convolution crashing during execution due to a corrupted dst pointer.
ONEDNN_VERBOSE=trace ./benchdnn --conv mb1_ic64oc64_ih56oh56kh3sh1dh0ph1 
```
Example output:

```text
onednn_verbose,v1,primitive,create:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,start_id:1
onednn_verbose,v1,primitive,create:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,end_id:1
onednn_verbose,v1,primitive,exec:trace,cpu,convolution,brg_conv_fwd:avx512_core,forward_training,src:f32::blocked:acdb::f0 wei:f32::blocked:AcdB16b16a::f0 bia:f32::blocked:a::f0 dst:f32::blocked:acdb::f0,,alg:convolution_direct,mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1,start_id:2
Segmentation fault (core dumped)
```

The unmatched start trace marker identifies the crashing primitive. Its problem
description is in benchdnn format and can be used to reproduce the crash in
isolation.

## Limitations/Drawbacks
- Start and end trace markers add two lines per creation and execution call,
	which increases output volume and can perturb short workloads.
- Without the asynchronous verbose profiler, including `ONEDNN_VERBOSE=trace`
	alone, the stream is waited before and after each execution so the end trace
	marker follows completion. On asynchronous runtimes this serializes
	execution, adds overhead, and can hide timing-dependent issues.
- With the asynchronous verbose profiler, end trace markers are printed only
	once device events complete. A crash or lost event leaves unmatched start
	trace markers, and several calls may be in flight at once, so the culprit is
	the earliest unmatched start trace marker rather than the last line.
- Initial coverage is limited to instrumented primitive creation, primitive
	and graph compiled partition execution, and profiler paths.
	Primitive-descriptor, memory-descriptor, memory creation, and other
	component boundaries do not emit markers, and ukernel entries emit none.
- The trace ID is not available inside primitive implementations or other
	verbose records, so records printed between the start and end trace markers
	of a call cannot always be attributed to it when calls overlap.

## Future Work
- Add start/end markers to primitive-descriptor, memory-descriptor,
	memory-creation, and other selected lifecycle boundaries.
- Extend trace coverage across remaining components where useful.
- Enable the asynchronous verbose profiler for `ONEDNN_VERBOSE=trace` without
	execution profiling, so end trace markers are printed from device event
	completion instead of a blocking wait. This requires the profiler to skip
	the profile record for entries without profiling enabled.
- Introduce breakpoint markers that can be placed at selected points inside
	long-running creation or execution, such as nested primitive operations,
	e.g. `exec:trace,<pd_info>,break_id:<id>`. The last breakpoint printed
	before a crash narrows down where within the call the failure occurred.
	This requires the call's trace ID to be available inside primitive
	implementations (see below).
- Propagate trace IDs to primitive implementations and nested invocations;
	consider parent IDs and process IDs for cross-process log correlation.

## References:
- [oneDNN verbose mode](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)
- [Verbose message catalogue](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose_table.html)
- [Verbose runtime API](https://github.com/uxlfoundation/oneDNN/blob/main/include/oneapi/dnnl/dnnl_common.h)
- [spdlog log levels](https://github.com/gabime/spdlog#log-levels)
- [Verbose converter](https://github.com/uxlfoundation/oneDNN/tree/main/scripts/verbose_converter)
- [PoC: `trace` verbose mode](https://github.com/uxlfoundation/oneDNN/pull/6527)
- [Proposal 2 reference: start record in profile logic](https://github.com/uxlfoundation/oneDNN/pull/5089)

(EOD)
