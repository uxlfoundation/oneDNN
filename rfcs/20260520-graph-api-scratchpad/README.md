# RFC: Scratchpad Memory Management for Graph API

## Summary

This RFC describes the current scratchpad memory management design in oneDNN's
Graph API execution, identifies its incompatibilities with SYCL command graph
capture and asynchronous thread pool runtime, and proposes alternative designs.

## 1. Current Design

### 1.1 Allocation and Deallocation Timing

Each graph partition allocates scratchpad memory per execution call. The
scratchpad memory will be deallocated when the executioncall exits. Take the
`larger_partition_kernel_t` as an example:

```cpp
status_t larger_partition_kernel_t::sycl_execute_impl(...) {
    // RAII. Allocated at the start of each execute call
    temporary_scratchpad_t scratchpad(
            memory_planner_.total_internal_temporary_size(), p_engine_, *g_alloc_);

    // Scratchpad is divided and reused as arguments of executables.
    prepare_args_set(res, inputs, outputs, scratchpad);

    // Loop over executables/primitives.
    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        returned_event = subgraph_->execs_[i]->execute_sycl(
                p_stream, res->get_exec_args()[i], deps);
    }

    scratchpad.set_deps(returned_event);
    // Deallocated here when scratchpad goes out of scope.
    // The event dependency ensures the GPU finishes before the
    // user-provided deallocator frees the device memory.
}
```

### 1.2 Key Characteristics

- The user-provided allocator (via the `allocator` and `engine` API) manages the
  actual device memory. oneDNN merely requests and releases buffers.
- Scratchpad allocation happens at the beginning of each `execute()` /
  `sycl_execute_impl()` call.
- Scratchpad deallocation happens at the end of each `execute()` call. The
  `temporary_scratchpad_t` destructor passes a dependency event (if applicable)
  to the user's deallocator so that asynchronous device work completes before
  memory is freed.
- The scratchpad object is scoped to a single `execute()` invocation. The
  underlying device memory lifetime can be prolonged by the dependency event.
- Each thread allocates its own scratchpad (stack-local), so concurrent
  `execute()` calls on the same partition are safe.
- The memory planning stage at partition compilation computes a flat, contiguous
  scratchpad buffer. Intermediate tensors inside the partition and scratchpad
  memories required by underlying primitives are assigned offsets within that
  buffer.
- In most cases, the scratchpad size is determined at partition compilation
  time. When needed, the size can also be adjusted at execution time before the
  scratchpad object is created.

### 1.3 Pros and Cons

**Pros:**

- Thread-safe: Stack-local allocation means concurrent `execute()` calls on the
  same compiled partition are inherently safe without any synchronization.
- Minimal memory footprint: Scratchpad memory is held only during execution.
  Idle compiled partitions consume zero temporary memory.
- Honors allocator contract: Memory is allocated through allocator API. Users
  can implement pool-based or per-device allocators externally and count memory
  consumption if they wish. Memory alignment is passed to the allocator as part
  of the interface.
- Simple implementation: No cache management, eviction policy, or lifetime
  tracking needed (compared to the alternatives proposed below).
- Execution-time size control: Scratchpad size can vary per execution (compared
  to the fixed scratchpad in the primitive API).

**Cons:**

- Per-call allocation overhead: Every `execute()` call triggers `malloc` /
  `free` (or equivalent). System/driver calls are expensive and can dominate
  execution time for small partitions. This makes a pool-based allocator or
  memory allocation library (tcmalloc, jemalloc, etc.) a necessity on the user
  side. Kernel performance depends on the quality of the user-provided
  allocator.
- Incompatible with SYCL command graph capture: Recorded kernels reference
  scratchpad addresses that are freed before replay. See Section 2.
- Async deallocation hazard: Incorrect user allocators cause use-after-free on
  the device. This is especially critical for the async CPU thread pool runtime,
  which does not support dependency events. See Section 3.

## 2. Incompatibility with SYCL Command Graph Capture

### 2.1 Problem

SYCL command graph (`sycl::ext::oneapi::experimental::command_graph`) records
kernel submissions for later replay. During recording:

- The queue is in recording mode. All kernel submissions are captured as graph
  nodes rather than executed immediately.
- All memory addresses referenced by recorded kernels must remain valid and
  stable across recording and all subsequent replays.

The current scratchpad design violates this requirement:

```
Recording:
  execute() → allocates scratchpad at address 0xA000
           → records kernels referencing 0xA000
           → deallocates scratchpad (0xA000 freed)

Replay:
  graph.update()  → kernels still reference 0xA000 (dangling!)
```

### 2.2 Workaround

Following the zero pool design in GPU backend, a `recorded_scratchpad_cache_t`
was developed to persist scratchpad memory during recording:

```cpp
temporary_scratchpad_t &scratchpad = recording
        ? recorded_scratchpad_cache_t::get_or_create(
                  p_stream, this, sp_size, p_engine_, *g_alloc_)
        : normal_scratchpad;
```

**Pros:**

- Minimal change to existing logic — only the recording path is affected.
- Correctly ensures scratchpad addresses are stable across recording and replay.
- Follows established pattern used in `zero_pool_t` design.
- No API changes required.

**Cons:**

- Permanent memory leak — because the recorded graph's lifetime is unknown,
  cached scratchpads are never returned to the user's allocator, consuming
  device memory for the entire process lifetime. The problem is more severe than
  `zero_pool_t` because scratchpads can be much larger.
- Memory footprint grows with multiple recorded graphs and concurrent execution.
- Code duplication — every kernel requires recording-aware branching.

## 3. Incompatibility with Asynchronous Thread Pool Runtime

### 3.1 Problem

When using `ONEDNN_CPU_RUNTIME=THREADPOOL`, the user provides an external thread
pool. Execution may be asynchronous — `execute()` returns before the actual CPU
work completes.

Unlike the SYCL or OCL runtime for GPU, where the returned event can be used to
prolong the scratchpad lifetime and defer deallocation, the current async thread
pool runtime on CPU does not support dependency events. Consequently, the
on-stack scratchpad object is destroyed immediately when `execute()` returns and
the underlying buffer is deallocated which causes use-after-free because the
actual CPU work has not yet completed.

### 3.2 Workaround

We rely on the in-order nature of the CPU async thread pool runtime and prolong
the scratchpad lifetime by submitting it to the thread pool as a follow-up task.
This ensures the scratchpad buffer is deallocated only after the main CPU work
completes.

```cpp
void prolong_temporary_scratchpad_lifetime(const stream_t *stream,
        const std::shared_ptr<temporary_scratchpad_t> &scratchpad) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto *tp_stream
            = dnnl::impl::utils::downcast<dnnl::impl::cpu::cpu_stream_t *>(
                    const_cast<stream_t *>(g_stream));
    tp_stream->before_exec_hook();

    // Capture the scratchpad object and extend its lifetime till the finish of
    // this dummy kernel.
    parallel(1, [=](int, int) { UNUSED(scratchpad); });

    tp_stream->after_exec_hook();
#endif
}
```

**Pros:**

- Correctly ensures the scratchpad remains available until async CPU work
  completes.
- No API changes required.

**Cons:**

- Code duplication: every kernel must add runtime-specific branches.
- Inconsistent logic between different CPU and GPU runtimes.
- The extra follow-up task submission incurs performance overhead.

## 4. Proposed Alternative Designs

### 4.1 Proposal A: Kernel-Owned Persistent Scratchpad

Allocate the scratchpad once at compile time (or first execution) and reuse it
across all subsequent `execute()` calls. It similar to the library mode
scratchpad used in primitive API. The main difference is that Graph API's
scratchapd is allocated from the allocator API specified by user.

```cpp
class kernel_base_t {
    // Allocated once, reused across executions
    std::unique_ptr<temporary_scratchpad_t> scratchpad_;

    status_t compile_impl(...) {
        ...
        scratchpad_ = std::make_unique<temporary_scratchpad_t>(
                memory_planner_.total_internal_temporary_size(), p_engine_, *g_alloc_);
    }

    status_t sycl_execute_impl(...) {
        prepare_args_set(res, inputs, outputs, *scratchpad_);
        ...
    }
};
```

**Pros:**

- Zero per-call allocation overhead.
- Naturally compatible with SYCL graph capture with stable address.
- Simple lifecycle: the scratchpad is freed when the compiled partition is
  destroyed.

**Cons:**

- Not thread-safe: Concurrent `execute()` calls on the same compiled partition
  would corrupt the shared scratchpad.
- Memory footprint: Scratchpad memory is held for the entire lifetime of the
  compiled partition, even when idle.
- Requires pre-calculated scratchpad size: In some cases the library cannot
  determine the scratchpad size at compilation time.

### 4.2 Proposal B: User-Managed Scratchpad (Explicit API)

Expose the scratchpad as a user-visible buffer, similar to the primitive API's
user mode scratchpad and cuDNN's workspace API. The user allocates the buffer
and passes it to `execute()` as a raw host memory pointer or device USM pointer
which is allocated from the engine which is used form compialtion and execution.

C++ API example:

```cpp
// At compile time, query the required scratchpad size, in byte.
auto cp = part.compile(eng, inputs, outputs);
size_t spad_lt = cp.get_scratchpad_logical_tensor();

// User allocates scratchpad buffer
void *spad = sycl::malloc_device(spad_lt.get_mem_size(), q);

// Create scratchpad tensor
auto spad_ts = tensor(spad_lt, engine, spad);

// Pass to every execute call
cp.execute(stream, inputs, outputs, spad_ts);

// SYCL execution
sycl_interop::execute(cp, stream, inputs, outputs, spad_ts, deps);
```

It's preferred to pass scratcpad to `exectue()` as a tensor rather than raw
pointer:

- Consistent type as inputs and output arguments which are all tensors.
- Explicit semantics of the engine used by scratchapd.
- Explicit scratchpad size which is part of the logical tensor.

oneDNN should be able to calculate or estimate the scratchpad size at partition
compilation stage and return the size to users via the new
`get_scratchpad_logical_tensor()` interface. For internal implementations, any
cases that scratchpad size cannot be got at compilation stage, an error should
be reported and escalated.

**Pros:**

- Zero internal allocation overhead.
- The user gains total control over scratchpad memory including lifetime
  management and memory properties (alignment, placement, etc.).
- Matches established patterns (cuDNN workspace, oneDNN primitive user mode
  scratchpad).

**Cons:**

- API change: Requires new public API surface.
- Memory misalignment: Misaligned memory can cause performance penalty.
  Currently we pass the alignment parameter to the user through the allocator
  API and the user need to fulfill this requirement when returning a buffer to
  the library. With the new API, the user also own the memory alignment property
  of a scratchpad. Additionally, the library implementations may issue a verbose
  warning message if the user-provided scratchpad does not satisfy the execpted
  alignment.

**Backward compatibility:**

If scratchpad buffer is passed to `execute()`, it will be used for kernel
executions. The library will skip the internal temporary sctachpad allocation.

If no scratchpad buffer is passed to `execute()`, the library falls back to the
current behavior: creating a temporary scratchpad object and allocating through
the allocator API. In this mode, scratchpad memory is managed per-call and
remains incompatible with SYCL graph recording. Documentation need update to
clarify this behavior.

## 5. Comparison

| Property                    | Current (per-call) | A (kernel-owned) | B (user-managed) |
|-----------------------------|:------------------:|:-----------------:|:----------------:|
| Alloc overhead per execute  | High (depends)     | None              | None             |
| Thread-safe concurrent exec | Yes                | No                | User's duty      |
| SYCL graph compatible       | No (need WA)       | Yes               | User's duty      |
| Async threadpool compatible | No (need WA)       | Yes               | User's duty      |
| Memory efficiency           | Good               | Poor              | User's duty      |
| API change required         | No                 | No                | Yes              |
| Implementation complexity   | Low                | Low               | Low              |
| Runtime scratchpad size     | Yes                | No                | No               |
| Memory alignment control    | Yes                | Yes               | No               |


## Recommendation

It is recommended to implement the proposal B in section 4.2 with a backward
compatible support to the current API behavior.

Please note that we may consider to deprecate and drop the current pre-call
managed scratchpad approach and stick to the user-managed scratchpad in the next
major release where backward compatibility breaking change is allowed. We will
discuss it with another RFC document.
