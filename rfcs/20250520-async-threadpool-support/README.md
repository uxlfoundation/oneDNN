# Proposal to support asynchronous threadpool runtime

## Background

XLA is switching to a new threadpool runtime on CPU. In particular:
- It has an asynchronous `parallel_for` implementation, meaning the
  `parallel_for` function call returns while the submitted tasks are still
  running on the threadpool.
- XLA will have a single threadpool for both primitive creation and
  execution. This allows overlapping primitive creation time with
  primitive execution. As a result, multiple primitives will be
  submitted from a threadpool to that same threadpool, so parallelism
  will always be nested.

There are some properties that we can leverage on the oneDNN side:
- Primitive and memory object lifetimes are maintained by the thunk
  runtime while execution happens, so there is no need to reference-count
  oneDNN objects to extend their lifetime.
- Consecutive `parallel_for` calls happen in order (similar to SYCL
  in-order queues). We can leverage that to implicitly synchronize
  multiple parallel regions.

## API changes

### Waiting on threadpool task completion

The external threadpool interface that users implement already exposes
flags that inform the oneDNN implementation of the threadpool's
properties. The only flag exposed so far is the `ASYNCHRONOUS` flag,
which, when set, implies that the threadpool interface implementation has
a non-blocking implementation of the parallel method.

However, until now, when the threadpool interface implementation was
`ASYNCHRONOUS`, oneDNN synchronized explicitly inside the library, and we
plan to remove that synchronization. With that change, we need to add a
`wait()` method to the threadpool interface so that we can properly handle
waiting through the threadpool in the `stream::wait()` method.

Additions to `dnnl_threadpool_iface.hpp`:
```cpp
struct threadpool_iface {
    // Does nothing if SYNCHRONOUS, waits for all jobs if ASYNCHRONOUS.
    virtual void wait() = 0;
};
```

### Dependency tracking

Technically, we need a way to express dependencies between different
`parallel_for` calls, for example through events or another dependency
tracking mechanism. However, because the thunk runtime provides implicit
in-order synchronization within a threadpool, we can leverage that
instead.

Two options were considered:
- (recommended) No API extension is needed. We document that the
  `ASYNCHRONOUS` flag also implies in-order execution of consecutive
  `parallel_for` calls.
- Expose a new `IN_ORDER` flag and require it to be set whenever
  `ASYNCHRONOUS` is set.

**Resolution:** The recommended option was productized. No new flag was
added; the shipped `threadpool_iface` exposes only the `ASYNCHRONOUS`
flag, and the in-order property of consecutive `parallel_for` calls is
part of its documented contract.

### Asynchronous verbose profiling

Verbose `profile` mode was historically synchronous (see the note below).
To keep it working with an asynchronous threadpool without introducing
deadlocks, a completion-event interface was added to
`dnnl_threadpool_iface.hpp` together with a `get_event()` method on the
threadpool interface. oneDNN's verbose profiler polls these events to
determine when deferred execution has finished and to extract timing
information.

```cpp
/// Completion event interface for async threadpool work.
struct threadpool_event_iface_t {
    virtual ~threadpool_event_iface_t() = default;
    /// Returns true if the associated work has completed.
    virtual bool is_complete() const = 0;
    /// Blocks until completion.
    virtual void wait() const = 0;
    /// Measured execution time in milliseconds. Valid only after completion.
    virtual double exec_time_ms() const = 0;
};

struct threadpool_iface {
    /// Returns a single completion event for the most recently submitted
    /// primitive dispatch, or nullptr if profiling is disabled or
    /// unsupported.
    virtual std::shared_ptr<threadpool_event_iface_t> get_event() {
        return nullptr;
    }
};
```

## Internal changes

The main idea behind supporting asynchronous execution in the oneDNN
implementation is to tie the lifetime of temporary objects used during
execution to the lifetime of the `std::function` object (often a lambda)
submitted to the threadpool. We can then transfer ownership of this
`std::function` to the threadpool implementation, which keeps the object
alive while it is being executed.

### Managing std::function lifetime

This change can be localized to `src/common/dnnl_thread.hpp`, where we
need to:
- When redirecting the internal `parallel*` helpers to the threadpool
  implementation's `parallel_for`, stop capturing variables by reference
  in the lambda and capture them by copy instead. In general, these are
  just `dim_t` variables, so there is no meaningful additional overhead.
- Transfer ownership of these function objects using move semantics.

This also adds a requirement on the threadpool interface implementation to
properly handle the lifetime of the lambda closure.

### Managing local variable lifetime

oneDNN implementations currently use quite a few variables allocated on
the stack of the main thread and then pass them by reference to the lambda
capture in `parallel_for`. However, when `parallel_for` is asynchronous,
parallel tasks might access these variables after the main thread has
already exited the function scope (and hence freed those variables).

Consequently, the oneDNN CPU implementation can no longer pass
stack-allocated variables to a lambda capture by reference. A few
solutions are available:
- Allocate the variables on the heap behind a `shared_ptr` and pass it by
  copy to the lambda capture.
- Declare the variables directly inside the lambda (this is, for example,
  the preferred approach for `DEFINE_ARG_SCALES_BUFFER`). This duplicates
  initialization, but when initialization is cheap it simplifies the logic
  and removes synchronization overhead.
- Pass stack variables by copy when they are small, when copying is
  cheaper than initialization, or when a shared resource would otherwise
  require handling concurrent access.

### Managing execution context lifetime

In oneDNN internals, user-provided values are captured at execution time
inside a context structure (see
[here](https://github.com/uxlfoundation/oneDNN/blob/f0d20cd39c101a16df8c40aa2baa47d1908ac3fc/src/common/primitive_iface.cpp#L201)).

Two options were considered:
- Allocate this structure on the heap instead of the stack and free it
  asynchronously in `stream->after_exec_hook()` by submitting an extra
  `parallel_for` call with a single task.
- Use a smart pointer (`std::shared_ptr`) and tie the context lifetime to
  the lambda capture lifetime.

**Resolution:** The second option was productized. `exec_ctx_t` now holds
its state through a `std::shared_ptr<exec_ctx_impl_t>` (see
`src/common/primitive_exec_types.hpp`). When a lambda submitted from a
`parallel` call captures the context by copy (or calls one of its
methods), it dereferences the `shared_ptr` and increments its reference
count, keeping the underlying implementation members alive for as long as
the asynchronous tasks need them. This avoids submitting extra jobs to the
threadpool and works with library components that issue `parallel_for`
calls without manipulating context objects (e.g. the graph component).
Related to this, resources whose lifetime must match the context (such as
the scratchpad grantor) are now owned by `exec_ctx_t`, and the
`memory_arg_t` abstraction reference-counts the underlying memory objects
to prolong their lifetime.

### Synchronization

To avoid deadlocks, oneDNN must also not perform any waits or barriers
inside implementations that are compatible with the threadpool runtime.
oneDNN implementations already comply with this, and we will have to
maintain the property.

#### Note on verbose mode

oneDNN verbose `profile` mode was historically synchronous and relied on
`stream::wait()` (see
[code](https://github.com/uxlfoundation/oneDNN/blob/3f31492d3cd765b7a3e313ab0f86bbe1a6493c8e/src/common/primitive_iface.cpp#L100-L103)).
Keeping this behavior with asynchronous threadpool support would introduce
a deadlock when primitives are submitted within the pool they run in.

**Resolution:** Verbose `profile` mode was made asynchronous rather than
disabled. When the stream is backed by an asynchronous threadpool, oneDNN
no longer blocks on `stream::wait()`. Instead, a per-thread verbose
profiler retrieves a completion event via `threadpool_iface::get_event()`
after dispatch, records the pending primitive, and later collects the end
timestamp and prints the profiling record once the event reports
completion (see `src/cpu/verbose_profiler.{hpp,cpp}` and
`src/cpu/cpu_stream.hpp`). Threadpools that do not implement `get_event()`
fall back to returning `nullptr`, in which case profiling is a no-op.

### No computation on the main thread

The main thread can no longer be used to carry out computation after a
parallel region, as there is no guarantee that the parallel tasks have
completed. In particular, running a parallel reduction in a parallel
region and then reducing the partial sums on the main thread is no longer
possible: that reduction of partial sums must be submitted to the
threadpool with a `parallel_for` call containing a single task.

## Validation

Validation of the asynchronous runtime relies on several assumptions:
- All tests keep their primitives and memory objects alive until execution
  completes. To maintain this property, chaining the `execute(...)` method on
  a temporary primitive object is no longer possible. Additionally, an explicit
  `stream.wait()` is required wherever one was not already present.
- Exit points from parallel interfaces (tests rely on internal parallel calls)
  must be synchronized through a `wait()` interface, whether the stream's (when
  available) or the threadpool's.
- benchdnn uses a dedicated `stream_staller_t` abstraction to stall execution
  until all work has been submitted. This consistently reproduces the situation
  where the submitter goes out of scope during execution, reliably catching
  out-of-scope variables and objects.

Debugging these issues is easier with Clang AddressSanitizer, which pinpoints
out-of-scope variable accesses when they occur.
