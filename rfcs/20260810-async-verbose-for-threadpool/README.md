# Proposal for Extending Asynchronous Verbose Mode for Threadpool Runtime

## Background and Motivation

Asynchronous verbose profiling is functional in oneDNN for OpenCL and SYCL GPU runtimes where the profiler tracks primitive executions using device-side events, polling for completion during `after_exec_hook()` calls without blocking the host thread. This allows accurate device-measured execution times to be reported without introducing synchronization overhead into the critical execution path. Extending the existing non-blocked profiling mechanism to asynchronous threadpool CPU streams addresses the following:
- **Accurate timing**: Host-side submission latency is not a meaningful proxy for primitive execution time. Completion-driven measurement captures the true execution duration.
- **Synchronization Overhead**: Introducing `stream->wait()` to fix accuracy would serialize execution and defeat the purpose of an asynchronous dispatch. A polling-driven approach avoids this.
- **Current Profiling Semantics**: Users running the same workload on GPU and async CPU Threadpool should receive profiling data of equivalent meaning, enabling reliable cross-backend performance comparisons.

## Objectives
The goal of this RFC is to implement asynchronous verbose profiling for CPU streams with Asynchronous Threadpool runtime, following the same polling-driven design already established for GPU runtimes. The implementation proceeds in the following steps:
- **Establishing Event Definitions** - The verbose profiler API tracks primitive completion through device event handles (`xpu::event_t`) which are defined for GPU runtimes. An equivalent event abstraction  (`threadpool_event_t`) is needed for the CPU threadpool runtime. The event type must support: (i) querying execution start and end times, (ii) querying completion status and (iii) blocking wait for completion.
- **Extending Verbose Profiler API for Async Threadpool** - The base `verbose_profiler_t` abstraction defines the polling workflow and the virtual methods that the backend-specific profilers must implement. A new CPU-specific subclass is introduced that implements these methods using the event definitions.
- **Profiler Initialization Based on Capability** - The verbose profiler is instantiated and activated only when the stream's threadpool supports the mechanisms required for completion tracking. Initialization logic checks for the necessary threadpool capabilities and sets profiler state accordingly.
- **Correctness checks and graceful fallback** - accounts for unsupported/invalid cases and switches to synchronous mode when asynchronous profiling is not supported.

This design provides accurate execution timings, minimizes host-side interference, and keeps verbose logging compatible with CPU Threadpool backends across all execution modes.

## Proposals

### Proposal 1: Extend XPU `verbose_profiler_t` Directly

Keep the existing `verbose_profiler_t` structurally intact and extend it with threadpool-specific members and method overloads. This includes:
- Adding a `threadpool_event_t` member to the profiling data container, `prim_profile_data_t` based on the sate of `DNNL_THREADPOOL`. 
- Overloading methods operating on XPU events, `xpu::event_t` to accept `threadpool_event_t`.

**Drawbacks**: While simpler in implementation, this apporach conflates two unrelated runtimes in a single class. Conditional compilation guards inside `prim_profile_data_t` create implicit coupling between XPU and threadpool build configurations. This also does not scale cleanly to additional runtimes.

### Proposal 2: Common Base `verbose_profiler_t` with Runtime Subclasses (Recommended)
This approach defines a runtime-agnostic base `verbose_profiler_t` in the `dnnl::impl` namespace that holds the polling workflow and profiling data attributes without any XPU-specific event types. The existing XPU profiler becomes a subclass, retaining its current behavior unchanged. A new threadpool subclass is introduced alongside it, replacing XPU `xpu::event_t` usage with `threadpool_event_t`.

This follows the established `stream_impl_t` / `engine_impl_t` pattern and is the recommended approach. The base class declares the event-dependent methods (`get_aggregate_exec_time`, `is_event_complete`, `wait_for_event_completion`) as pure virtuals with no event type in their signatures, leaving each subclass to resolve the event type internally. Since `threadpool_event_t` can be defined independently of `xpu::event_t`, no common event base is required.

**Drawbacks**: This requires refactoring the existing `verbose_profiler_t` structure to extract XPU-specific members into the XPU subclass, carrying a bounded but non-trivial risk of regression in the GPU profiler path.

## Implementation — API Extensions (for Proposal 2)
The proposed asynchronous verbose mechanism has the following design elements:

### Event API — `threadpool_event_t` in `threadpool_interop` namespace
- A new event type is introduced to represent the completion state of a single threadpool primitive dispatch. It is defined indepndently of the `xpu::event_t` abstraction in the `threadpool_interop` namespace:
```cpp
namespace dnnl {
namespace threadpool_interop {

struct threadpool_event_t {
    // Non-blocking completion status query.
    // Used by verbose_profiler_t::is_event_complete() during polling.
    bool is_complete() const;

    // Returns the host-measured start time in milliseconds.
    // Recorded immediately before enqueue_primitive() on the submission thread.
    double start_ms() const;

    // Returns the host-measured end time in milliseconds.
    // Recorded by the completion callback on a threadpool worker thread.
    // Valid only after is_complete() returns true.
    double end_ms() const;

    // Blocking wait for completion.
    // Used only during profiler cleanup at stream destruction.
    void wait() const;

    // Called by the threadpool completion callback to signal completion
    // and record the end timestamp.
    void mark_complete();

    // Called on the submission thread immediately before enqueue_primitive()
    // to record the primitive start time.
    void set_start_ms(double ms);

private:
    // Atomic flag set by mark_complete() on a worker thread and polled
    // by is_complete() on the submission thread. acquire/release ordering
    // ensures end_ms_ is visible when completed_ is observed as true.
    std::atomic<bool> completed_ {false};

    // Host-measured start time recorded on the submission thread.
    std::atomic<double> start_ms_ {0.0};

    // Host-measured end time recorded by the completion callback.
    // Valid only after completed_ is true.
    std::atomic<double> end_ms_ {0.0};
};

} // namespace threadpool_interop
} // namespace dnnl
```

###  Breakdown of `verbose_profiler_t` mechanism

#### Common Base `verbose_profiler_t`
The base class retains the polling workflow and profiling data containers without any runtime-specific event types:
```cpp
namespace dnnl {
namespace impl {

// Runtime-agnostic base for asynchronous verbose profiling.
// Manages the pending primitive list and polling workflow independently
// of the underlying runtime event type. Defined in common with XPU and 
// threadpool subclasses providing runtime-specific event handling.
//
// Lifecycle:
//   before_exec_hook -> update_event_list()
//   enqueue_primitive -> register_event() (subclass)
//   run_verbose_profiler -> add_to_pending_primitive_list()
//   after_exec_hook -> check_for_completed_primitives()
//   destructor -> cleanup() -> wait_for_pending_primitives()
struct verbose_profiler_t {

    // Holds profiling metadata for a single pending primitive.
    // Event storage is managed by subclasses in parallel structures
    // to avoid coupling the base class to any runtime event type.
    struct prim_profile_data_t {
        double start_ms_ = 0.0;
        std::string pd_info_;
    };

    verbose_profiler_t(const stream_t *stream);
    virtual ~verbose_profiler_t() = default;

    // Returns true if the profiler is currently active.
    // Profiling may be paused when the stream does not support
    // event queries (e.g., queue has no profiling enabled).
    bool is_active() const;

    // Resumes profiling after a pause.
    void start_profiling();

    // Pauses profiling. update_event_list() still adds entries but
    // register_event() calls are suppressed, leaving prim_events_ empty.
    // Handled by the empty-event path in check_for_completed_primitives().
    void pause_profiling();

    // Adds a new empty entry to profiling_data_ for the incoming primitive.
    // Called once per primitive from before_exec_hook() to guarantee a
    // single entry point regardless of how many register_event() calls follow.
    void update_event_list();

    // Populates profiling metadata for the last entry in profiling_data_.
    // Called from run_verbose_profiler() after enqueue_primitive() returns.
    void add_to_pending_primitive_list(
            double start_ms, const std::string &pd_info);

    // Polls pending primitives for completion and logs those that are done.
    // Iterates from the front of profiling_data_ and stops at the first
    // incomplete primitive to preserve in-order logging.
    // Called from after_exec_hook() after each primitive enqueue.
    void check_for_completed_primitives();

protected:
    // Reference to the owning stream. Used by subclasses for
    // runtime-specific event queries.
    const stream_t *stream_;

    // Ordered list of pending primitive metadata entries.
    // Front entries are erased as primitives complete.
    std::vector<prim_profile_data_t> profiling_data_;

    // Whether the profiler is currently active.
    bool active_;

    // Called from subclass destructors to block on any remaining pending
    // primitives and ensure no executions are left unlogged before
    // stream destruction.
    void cleanup();

private:
    // Blocks on the last pending event then calls
    // check_for_completed_primitives() to log any remaining entries.
    // For in-order queues, waiting on the last event is sufficient since
    // all prior events will have completed when the last one completes.
    void wait_for_pending_primitives();

    // Clears profiling_data_ and any subclass event storage.
    void reset();

    // Returns the aggregate execution time in milliseconds for the primitive
    // at `index` in profiling_data_. Implemented by subclasses using
    // runtime-specific event timestamp queries.
    virtual status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const = 0;

    // Non-blocking completion check for the primitive at `index`.
    // Implemented by subclasses using runtime-specific event status queries.
    virtual bool is_event_complete(size_t index) const = 0;

};

} // namespace impl
} // namespace dnnl
```

#### XPU Subclass
The XPU subclass retains all existing GPU profiler behavior. XPU-specific event storage is moved here from the former monolithic `verbose_profiler_t`:

```cpp
namespace dnnl {
namespace impl {
namespace xpu {

// XPU (OpenCL/SYCL) specialization of verbose_profiler_t.
// Tracks primitive completion using device-side xpu::event_t handles.
// Device-measured execution times are retrieved via get_aggregate_exec_time()
// using runtime-specific event timestamp queries.
// Instantiated per-thread via thread_local_storage_t on the GPU stream.
struct verbose_profiler_t : public dnnl::impl::verbose_profiler_t {

    verbose_profiler_t(const stream_t *stream);
    ~verbose_profiler_t() override;

    // Appends a device event to the event list for the last primitive entry.
    // Called during primitive execution for each dispatched kernel.
    // Multiple events may be registered per primitive for multi-kernel
    // primitives; get_aggregate_exec_time() aggregates across all of them.
    void register_event(const std::shared_ptr<xpu::event_t> &event);

private:
    // Device event lists, parallel to profiling_data_.
    // Each entry holds the device events for the corresponding primitive.
    std::vector<std::vector<std::shared_ptr<xpu::event_t>>> event_data_;

    // Aggregates device-measured execution time across all events for the
    // primitive at `index` using runtime-specific timestamp queries
    status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const override;

    // Returns true if the last device event for the primitive at `index`
    // has completed. For in-order queues, completion of the last event
    // implies completion of all prior events for the same primitive.
    bool is_event_complete(size_t index) const override;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl
```

#### Threadpool Subclass
```cpp
namespace dnnl {
namespace impl {
namespace threadpool {

// CPU async threadpool specialization of verbose_profiler_t.
// Tracks primitive completion using threadpool_event_t handles populated
// by completion callbacks registered with the threadpool after each
// parallel_for() dispatch.
// Instantiated per-thread via thread_local_storage_t on the CPU stream.
struct verbose_profiler_t : public dnnl::impl::verbose_profiler_t {

    verbose_profiler_t(const stream_t *stream);
    ~verbose_profiler_t() override;

    // Appends a threadpool event to the event list for the last primitive
    // entry. Called from enqueue_primitive() after registering the
    // completion callback with the threadpool.
    void register_event(
            std::shared_ptr<threadpool_interop::threadpool_event_t> event);

private:
    // Threadpool event lists, parallel to profiling_data_.
    // Each entry holds the threadpool events for the corresponding primitive.
    // Unlike the XPU profiler, typically one event per primitive since
    // threadpool primitives issue a single parallel_for() dispatch.
    std::vector<std::vector<
            std::shared_ptr<threadpool_interop::threadpool_event_t>>>
            event_data_;

    // Computes execution duration as the delta between end_ms() and
    // start_ms() of the last threadpool event for the primitive at `index`.
    // Both timestamps are host-measured: start_ms recorded before
    // enqueue_primitive(), end_ms recorded by the completion callback.
    status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const override;

    // Returns true if the last threadpool event for the primitive at
    // `index` has been marked complete by its callback.
    // Non-blocking: reads the atomic completed_ flag on threadpool_event_t.
    bool is_event_complete(size_t index) const override;
};

} // namespace threadpool
} // namespace impl
} // namespace dnnl
```

### Stream Integration and Hooks
The threadpool stream is extended with the profiling methods and hook modifications, mirroring the OpenCL stream's verbose profiler integration: 
- Event completion is checked using non-blocking `check_for_completed_primitives()` queries.
- Completed events are processed immediately, with timing information logged and event data cleaned up.
- The polling mechanism operates during natural execution flow without introducing additional callback threads.

### Fallback and Compatibility
- Automatic detection of profiling support ensures compatibility across different runtimes.
- Graceful fallback to synchronous verbose mode when event profiling is unavailable.

## Usage and PoC
The implementation will be added as a functionality that is enabled during run-time whenever `DNNL_VERBOSE` is set to print verbose profiling info:
```bash
DNNL_VERBOSE=profile_exec ./examples/primitives-matmul-cpp gpu
```

An Example PoC is presented in the following implementation: [[link]()]

## Implementation Features
- Polling-driven completion tracking is carried out via `threadpool_event_t::is_complete()` during `after_exec_hook()` — no `stream->wait()` in the execution path.
- In-order primitive logging is guaranteed by front-to-first-incomplete iteration in `verbose_profiler_t::check_for_completed_primitives()` calls.
- Blocking cleanup at stream destruction via `wait_for_pending_primitives()` ensures no executions are left unlogged
- Graceful fallback to existing behavior occurs when `COMPLETION_CALLBACKS` is not advertised by the threadpool
- The approach allows zero overhead when profiling is disabled since no profiler objects are allocated and the capability check is a single flag read

### References

- `stream_profiler` Profiling API [[link](https://github.com/uxlfoundation/oneDNN/pull/1642)]
- oneDNN Verbose Mode [[link](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)]
- oneDNN Performance Profiling Examples [[link](https://uxlfoundation.github.io/oneDNN/page_performance_profiling_cpp.html)]

(EOD)