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
- Adding a `threadpool_event_t` member to the profiling data container, `prim_profile_data_t` based on the state of `DNNL_THREADPOOL`. 
- Overloading methods operating on XPU events, `xpu::event_t` to accept `threadpool_event_t`.

**Drawbacks**: While simpler in implementation, this approach conflates two unrelated runtimes in a single class. Conditional compilation guards inside `prim_profile_data_t` create implicit coupling between XPU and threadpool build configurations. This also does not scale cleanly to additional runtimes.

### Proposal 2: Common Base `verbose_profiler_t` with Runtime Subclasses (Recommended)
This approach defines a runtime-agnostic base `verbose_profiler_t` in the `dnnl::impl` namespace that holds the polling workflow and profiling data attributes without any XPU-specific event types. The existing XPU profiler becomes a subclass, retaining its current behavior unchanged. A new threadpool subclass is introduced alongside it, replacing XPU `xpu::event_t` usage with `threadpool_event_t`.

This follows the established `stream_impl_t` / `engine_impl_t` pattern and is the recommended approach. The base class declares the event-dependent methods (`get_aggregate_exec_time`, `is_event_complete`, `wait_for_event_completion`) as pure virtuals with no event type in their signatures, leaving each subclass to resolve the event type internally. Since `threadpool_event_t` can be defined independently of `xpu::event_t`, no common event base is required.

**Drawbacks**: This requires refactoring the existing `verbose_profiler_t` structure to extract XPU-specific members into the XPU subclass, carrying a bounded but non-trivial risk of regression in the GPU profiler path.

## Implementation — API Extensions (for Proposal 2)
The proposed asynchronous verbose mechanism has the following design elements:

### Event API — `threadpool_event_t` in `threadpool_interop` namespace
- A new event type is introduced to represent the completion state of a single threadpool primitive dispatch. It is defined independently of the `xpu::event_t` abstraction in the `threadpool_interop` namespace:
```cpp
namespace dnnl {
namespace threadpool_interop {

/// Completion event interface for async threadpool work. oneDNN's verbose
/// profiler polls these to determine when deferred execution has finished
/// and to extract timing information.
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
    // Existing methods ...

    /// Returns a completion event for the most recently submitted work.
    /// oneDNN's verbose profiler polls this to detect completion and extract
    /// timing. Returns nullptr if profiling is disabled or unsupported.
    virtual std::shared_ptr<threadpool_event_iface_t> get_event() {
        return nullptr;
    }
}

} // namespace threadpool_interop
} // namespace dnnl
```

- Borrowing from the `threadpool_event_iface_t` definitions in [[link]](https://github.com/uxlfoundation/oneDNN/pull/5757), the interface defines a single event for each primitive execution whose completion is polled for duinrg profiling operations.

###  Breakdown of `verbose_profiler_t` mechanism

#### Common Base `verbose_profiler_t`
The base class retains the polling workflow and profiling data containers without any runtime-specific event types:
```cpp
namespace dnnl {
namespace impl {
// The verbose profiler logs primitive profiling information using device-
// measured execution times without host-to-device synchronization overhead or
// blocking stream.wait() calls. It operates asynchronously by polling device
// events to track primitive completion status.
// During primitive execution, the profiler groups and registers kernel events
// for each primitive with the associated profiling metadata. During each
// primitive post-exec hook, it polls previously registered events to identify
// completed primitives and logs their timing info. Pending primitives remain
// in the profiling data containers (different for different runtimes) list
// until detected as complete in subsequent polling cycles.
// During profiler destruction, any remaining primitives are checked and
// waited for to ensure no executions are left unlogged.
// This profiler is intended to be thread-local via thread_local_storage_t,
// ensuring thread-safety for multi-threaded execution environments. Each
// thread maintains its own profiler instance and event tracking state,
// operating independently from other stream profilers during primitive
// execution.
struct verbose_profiler_t {
    verbose_profiler_t(const stream_t *stream)
        : stream_(stream), active_(true) {}

    virtual ~verbose_profiler_t() = default;

    // Pausing capabilities are added to allow skipping event profiling
    // queries when they are temporarily unavailable.
    // These methods check and update profiler status where such force-pausing
    // is required. Pausing action is localized to each thread for multi-
    // threaded execution
    bool is_active() const { return active_; }
    void start_profiling() { active_ = true; }
    void pause_profiling() { active_ = false; }

    // The profiler operates through a multi-step event tracking workflow:
    // 1. stream->before_exec_hook() calls update_event_list()
    //    to add a new entry for the current primitive. Since there can be
    //    multiple event registration calls, this spot is a guaranteed single
    //    call for the coming primitive.
    // 2. During primitive execution, register_event() (defined in subclasses)
    //    adds device events to the latest primitive entry corresponding to the
    //    number of invoked kernels.
    // 3. add_to_pending_primitive_list() stores profiling metadata
    //    for the registered primitive
    // 4. stream->after_exec_hook() calls check_for_completed_primitives()
    //    to poll events and log completed primitives
    // 5. Incomplete primitives are not removed until detected as
    //    complete in future polling cycles
    // This asynchronous workflow allows tracking multiple concurrent
    // primitives without blocking execution.
    virtual void update_event_list() = 0;

    // populates profiling metadata for the last primitive entry
    virtual void add_to_pending_primitive_list(
            double start_ms, const std::string &pd_info, uint64_t component)
            = 0;

    // Completed primitive executions are periodically checked and logged
    // during after_exec_hook() calls and during stream destruction.
    // The profiler does not wait for pending events to complete
    // and instead prints them at the next concurrent after_exec_hook()
    // call.
    virtual void check_for_completed_primitives() = 0;

protected:
    const stream_t *stream_;
    bool active_;

private:
    // This is invoked during profiler destruction to account for any
    // pending primitives that have not yet been logged.
    virtual void wait_for_pending_primitives() = 0;
};

} // namespace impl
} // namespace dnnl
```

#### XPU Subclass
The XPU subclass retains all existing GPU profiler behavior. XPU-specific
event storage is moved here from the former monolithic `verbose_profiler_t`:

```cpp
namespace dnnl {
namespace impl {
namespace xpu {

// XPU (OpenCL/SYCL/L0) specialization of verbose_profiler_t.
// Tracks primitive completion using device-side xpu::event_t handles.
// Device-measured execution times are retrieved via get_aggregate_exec_time()
// using runtime-specific event timestamp queries.
// Instantiated per-thread via thread_local_storage_t on the GPU stream.
struct verbose_profiler_t : public dnnl::impl::verbose_profiler_t {
    verbose_profiler_t(const stream_t *stream)
        : dnnl::impl::verbose_profiler_t(stream) {}

    virtual ~verbose_profiler_t() = default;

    struct prim_profile_data_t {
        uint64_t component_kind_ = 0;
        double start_ms_ = 0.0;
        std::string pd_info_;
        std::vector<std::shared_ptr<xpu::event_t>> prim_events_;
    };

    void update_event_list() override { profiling_data_.emplace_back(); }

    // appends primitive event to the last primitive entry in profiling_data_
    void register_event(const std::shared_ptr<xpu::event_t> &event) {
        if (!event || profiling_data_.empty()) return;
        profiling_data_.back().prim_events_.push_back(event);
    }

    // populates profiling metadata for the last primitive entry in
    // profiling_data_
    void add_to_pending_primitive_list(double start_ms,
            const std::string &pd_info, uint64_t component) override;

    // Completed primitive executions are periodically checked and logged
    // during after_exec_hook() calls and during stream destruction.
    // The profiler does not wait for pending events to complete
    // and instead prints them at the next concurrent after_exec_hook()
    // call.
    void check_for_completed_primitives() override;

protected:
    std::vector<prim_profile_data_t> profiling_data_;

    // destructor logic to check for unlogged primitives before
    // stream destruction
    void cleanup();

private:
    // This is invoked during profiler destruction to account for any
    // pending primitives that have not yet been logged.
    void wait_for_pending_primitives() override;

    void reset() { profiling_data_.clear(); }

    virtual status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const
            = 0;
    virtual bool is_event_complete(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;
    virtual void wait_for_event_completion(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl
```

#### Threadpool Subclass
```cpp
namespace dnnl {
namespace impl {
namespace cpu {

// CPU async threadpool specialization of verbose_profiler_t.
// Tracks primitive completion using threadpool_event_iface_t handles registered
// with the threadpool after each parallel_for() dispatch.
// Instantiated on the submission thread via thread_local_storage_t on the
// CPU stream. Each submission thread maintains its own profiling_data_,
// operating independently from other primitive threads.
// Using a thread_local_storage_t definition is still valid as the
// profiling operations via enqueue_primtive() and pre-exec/post-exec hooks
// are done through a single calling thread.
struct verbose_profiler_t : public dnnl::impl::verbose_profiler_t {

    verbose_profiler_t(const stream_t *stream)
        : dnnl::impl::verbose_profiler_t(stream) {}

    ~verbose_profiler_t() override { cleanup(); }

    // Holds profiling metadata and threadpool event for a single pending
    // primitive. Mirrors xpu::verbose_profiler_t::prim_profile_data_t
    // but uses threadpool_event_iface_t in place of xpu::event_t.
    struct prim_profile_data_t {
        uint64_t component_kind_ = 0;
        double start_ms_ = 0.0;
        std::string pd_info_;
        std::shared_ptr<dnnl::threadpool_interop::threadpool_event_iface_t>
                prim_event_;
    };

    void update_event_list() override { profiling_data_.emplace_back(); }

    // Sets the threadpool event handle for the last entry in profiling_data_.
    // Called from enqueue_primitive() after registering the primitive event
    // with the threadpool.
    // Unlike the XPU profiler which may register multiple events per
    // primitive (one per kernel), the threadpool profiler registers a
    // single event per primitive dispatch.
    void register_event(const std::shared_ptr<
            dnnl::threadpool_interop::threadpool_event_iface_t> &event) {
        if (!event || profiling_data_.empty()) return;
        profiling_data_.back().prim_event_ = event;
    }

    void add_to_pending_primitive_list(double start_ms,
            const std::string &pd_info, uint64_t component) override;

    void check_for_completed_primitives() override;

protected:
    std::vector<prim_profile_data_t> profiling_data_;

    // destructor logic to check for unlogged primitives before
    // stream destruction
    void cleanup();

private:
    void wait_for_pending_primitives() override;

    void reset() { profiling_data_.clear(); }

    status_t get_aggregate_exec_time(size_t index, double &duration_ms) const;
    bool is_event_complete(size_t index) const;
};

} // namespace cpu
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
The implementation will be added as a functionality that is enabled during run-time whenever `DNNL_VERBOSE` is set to print verbose profiling info and when the library is built supporting async threadpool:
```bash
DNNL_VERBOSE=1 ./tests/benchdnn/benchdnn -v5 --matmul 128x128:128x128
```
An Example PoC is presented in the following implementation: [[link](https://github.com/uxlfoundation/oneDNN/pull/5891)]

To enable verbose profiling to work with the threadpool CPU runtime, oneDNN requires the user to implement a threadpool event interface in addition to the threadpool interface. The `threadpool_event_iface_t` interface and an example implementation can be found in [`tests/test_thread.cpp`](https://github.com/uxlfoundation/oneDNN/blob/3fdfdf7031f461f71f857255b0879cabb5595de7/tests/test_thread.cpp):

```cpp
// Lightweight completion event exposed to oneDNN's verbose profiler.
// start_ns_ is stamped by the first parallel_for() worker via CAS to avoid
// races between workers. end_ns_ and complete_ are set by the AndThen
// callback registered in get_event() when the chain resolves.
class threadpool_event_t
    : public dnnl::threadpool_interop::threadpool_event_iface_t {
public:
    threadpool_event_t() = default;

    bool is_complete() const override {
        return complete_.load(std::memory_order_acquire);
    }

    void wait() const override {
        while (!complete_.load(std::memory_order_acquire)) {}
    }

    double exec_time_ms() const override {
        int64_t start = start_ns_.load(std::memory_order_relaxed);
        int64_t end = end_ns_.load(std::memory_order_relaxed);
        return static_cast<double>(end - start) * 1e-6;
    }

    void stamp_start(int64_t ns) {
        int64_t expected = 0;
        start_ns_.compare_exchange_strong(
                expected, ns, std::memory_order_relaxed);
    }

    // Called by the AndThen callback registered in get_event() when the
    // chain resolves. Records the end timestamp and sets complete_.
    // end_ns_ is written before complete_ to ensure exec_time_ms() is
    // valid when is_complete() returns true.
    void mark_complete() {
        end_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now()
                                      .time_since_epoch())
                              .count(),
                std::memory_order_relaxed);
        complete_.store(true, std::memory_order_release);
    }

private:
    // Completion flag set by mark_complete() in the AndThen callback.
    std::atomic<bool> complete_ {false};

    // Start timestamp in nanoseconds. Stamped by the first worker via CAS.
    std::atomic<int64_t> start_ns_ {0};

    // End timestamp in nanoseconds. Written by mark_complete() before
    // complete_ is set.
    std::atomic<int64_t> end_ns_ {0};
};
```


## Implementation Features
- Polling-driven completion tracking is carried out via `threadpool_event_t::is_complete()` during `after_exec_hook()` — no `stream->wait()` in the execution path.
- In-order primitive logging is guaranteed by front-to-first-incomplete iteration in `verbose_profiler_t::check_for_completed_primitives()` calls.
- Blocking cleanup at stream destruction via `wait_for_pending_primitives()` ensures no executions are left unlogged.
- The approach allows zero overhead when profiling is disabled since no profiler objects are allocated and the capability check is a single flag read.

### References

- `stream_profiler` Profiling API [[link](https://github.com/uxlfoundation/oneDNN/pull/1642)]
- oneDNN Verbose Mode [[link](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)]
- oneDNN Performance Profiling Examples [[link](https://uxlfoundation.github.io/oneDNN/page_performance_profiling_cpp.html)]
- Asynchronous Verbose Mode in oneDNN [[link](https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20250527-async-verbose-mode)]
- Using Threadpool-based Threading in oneDNN [[link](https://uxlfoundation.github.io/oneDNN/dev_guide_threadpool.html)]
- OpenXLA source code [[link](https://github.com/openxla/xla)]
- Verbose profiling hooks for asynchronous threadpool execution [[link](https://github.com/uxlfoundation/oneDNN/pull/5757)]

(EOD)