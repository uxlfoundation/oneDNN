# Proposal for an Asynchronous Verbose Mode

## Background and Motivation

In the oneDNN verbose mode, the current approach to keeping track of kernel execution times relies on a host-side timing approach using `stream.wait()` calls. 
While simple, this approach introduces the following drawbacks, especially in the context of GPU kernel profiling:
- **Synchronization Overhead**: Measuring execution times on the host incurs synchronization costs when waiting for the GPU kernels to finish. For fast kernels, this overhead can also surpass the kernel execution time itself.
- **Timing Bubbles**: Latency between sequential kernels further distorts the actual execution window, making it difficult to isolate pre-kernel timings.
- **Blocking Semantics**: The use of `stream.wait()` is inherently blocking, which undermines performance in multi-stream or asynchronous execution environments.

This results in a discrepancy between measured and true kernel times, limiting the utility of verbose logging data for accurate perfromance profiling.

## Objectives
The goal of this proposal is to introduce an *asynchronous verbose mode* that enables non-blocking, accurate tracking of kernel executions with minimal synchronization latencies. The new mechanism will be implemented with the following approach: 
- Implement an event-polling mechanism that periodically checks for completed primitives without blocking execution.
- Leverage device-measured timing data from SYCL/OpenCL events to ensure accurate profiling information.
- Maintain compatibility with both regular execution and SYCL graph recording modes.
- Provide graceful fallback to synchronous mode when asynchronous profiling is not supported.

This design provides accurate execution timings, minimizes host-side interference, and keeps verbose logging compatible with both OpenCL and SYCL backends across all execution modes.


## Proposal
The new asynchronous verbose mechanism has the following design elements:

### Event-Polling Based Architecture
- A new `verbose_profiler_t` class manages a map of primitive execution events and their associated profiling data.
- The profiler operates independently from the main execution thread, polling for event completion without blocking stream operations.
- The profiling data for each primitive is queued in the order of execution, and its associated events are tracked until completion.
- Completed primitives are detected during periodic polling and logged asynchronously using the stored profiling data.

### Device-Measured Timing Data Acquisition
- The profiler queries device-measured execution times directly from SYCL/OpenCL events using:
    - `event.get_profiling_info<command_start/end>()` for SYCL runtimes.
    - `clGetEventProfilingInfo()` for OpenCL runtimes.
- Timing data is aggregated across multiple kernels within a single primitive execution.
- No host-side synchronization is required during the polling process.

### Stream Integration and Hooks
- Modified stream hooks (`before_exec_hook()` and `after_exec_hook()`) manage primitive stamping and completion checking.
- The profiler integrates seamlessly with existing stream execution without modifying primitive implementations.

### Asynchronous Completion Detection
- Event completion is checked using non-blocking `check_for_completed_primitives()` queries.
- Completed events are processed immediately, with timing information logged and event data cleaned up.
- The polling mechanism operates during natural execution flow without introducing additional callback threads.

### Fallback and Compatibility
- Automatic detection of profiling support ensures compatibility across different GPU runtimes.
- Graceful fallback to synchronous verbose mode when event profiling is unavailable.

## Usage and API Introduction
The implementation will be added as a functionality that is enabled during run-time whenever `DNNL_VERBOSE` is set to print verbose profiling info:
```bash
DNNL_VERBOSE=profile_exec ./examples/primitives-matmul-cpp gpu
```

A PoC implementation is available for both the OpenCL and Intel SYCL GPU backends.

**PoC Implementation**: [[link](https://github.com/uxlfoundation/oneDNN/pull/5102)]

### New Stream API
The implementation introduces new methods in `stream_t` for verbose profiler management:

```cpp
class stream_t {
public:
    // Stream execution hooks for profiler integration
    virtual void before_exec_hook();
    virtual void after_exec_hook();

    // Check if asynchronous verbose profiling is supported and enabled
    virtual bool is_verbose_profiler_enabled() const;
    
    // Execute verbose profiler for current primitive
    virtual status_t run_verbose_profiler(
        std::string &pd_info, double start_ms, uint64_t component) const = 0;
};
```

### New Verbose Profiler API
A new verbose_profiler_t class provides the core asynchronous profiling functionality:
```cpp
struct verbose_profiler_t {
    // Constructor with stream association
    verbose_profiler_t(const stream_t *stream);
    
    // Container to store profiling data for each primtive
    struct prim_profile_data_t {
        uint64_t component_kind_ = 0;
        double start_ms_ = 0.0;
        std::string pd_info_;
        std::vector<std::shared_ptr<xpu::event_t>> prim_events_;
    };

    // Primitive execution tracking
    void register_event(const std::shared_ptr<xpu::event_t> &event);
    void update_event_list();
    status_t add_to_pending_primitive_list(
            double start_ms, const std::string &pd_info, uint64_t component);
    
    // Asynchronous completion checking
    void check_for_completed_primitives();
    void wait_for_pending_primitives();
    
    // Backend-specific implementations
    virtual status_t get_aggregate_exec_time(size_t index, double &duration_ms) const = 0;
    virtual bool is_event_complete(const std::shared_ptr<xpu::event_t> &event) const = 0;
    virtual void wait_for_event_completion(const std::shared_ptr<xpu::event_t> &event) const = 0;

    // State management
    bool is_active() const;
    void start_profiling();
    void pause_profiling();
    void reset();

    // destructor logic for unlogged primitives at end of stream
    void cleanup();
};
```

### Runtime Usage 
The asynchronous verbose mode is automatically enabled when verbose profiling is requested and supported:

```bash
# Enable asynchronous verbose profiling
DNNL_VERBOSE=profile_exec ./examples/primitives-matmul-cpp gpu
```

### Implementation Features

- **Automatic Detection**: Runtime detection of profiling support with fallback to synchronous mode.
- **Multi-Backend Support**: Unified API with backend-specific implementations for SYCL and OpenCL.
- **Performance Optimization**: Minimal overhead through efficient event polling and cleanup.

### References

- `stream_profiler` Profiling API [[link](https://github.com/uxlfoundation/oneDNN/pull/1642)]
- oneDNN Verbose Mode [[link](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)]
- oneDNN Performance Profiling Examples [[link](https://uxlfoundation.github.io/oneDNN/page_performance_profiling_cpp.html)]

(EOD)