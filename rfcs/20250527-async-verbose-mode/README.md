# Proposal for an Asynchronous Verbose Mode

## Background and Motivation

In the oneDNN verbose mode, the current approach to keeping track of kernel execution times relies on a host-side timing approach using `stream.wait()` calls. 
While simple, this approach introduces the following drawbacks, especially in the context of GPU kernel profiling:
- **Synchronization Overhead**: Measuring execution times on the host incurs synchronization costs when waiting for the GPU kernels to finish. For fast kernels, this overhead can also surpass the kernel execution time itself. 
- **Timing Bubbles**: Latency between sequential kernels further distorts the actual execution window, making it difficult to isolate pre-kernel timings.
- **Blocking Semantics**: The use of `stream.wait()` is inherently blocking, which undermines performance in multi-stream or asynchronous execution environments.

This results in a discrepancy between measured and true kernel times, limiting the utility of verbose logging data for accurate perfromance profiling.

## Objectives

The goal of this proposal is to introduce an *asynchronous verbose mode* that enables non-blocking, accurate tracking of kernel executions with minimal synchronization latencies. The new mechanism will be implemented with the following apporach: 
- Leverage verbose trackers that monitor kernel execution by registering events directly on the stream.
- Collect executiom timings from the trackers via callbacks upon event completion - eliminating synchronization overheads between host and GPU.  
- If stream profiling is enabled, retrieve device-accurate kernel execution times by querying event profiling information.
- Use event-driven callbacks to print verbose profiling data in a non-blocking manner.
- Fall back to the default synchronous mode if the timing data cannot be retrieved asynchronously.

## Proposal
This proposal implements the above mentioned approach for an asynchronous, event-driven verbose profiling. Described below are the main elements of this design: the data structures to hold timing information, the process for registering trackers and callbacks and the behavior of the asynchronous callback function.

### Timing Data Management
- We introduce a dedicated data container to store, for each stream, the timing information for the events being tracked. This container will be managed internally by the stream and the profiling subsystem therein.

```cpp
    // container for holding timing data for asynchronous verbose mode
    struct async_timing_data_t {
        double start_ms = 0; // event start timestamp (ms)
        double end_ms = 0; // event end timestamp (ms)
        double duration_ms = 0; // event duration (ms)
        bool profiler_enabled = false; // whether stream profiling is enabled
        bool timing_stat = true; // event registration status
        std::string vinfo; // verbose info for the event
    };
```

- The sructure is responsible for encapsulating all timing-related data for a particular event, primitive or stream. 
- For each stream, the are two `async_timing_data` objects that keep track of profiling information: 
    
    1) `event_timing_data_`: to record individual event timings.
    2) `stream_timing_data_`: to record stream-wise timings.

### Tracker Registration

- After a kernel or relevant operation is enqueued, its completion event can be registered for asynchronous tracking using the following function:
```cpp
dnnl::impl::status_t register_async_tracker(event_t event);
```
- This function performs the following duties:
    - It stores the event that will be tracked for profiling purposes.
    - It sets the appropriate flags in the `timing_data_` container, dpending on whether profiling and tracking capabilities are enabled.
    - Most importantly, it registers a callback function, `async_tracker_callback` on the event to be invoked upon event completion. This callbak is responsible for gathering and recording profiling information as soon as the primitive operation completes, without requiring host-side polling or blocking.

### Callback Behavior

The event callback `async_tracker_callback` is a core component of the asynchronous mode. When the event signals completion, this callback is triggered as per the GPU runtime (OpenCL or SYCL). Its responsibilities include:
- capturing and updating the timing data (`start_ms`, `end_ms`, `duration_ms`) for the tracked event.
- handling both profiling-enabled and fallback (profiling-disabled or error) cases gracefully.
- printing or recording profiling information in a non-blocking fashion.
- advancing the tracker for the next event.

### Fallback Retrieval

To support robust error handling and user queries, we also provide a synchronous fallback API:

```cpp
    // This ensures that verbose the profiling info is printed for
    // the operation in case of failures / disabled functionalities
    // for the asynchronous mode. The operation then falls back to
    // using the default blocking stream.wait() calls. 
    dnnl::impl::status_t check_async_exec_times(std::string vinfo, double *start_ms);
```
- This function retrieves the last recorded timing values, allowing the system to deliver profiling information even if asynchronous tracking is unvailable or disabled.

## Usage and PoC Implementation
The implementation will be added as a functionality that is enabled during run-time whenever `DNNL_VERBOSE` is set to print verbose profiling info:
```bash
DNNL_VERBOSE=profile_exec ./examples/primitives-matmul-cpp gpu
```

A PoC implementation is available for both the OpenCL and SYCL GPU backends. The OpenCL implementation demonstrates the use of `clSetEventCallback` for non-blocking, event-driven timing and tracking whereas the SYCL API uses `host_task` to implement the asynchronous mode. 

**Reference**: [[link](https://github.com/uxlfoundation/oneDNN/pull/3055)]

## Multi-Kernel and Out-of-Order Queue Support
The proposed mechanism for the asynchronous mode will result in incorrect tracking for two scenarios: for out-of-order queues and when there are multiple kernels on queue. The following proposed extension can account for these cases:

1) For each stream, we maintain a list (`primitive_timing_data_`) of `async_timing_data_` objects (one per primitive) instead of a single `stream_timing_data_` instance. For the listed instances, The `vinfo` field serves as the primitive marker, associating each timing data object with its respective primitive.
2) As each event callback is triggered, the corresponding `primitive_timing_data_` object is updated with the new end time and duration. This ensures that each primitives's timing aggregrates all its relevant kernel events, even in out-of-order execution.
3) During stream destruction, timing data for each primitive is printed in the order the primitives were queued. Optionally, event polling can also be used to check for primitives whose events are complete, but it should be avoided as it does not account for missed late event submissions.

There remain two considerations for the proposed extension:
- The profiling info is printed in an aggregrated manner for all the primitives at the end of the stream. This can be cumbersome when there are many primitives executed on the stream.
- The `primitive_timing_data_` objects must be shared safely across all callbacks which requires atomic operations or proper synchronization to ensure thread safety in presence of concurrent calbacks.

(EOD)