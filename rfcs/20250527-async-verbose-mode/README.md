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
- Leverage the existing `stream_profiler_t` infrastructure to ensure that verbose timing matches with device-reported values - this also helps ensure consistency with `benchdnn` which uses the same profiler for performance analysis. 
- Attach callbacks to the output (last) event of each primitive, so the verbose information is printed in a non-blocking fashion upon primitive execution without stalling the stream.
- Revert to the synchronous verbose mode when profiling cannot be enabled or event callbacks are unsupported.

This design provides accurate execution timings, minimizes host-side interference and keeps verbose logging compatible with both OpenCL and SYCL backends.

## Proposal
The new asynchronous verbose mechanism has the following design elements:

### `stream_profiler`-assisted Timing Data Acquisition
- The [`stream_profiler_t`](https://github.com/uxlfoundation/oneDNN/pull/1642) profiling API provides an experimental functionality to record device-measured, event execution times ensuring profiling accuracy and minimizing host-side interference.
- A `stream_profiler_t` instance instantiated on the GPU stream can directly query start and end times for timestamps from the device using
    - `clGetEventProfilingInfo()` for OpenCL runtimes.
    - `event.get_profiling_info()` for SYCL runtimes.
- The same functionality can be adopted to compute the primitive execution times without relying on `stream.wait()` calls.

### Asynchronous Verbose Printing 
- Instead of attaching raw callbacks, verbose printing is tied to profiler event completion.
- When the profiler finalizes timing data for a primtive, it invokes the logging path in a non-blocking fashion.
- The main execution thread is thus not stalled - verbose output happens when events naturally complete.

### Fallback Retrieval

- If profling cannot be enabled (e.g. for L0 runtimes), asynchronous verbose mode falls back to the legacy blocking implementation.
- This guarantees function verbose output in all cases, with optimal accuracy only when profiling is available.

## Usage and PoC Implementation
The implementation will be added as a functionality that is enabled during run-time whenever `DNNL_VERBOSE` is set to print verbose profiling info:
```bash
DNNL_VERBOSE=profile_exec ./examples/primitives-matmul-cpp gpu
```

A PoC implementation is available for both the OpenCL and SYCL GPU backends.

**PoC Implementation**: [[link](https://github.com/uxlfoundation/oneDNN/pull/3055)]

### References

- `stream_profiler` Profiling API [[link](https://github.com/uxlfoundation/oneDNN/pull/1642)]
- oneDNN Verbose Mode [[link](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)]
- oneDNN Performance Profiling Examples [[link](https://uxlfoundation.github.io/oneDNN/page_performance_profiling_cpp.html)]

(EOD)