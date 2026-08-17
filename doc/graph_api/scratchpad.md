Scratchpad Management {#dev_guide_graph_scratchpad}
===================================================

## Introduction

Some compiled partitions require temporary memory (scratchpad) during execution
to store the intermediate results. The amount of space required for the
scratchpad depends on the compiled partition and its actual implementation.
oneDNN Graph API supports two different scratchpad management modes:

1. **User-managed scratchpad**: The user queries the required scratchpad size,
   allocates the buffer, and passes it to the execution call. This mode gives
   the user full control over scratchpad memory lifecycle, enabling buffer reuse
   across multiple executions and reducing allocation overhead.

2. **Library-managed scratchpad**: The library internally allocates and frees
   the scratchpad buffer during each execution call. This mode is simple to use
   but with its own limitations.

## User-Managed Scratchpad

To use user-managed scratchpad, follow these steps:

1. Query the scratchpad logical tensor from the compiled partition.
2. Check if a scratchpad is required (memory size > 0).
3. Allocate a tensor with the queried logical tensor descriptor.
4. Pass the scratchpad tensor to the execution call.

~~~cpp
// Compile the partition
dnnl::graph::compiled_partition cp = partition.compile(inputs, outputs, engine);

// Step 1: Query scratchpad requirements
dnnl::graph::logical_tensor scratchpad_lt = cp.get_scratchpad_logical_tensor();

// Step 2: Check if scratchpad is needed
dnnl::graph::tensor scratchpad_ts;
if (scratchpad_lt.get_mem_size() > 0) {
    // Step 3: Allocate scratchpad tensor
    void * user_scratchpad = user_allocate_method(scratchpad_lt.get_mem_size());
    scratchpad_ts = dnnl::graph::tensor(scratchpad_lt, engine, user_scratchpad);
}

// Step 4: Execute with user-managed scratchpad
cp.execute(stream, input_tensors, output_tensors, scratchpad_ts);
~~~

The user-managed scratchpad tensor follows the same rules and limitations as
input and output tensors. It must be created using the same engine as partition
compilation unless specified otherwise. If the same compiled partition is
executed in multiple threads concurrently, a separate scratchpad buffer must be
used per thread to ensure the thread safety. A user-managed scratchpad buffer
can be reused across multiple executions of the same or different compiled
partitions where the size fits.

## Library-Managed Scratchpad

By default, when no scratchpad tensor is provided to the execution call, the
library allocates the required scratchpad memory internally and frees it after
execution completes, through the allocator interfaces associated with engine
object. This mode requires no additional user action.

~~~cpp
// Compile the partition
dnnl::graph::compiled_partition cp = partition.compile(inputs, outputs, engine);

// Execute without scratchpad - library manages it internally
cp.execute(stream, input_tensors, output_tensors);
~~~

## Work with SYCL Graph Recording Mode

User-managed scratchpad is required when working with SYCL graph recording mode.

SYCL graph recording captures a sequence of kernel submissions into a graph
object that can be replayed multiple times. During recording, all memory buffers
accessed by the recorded kernels are bound to the graph. These buffers must
remain valid across all subsequent replays.

When library-managed scratchpad is used, the library allocates a temporary
buffer at execution time and frees it immediately after execution completes.
This is incompatible with SYCL graph recording.

To work correctly with SYCL graph recording mode, users must pass the
pre-allocated scratchpad tensor to the execute call during recording and keep
the scratchpad buffer alive across all replay iterations.

## API Reference

| API                                                                 | Description                                                   |
| :---                                                                | :-----------                                                  |
| @ref dnnl::graph::compiled_partition::get_scratchpad_logical_tensor | Returns the logical tensor describing the required scratchpad |
| @ref dnnl::graph::compiled_partition::execute                       | Executes with optional user-managed scratchpad                |
| @ref dnnl::graph::sycl_interop::execute                             | SYCL interop execute with user-managed scratchpad             |
| @ref dnnl::graph::ocl_interop::execute                              | OpenCL interop execute with user-managed scratchpad           |
