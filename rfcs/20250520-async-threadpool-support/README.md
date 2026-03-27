# Proposal to support asynchronous threadpool runtime

## Background

XLA is switching to a new threapool runtime on CPU. In particular:
- it has an asynchronous `parallel_for` implementation, meaning the
  `parallel_for` function call returns while submitted tasks are still
  running on the threadpool.
- XLA will have a single threadpool for both primitive creation and
  execution. This allows to overlap primitive creation time with
  primitive execution. As a result, multiple primitives will be
  submitted from a threadpool to that same threadpool, so parallelism
  will always be nested.

We have some properties that we can leverage on oneDNN side:
- Primitive and memory object lifetimes will be maintained by thunk
  runtime while execution happens: no need to reference count oneDNN
  objects to extend their lifetime.
- consecutive `parallel_for` calls happen in order (similar to sycl
  in\_order queues). We can leverage that to implicitly synchronize
  multiple parallel regions.

## API changes

### Waiting on threadpool tasks completion

Currently the external threadpool interface that users need to
implement already exposes flags, to inform oneDNN implementation of
the threadpool implementation properties.  The only flag we expose as
of now is the `ASYNCHRONOUS` flag, which when set, implies that the
threadpool iface implementation has a non-blocking implementation of
the parallel method.

However, until now, when the threadpool iface implementation was
ASYNCHRONOUS, we used to explicit synchronize inside oneDNN library,
and we plan on removing that synchronization. With that change, we
need to add a wait() method to the threadpool iface, so that we can
properly handle waiting through threadpool in stream:wait() method.

Additions to threadpool_iface.hpp:
```c++
struct threadpool_iface {
     // Does nothing if SYNCHRONOUS, waits for all jobs for ASYNCHRONOUS 
     virtual void wait() = 0;
}
```

### Dependency tracking

Technically, we need to expose events or some dependency tracking
mecanism to express dependencies between different `parallel_for`
calls. However, because thunk runtime has implicit synchronization in
a threadpool, we can leverage that.

Here there are two options:
- (recommended) no API extension is needed. We need to document that the ASYNCHRONOUS flag implies this property
- we expose a new `IN_ORDER` flag, and we would require it to be set when ASYNCHORNOUS is set.

## Internal changes

The main idea to support asynchronous execution in oneDNN
implementation will be to tie the lifetime of temporary objects during
excecution to the closure of the std::function object (often a lambda)
that will be submited to the threapool. We then can transfert the
ownership of this std::function to the threadpool implementation, that
will maintain the std::function object alive while it is beeing
executed.

### Managing std::function lifetime

Here the change can be local to src/common/dnnl_thread.hpp, where we need to:
- when redirecting internal `parallel*` to threadpool implemenation
  `parallel_for`, we need to stop capturing references to variables in
  lambda, but copy those variables. In general, those are just `dim_t`
  variables and should have no additional overhead.
- we should move ownership of these function objects through std::move
  semantic.

Here this adds a requirement to the threadpool iface implemenation to
properly handle the lifetime of the lambda closure as well.

### Managing local variable lifetime

oneDNN implemenations currently use quiete a few variables allocated
on the stack of the main thread, and then pass those by reference to
the lambda capture in `parallel_for`. However, when parallel_for is
asynchronous, parallel tasks might access these variables while the
main thread already exited the function scope (and hence freed those
variables..

Hence, oneDNN CPU implementation cannot pass stack allocated variables
to a lambda capture by reference anymore. A few solutions are::
  - to allocate those on the heap behind a shared_ptr, which would be
    passed by copy to lambda capture.
  - to directly declare those variables inside the lambda (this is for
    example the preference way for `DEFINE_ARG_SCALES_BUFFER`). This
    introduces duplication of initialization but when initilization is
    cheap, it simplifies logic (and removes overhead of
    synchornization
  - to pass those stack variables by copy when they are small, or when
    it is cheaper to copy than to initialize those, or to handle
    parallel access to a shared ressource.

### Managing execution context lifetime

In oneDNN internals, we capture user passed values at execution time
inside a context structure (see
[here](https://github.com/uxlfoundation/oneDNN/blob/f0d20cd39c101a16df8c40aa2baa47d1908ac3fc/src/common/primitive_iface.cpp#L201)).

There are two options here:
- allocate this structure on the heap, instead of the stack, and free
  it asynchronously in `stream->after_exec_hook()`, by submiting an
  extra `parallel_for` call with a single task.
- use smart pointer (`std::shared_ptr`) and tie the context lifetime to the lambda capture
  lifetime.
  
The POC branch currently implements the first option as the changes
are more localized. However we might need to move to the second option
as it does not require extra jobs to be submitted to the threadpool,
and is compatible with components in the library that submit
parallel_for calls but do not manipulate context objects (e.g. graph
component).

### Synchronization and nested parallelism

Because primitives will be submitted from and to the same threadpool,
oneDNN need to enable nested parallelism, otherwise, all primitive
executions would be sequential.

To avoid deadlocks oneDNN would also not be able to do any wait or
barriers inside implementations compatible with threadpool
runtime. oneDNN implementations already comply with that but we will
have to maintain this property.

#### Note on verbose mode

Currently, oneDNN verbose `profile` mode is synchronous and uses
`stream::wait()` (see
[code](https://github.com/uxlfoundation/oneDNN/blob/3f31492d3cd765b7a3e313ab0f86bbe1a6493c8e/src/common/primitive_iface.cpp#L100-L103)). Keeping
this behavior with async threadpool support would introduce deadlock
when primitives are submitted within a pool they run in.  Here we have
to make verbose `profile` mode asynchronous (e.g. by submitting a task
for collecting end timestamp and printing), and disable verbose
`profile` mode in the meantime.

### No computation on main thread

Main thread can no more be used to carry computation after a parallel
region, as there is no guarentee of completion of the parallel
tasks. In particular, running a parallel reduction in a parallel
region, and then reduce the partial sums in the main thread is no more
possible: that reduction of partial sums will have to be submitted to
the threadpool with a `parallel_for` call with a single task.

## Validation

On benchdnn side, we will have two threadpool iface implementations:
- one asynchronous implementation passed to primitives under
  test. This implementation will be identical to the one used in thunk
  runtime for oneDNN integration.
- one synchronous implementation passed to benchdnn reference
  implementation and utilitity function (data filling, comparison,
  ...). This will allow to avoid intrusive and complex changes to
  benchdnn.

To check for correctness, running those benchdnn tests under clang
address sanitizer will be necessary to catch occurences where freed or
out of scope variables are accessed within parallel regions.
