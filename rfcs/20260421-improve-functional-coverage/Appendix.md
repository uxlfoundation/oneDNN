# Introduction
This RFC is an appendix to the bullet of ~80% optimization coverage.
Benchdnn case generation tool PoC revealed several issues that require some additional measures that will be discussed below.
Issue: benchdnn provides a flexible way to skip test cases but doesn’t have instruments to analyze itemized skipped statistics.

Why is it an issue?

When a test case is generated, run and skipped, it’s hard to say what is the exact reason for skipping without digging into which is not an easy task itself –
dispatch verbose may return too generic messages, or report a multi-return-point function as a single spot without telling which condition exactly failed.
It is helpful to see the distribution among skipped cases to make sure that a generator is prevented from crafting invalid cases and helped to fine tune the amount of corner cases that are not supported.
However, it turned out that each driver has lengthy logic about what it supports and what not and adjusting the generator
to each and one of them and introducing a long-named list of reasons for skipped cases doesn't seem to be the best approach.
Besides that, that logic can cover bugs unintentionally, e.g. Grouped Matmul with post-ops on GPU return skipped because GPU supports only Source sparse descriptor (that’s what benchdnn says so far).
Additionally, it doesn't feel right that the library provides API that simply returns unimplemented for a given set of settings, unless documented.

# Proposal
With the last sentence in mind, even though it is a broad thought, the proposal is to drop most of the logic that makes benchdnn producing nice SKIPPED statuses for certain reasons.
Such proposal leads to the following consequences:
* Unimplemented points must be implemented in ref implementations to make the library provide the meaningful results. While it's not possible for all API pieces, it's possible to cover most of them.
* Exception cases (listed below) would still be skipped on benchdnn side as their implementation on the library side doesn't provide any benefits for end users.
  Alternatively, it can be restricted from API side (src/common/ directory) for a given backend. However, it would limit the sharing capability of same input files between backends which is opposite to the desired goal.
* A special place in API is taken by data types.
  According to the documentation, such support is declared to be hardware-dependent and mustn't be supported in reference implementation if hardware doesn't support it natively.
  Currently, benchdnn handles this skipping logic via library functions (special holes with exported symbols) to verify if the library supports a given data type or not after a pd returns unimplemented status.
  Instead, the proposal is to provide a new special status that would indicate that the library doesn't support a given data type,
  and benchdnn to accept it and skip unconditionally without doing extra verification on its side with the same library functions that report unimplemented when pd is created.
  This action delegates responsibility to the library which would also provide the user with a clearer message if data types were not supported.

The list of unsupported functionalities:
* Convolutions:
    - Winograd algorithm. This algorithm provides tentative advantage on a short number of platforms with low compute capabilities. Suggestion is to unconditionally let this algorithm be skipped if unimplemented.
    - Depthwise post-op. Its support is limited to old CPU platforms only. Same suggestion for this post-op kind.
* RNNs:
    - Support for RNN is sparse and may exclude an algorithm support completely for a backend, or combination clusters of algorithm plus specific data types support.
      Suggestion is to let RNN be unconditionally skipped as it's a seldomly requested and outdated functionality to be worried about.
* Matmuls:
    - `packed` sparse kind is supported for CPU with AMX unit for weights execution argument only. The suggestion is to entirely skip cases with this kind.
* Reorders:
    - Runtime dimensions and compensation buffers on GPU are not supported by design. They are skipped now and the suggestion is to continue this practice.

There's a PoC branch which implements all missing pieces in reference:
`dzarukin/benchdnn_remove_unimpl` in case you'd like to get familiar with all the changes that are triggered by *existing* testing coverage.
It's hard to say if new things emerge once the coverage going to be extended with a generation tool.

# Summary
Proposal 1:
```cpp
typedef enum {
    ...
    // The data type is not supported on a system.
    dnnl_data_type_not_supported = 11,
} dnnl_status_t;
```

This will be used in consolidated functions that are responsible for data type hardware checks.

Proposal 2: implement missing functionality that is cheap to add in reference implementations to align backends on behavior for the same set of settings.
