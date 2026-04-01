Memory Descriptor Freelist {#dev_guide_memory_desc_freelist}
============================================================

The C API memory descriptor creation functions allocate a `dnnl_memory_desc`
object for each successful call. To reduce allocator traffic on hot paths, oneDNN
may keep a **thread-local freelist** of released descriptors and reuse them on
subsequent creations on the **same OS thread**.

This behavior is transparent to the API: pointers returned from
`dnnl_memory_desc_create_*` must still be destroyed exactly once with
@ref dnnl_memory_desc_destroy.

## Run-time Controls

The maximum number of descriptors retained per thread is read **once** when the
library loads the translation unit that implements the memory descriptor API.
The primary environment variable name uses the `ONEDNN_` prefix; the legacy
`DNNL_` prefix is accepted as well (same resolution order as other oneDNN
runtime variables).

| Environment variable                         | Value        | Description                                                                 |
|:---------------------------------------------|:-------------|:----------------------------------------------------------------------------|
| `ONEDNN_MEMORY_DESC_FREELIST_CAPACITY`       | \<number\>   | Maximum descriptors pooled **per thread** (default **256**)                  |
| `DNNL_MEMORY_DESC_FREELIST_CAPACITY`         | \<number\>   | Same as above (checked if the `ONEDNN_` variable is not set)                |
| \                                            | 0            | Disable freelist pooling (each destroy returns memory to the heap)         |

If the value is negative or cannot be parsed as an integer, the default **256**
is used.

## Example

~~~sh
$ export ONEDNN_MEMORY_DESC_FREELIST_CAPACITY=512
$ ./your_application
~~~
