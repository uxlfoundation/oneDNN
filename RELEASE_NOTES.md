# Performance Optimizations
## Intel(R) Architecture Processors
* Improved matmul performance on Intel(R) Xeon(R) processors with support for the Intel(R) AMX instruction set.
* Improved `fp16`/`bf16` depthwise convolution performance with `fp32` bias or `sum` post-op.
* Improved `bf16` pooling backpropagation performance.
* Improved binary post-op performance with `per_w` broadcast.

## Intel Graphics Products
* Improved performance on Intel(R) GPUs based on Xe3 architecture.
* Improved convolution performance on:
   * Intel® Arc™ Graphics for Intel® Core™ Ultra (Series 2, formerly Lunar Lake).
   * Intel® Arc™ B-series discrete graphics (formerly Battlemage).
* Improved `int8` matmul performance with zero-points support for source and weight tensors.
* Improved performance of the following subgraphs with Graph API:
	* Scaled Dot Product Attention (SDPA) with `int4` and `int8` KV cache.
	* SDPA with bottom-right implicit causal mask.
	* SDPA with head size between 257 and 512.
	* Grouped Query Attention (GQA) with 5D input tensors.

## AArch64-Based Processors
* Enabled BF16 forward-mode inner product via ACL. 170x and 160x speedups for BERT and AlexNet in torch compile-mode.
* Enabled post-ops in jit_sve_1x1 conv. Yields 1.2x-1.4x speedups for 1x1 kernel problems.
* Prefer jit_sve conv where faster. Yields 1.3x-3x speedups on average.

# Functionality
## Common
* Introduced select algorithm support in [binary primitive](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_binary.html). The functionality is implemented on CPUs and Intel(R) GPUs.

## Intel Graphics Products
* Introduced support for the [GenIndex](https://oneapi-src.github.io/oneDNN/v3.8/dev_guide_op_genindex.html) operation in Graph API.

## Intel(R) Architecture Processors
* Introduced support for `f32` convolution with `fp16` compressed weights.
* Enabled `int8`/`int4` compressed weight support in matmul primitive on Intel(R) CPUs. 

## Generic SYCL backend
* Introduced support for RNN Vanilla primitive forward propagation.
* Introduced support for inner product primitive backward propagation.
* Introduced support for group normalization primitive.
* Improved precision of inner product primitive with sum post-op for larger shapes.

## AArch64-Based Processors
* Enabled `fp16` support for JIT reorder kernels.
* Enabled static quantization in matmul operations.

## NVIDIA GPUs
* Introduced Graph API support.

# Usability
 
* Added support for Group Normalization primitive with [`ONEDNN_ENABLE_PRIMITIVE`](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html#onednn-enable-primitive) build option.
* Enabled support for ROCm 6 on AMD* GPUs.
* Improved CMake integration when installing oneDNN with the Nvidia backend enabled.

## AArch64-Based Processors
 * Default the number of threads to max for acl_threadpool. Fixes a crash in Tensorflow.
 * Fixed scratchpad being ignored for some GEMMs. Reduces memory and speeds up execution.
 * Fixed a bug in `fp32` reorders where ACL returned incorrect results.

# Validation
* Added benchdnn option [`--execution-mode`](https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/tests/benchdnn/doc/knobs_common.md#--execution-mode) to test oneDNN functionality with SYCL Graph record/execute mode.
* Extended benchdnn option [`--cold-cache`](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/knob_cold_cache.md) with support for cold TLB mode.
* Added benchdnn option `--bia-dt` to control bias data type for matmul, inner product, convolution, and deconvolution.
* Extended syntax of benchdnn `--dt` option in [Graph API driver](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/driver_graph.md) to manage data types of individual tensors in a pattern.

# Deprecated Functionality

# Breaking Changes
* Removed the experimental [Graph Compiler](https://uxlfoundation.github.io/oneDNN/v3.7/dev_guide_graph_compiler.html) backend.

# Thanks to Contributors
This release contains contributions from the project core team as well as Alexander Simonov @asimonov1, Denis @redradist, Dmitriy Ovchinnikov @inteldimitrius, Eliezer Weissmann @eliezerweissmann, Hubert Maciak @hmaciak, Ilya Lavrenov @ilya-lavrenov, James McGregor @Jmc18134, Marek Michalowski @michalowski-arm, Maria Zhukova @mzhukova, Orel Yehuda @yehudaorel, Ravi Pushkar @rpushkarr, Renato Barros Arantes @renato-arantes, Shreyas-fuj @Shreyas-fuj, Shu Chen @shu1chen, Viktoriia Gvozdeva @vgvozdeva, Yair Obodovsky @yair-obodovsky, hmaciak @hmaciak, jstachowintel @jstachowintel, zhangfei @zhangfeiv0, James McGregor @Jmc18134, Marek Michalowski @michalowski-arm, Renato Barros Arantes @renato-arantes.
