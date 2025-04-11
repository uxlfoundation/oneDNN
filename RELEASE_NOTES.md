# Performance Optimizations
## Intel Architecture Processors
* Improved matmul performance on Intel Xeon processors with Intel AMX instruction set support.
* Improved `fp16`/`bf16` depthwise convolution performance with `fp32` bias or `sum` post-op.
* Improved `bf16` pooling backpropagation performance.
* Improved binary post-op performance with `per_w` broadcast.

## Intel Graphics Products
* Improved performance for Intel GPUs based on Xe3 architecture.
* Improved convolution performance on Intel Arc Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake) and Intel Arc B-series discrete graphics (formerly Battlemage).
* Improved `int8` matmul performance with zero points for source and weight tensors.
* Improved performance of the following subgraphs with Graph API:
	* Scaled Dot Product Attention (SDPA) with `int4` and `int8` KV cache.
	* SDPA with bottom-right implicit causal mask.
	* SDPA with head size between 257 and 512.
	* Grouped Query Attention (GQA) with 5D input tensors.

## AArch64-based Processors
TBD
# Functionality
## Common
* Introduced select algorithm support in [binary primitive](https://uxlfoundation.github.io/oneDNN/v3.8/dev_guide_binary.html). The functionality is implemented on CPUs and Intel GPUs.

## Intel Graphics Products
* Introduced support for [GenIndex](https://oneapi-src.github.io/oneDNN/v3.8/dev_guide_op_genindex.html) operation in Graph API.

## Intel Architecture Processors
* Introduced support for `f32` convolution with `fp16` compressed weights.
* Enabled support for int8 or int4 compressed weights in matmul primitive. This functionality is implemented on Intel CPUs.

## NVIDIA GPUs
* Introduces Graph API support.

# Usability
 
* Added support for group normalization primitive with [ONEDNN_ENABLE_PRIMITIVE](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html#onednn-enable-primitive) build option.
* Enabled ROCm 6 support for AMD GPUs.

# Validation
* Added benchdnn option [`--execution-mode`](https://github.com/uxlfoundation/oneDNN/blob/rls-v3.8/tests/benchdnn/doc/knobs_common.md#--execution-mode) to test oneDNN functionality with SYCL Graph record/execute mode.
* Extended benchdnn option [`--cold-cache`](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/knob_cold_cache.md) with support for cold TLB mode.
* Added benchdnn option `--bia-dt` to control bias data type for matmul, inner product, convolution, and deconvolution.
* Extended syntax of benchdnn `--dt` option in [Graph API driver](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/driver_graph.md) to manage data types of individual tensors in a pattern.

# Deprecated Functionality

# Breaking Changes
* Removed experimental [Graph Compiler](https://oneapi-src.github.io/oneDNN/v3.6/dev_guide_graph_compiler.html) backend.

# Thanks to these Contributors
This release contains contributions from the project core team as well as Alexander Simonov @asimonov1, Denis @redradist, Dmitriy Ovchinnikov @inteldimitrius, Eliezer Weissmann @eliezerweissmann, Hubert Maciak @hmaciak, Ilya Lavrenov @ilya-lavrenov, James McGregor @Jmc18134, Marek Michalowski @michalowski-arm, Maria Zhukova @mzhukova, Orel Yehuda @yehudaorel, Ravi Pushkar @rpushkarr, Renato Barros Arantes @renato-arantes, Shreyas-fuj @Shreyas-fuj, Shu Chen @shu1chen, Viktoriia Gvozdeva @vgvozdeva, Yair Obodovsky @yair-obodovsky, hmaciak @hmaciak, jstachowintel @jstachowintel, zhangfei @zhangfeiv0.
