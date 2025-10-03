Examples and Tutorials {#dev_guide_examples}
============================================

Functional API
--------------

| Topic           | Engine  | C++ API                                | C API                       |
|:----------------|:--------|:---------------------------------------|:----------------------------|
| Tutorials       | CPU/GPU | @ref getting_started_cpp               |                             |
|                 | CPU/GPU | @ref memory_format_propagation_cpp     |                             |
|                 | CPU/GPU | @ref cross_engine_reorder_cpp          | @ref cross_engine_reorder_c |
|                 | CPU/GPU | @ref sycl_interop_buffer_cpp           |                             |
|                 | GPU     | @ref gpu_opencl_interop_cpp            |                             |
|                 | CPU/GPU | @ref bnorm_u8_via_binary_postops_cpp   |                             |
|                 | CPU/GPU | @ref matmul_with_host_scalar_scale_cpp |                             |
| Performance     | CPU/GPU | @ref performance_profiling_cpp         |                             |
|                 | CPU/GPU | @ref matmul_perf_cpp                   |                             |
|                 | CPU/GPU | @ref bnorm_u8_via_binary_postops_cpp   |                             |
| f32 inference   | CPU/GPU | @ref cnn_inference_f32_cpp             | @ref cnn_inference_f32_c    |
|                 | CPU     | @ref cpu_rnn_inference_f32_cpp         |                             |
| int8 inference  | CPU/GPU | @ref cnn_inference_int8_cpp            |                             |
|                 | CPU     | @ref cpu_rnn_inference_int8_cpp        |                             |
| f32 training    | CPU/GPU | @ref cnn_training_f32_cpp              |                             |
|                 | CPU     |                                        | @ref cpu_cnn_training_f32_c |
|                 | CPU/GPU | @ref rnn_training_f32_cpp              |                             |
| bf16 training   | CPU/GPU | @ref cnn_training_bf16_cpp             |                             |
| f8 quantization | CPU/GPU | @ref matmul_f8_quantization_cpp        |                             |

Graph API
---------

| Topic          | Engine  | Example Name                              |
|:---------------|:--------|:------------------------------------------|
| Tutorials      | CPU     | @ref graph_cpu_getting_started_cpp        |
|                | CPU     | @ref graph_cpu_inference_int8_cpp         |
|                | CPU/GPU | @ref graph_sycl_getting_started_cpp       |
|                | CPU     | @ref graph_cpu_single_op_partition_cpp    |
|                | GPU     | @ref graph_sycl_single_op_partition_cpp   |
|                | GPU     | @ref graph_gpu_opencl_getting_started_cpp |
