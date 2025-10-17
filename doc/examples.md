# Examples and Tutorials {#dev_guide_examples}

This page provides an overview of oneDNN examples organized by functionality and use case.

## Functional API Examples

The Functional API provides access to individual oneDNN primitives.

### Fundamental Concepts and API Basics

| Example | Description |
|:--------|:------------|
| @ref getting_started_cpp | \copybrief getting_started_cpp_brief |
| @ref memory_format_propagation_cpp | \copybrief memory_format_propagation_cpp_brief |
| @ref cross_engine_reorder_cpp | \copybrief cross_engine_reorder_cpp_brief |

### Interoperability with External Runtimes

| Example | Description |
|:--------|:------------|
| @ref sycl_interop_buffer_cpp | \copybrief sycl_interop_buffer_cpp_brief |
| @ref sycl_interop_usm_cpp | \copybrief sycl_interop_usm_cpp_brief |
| @ref gpu_opencl_interop_cpp | \copybrief gpu_opencl_interop_cpp_brief |

### Matrix Multiplication with Different oneDNN Features

Basic Operations:

| Example | Description |
|:--------|:------------|
| @ref matmul_example_cpp | \copybrief matmul_example_cpp_brief |
| @ref cpu_sgemm_and_matmul_cpp | \copybrief cpu_sgemm_and_matmul_cpp_brief |

Quantization flavors:

| Example | Description |
|:--------|:------------|
| @ref matmul_f8_quantization_cpp | \copybrief matmul_f8_quantization_cpp_brief |
| @ref cpu_matmul_quantization_cpp | \copybrief cpu_matmul_quantization_cpp_brief |
| @ref inference_int8_matmul_cpp | \copybrief inference_int8_matmul_cpp_brief |
| @ref mxfp_matmul_cpp | \copybrief mxfp_matmul_cpp_brief |

Advanced Usages:

| Example | Description |
|:--------|:------------|
| @ref matmul_with_host_scalar_scale_cpp | \copybrief matmul_with_host_scalar_scale_cpp_brief |
| @ref cpu_matmul_coo_cpp | \copybrief cpu_matmul_coo_cpp_brief |
| @ref cpu_matmul_csr_cpp | \copybrief cpu_matmul_csr_cpp_brief |
| @ref cpu_matmul_weights_compression_cpp | \copybrief cpu_matmul_weights_compression_cpp_brief |
| @ref weights_decompression_matmul_cpp | \copybrief weights_decompression_matmul_cpp_brief |

### Inference and Training

Neural network implementations demonstrating inference and training workflows:

| Type | Precision | Mode | Example | Description |
|:-----|:----------|:-----|:--------|:------------|
| CNN | f32 | Inference | @ref cnn_inference_f32_cpp | \copybrief cnn_inference_f32_cpp_brief |
| CNN | int8 | Inference | @ref cnn_inference_int8_cpp | \copybrief cnn_inference_int8_cpp_brief |
| CNN | f32 | Training | @ref cnn_training_f32_cpp | \copybrief cnn_training_f32_cpp_brief |
| CNN | bf16 | Training | @ref cnn_training_bf16_cpp | \copybrief cnn_training_bf16_cpp_brief |
| RNN | f32 | Inference | @ref cpu_rnn_inference_f32_cpp | \copybrief cpu_rnn_inference_f32_cpp_brief |
| RNN | int8 | Inference | @ref cpu_rnn_inference_int8_cpp | \copybrief cpu_rnn_inference_int8_cpp_brief |
| RNN | f32 | Training | @ref rnn_training_f32_cpp | \copybrief rnn_training_f32_cpp_brief |

### Recurrent Neural Networks

| Example | Description |
|:--------|:------------|
| @ref vanilla_rnn_example_cpp | \copybrief vanilla_rnn_example_cpp_brief
| @ref lstm_example_cpp | \copybrief lstm_example_cpp_brief |
| @ref lbr_gru_example_cpp | \copybrief lbr_gru_example_cpp_brief |
| @ref augru_example_cpp | \copybrief augru_example_cpp_brief |

### Performance Analysis

A few techniques for performance measurements:

| Example | Description |
|:--------|:------------|
| @ref matmul_perf_cpp | \copybrief matmul_perf_cpp_brief |
| @ref performance_profiling_cpp | \copybrief performance_profiling_cpp_brief |

### Individual Primitives

Convolution Operations:

| Example | Description |
|:--------|:------------|
| @ref convolution_example_cpp | \copybrief convolution_example_cpp_brief |
| @ref deconvolution_example_cpp | \copybrief deconvolution_example_cpp_brief |

Linear Operations:

| Example | Description |
|:--------|:------------|
| @ref inner_product_example_cpp | \copybrief inner_product_example_cpp_brief |

Pooling and Sampling:

| Example | Description |
|:--------|:------------|
| @ref pooling_example_cpp | \copybrief pooling_example_cpp_brief |
| @ref resampling_example_cpp | \copybrief resampling_example_cpp_brief |

Normalization Primitives:

| Example | Description |
|:--------|:------------|
| @ref batch_normalization_example_cpp | \copybrief batch_normalization_example_cpp_brief
| @ref group_normalization_example_cpp | \copybrief group_normalization_example_cpp_brief |
| @ref layer_normalization_example_cpp | \copybrief layer_normalization_example_cpp_brief |
| @ref lrn_example_cpp | \copybrief lrn_example_cpp_brief |

Activation Functions:

| Example | Description |
|:--------|:------------|
| @ref eltwise_example_cpp | \copybrief eltwise_example_cpp_brief |
| @ref prelu_example_cpp | \copybrief prelu_example_cpp_brief |
| @ref softmax_example_cpp | \copybrief softmax_example_cpp_brief |

Tensor Operations:

| Example | Description |
|:--------|:------------|
| @ref binary_example_cpp | \copybrief binary_example_cpp_brief |
| @ref bnorm_u8_via_binary_postops_cpp | \copybrief bnorm_u8_via_binary_postops_cpp_brief |
| @ref concat_example_cpp | \copybrief concat_example_cpp_brief |
| @ref reduction_example_cpp | \copybrief reduction_example_cpp_brief |
| @ref sum_example_cpp | \copybrief sum_example_cpp_brief |
| @ref shuffle_example_cpp | \copybrief shuffle_example_cpp_brief |

Memory Transformations:

| Example | Description |
|:--------|:------------|
| @ref reorder_example_cpp | \copybrief reorder_example_cpp_brief |

### C API Examples

| Example | Description |
|:--------|:------------|
| @ref cross_engine_reorder_c | \copybrief cross_engine_reorder_c_brief |
| @ref cnn_inference_f32_c | \copybrief cnn_inference_f32_c_brief |
| @ref cpu_cnn_training_f32_c | \copybrief cpu_cnn_training_f32_c_brief |

## Graph API Examples

The Graph API provides an interface for defining computational graphs with optimization and fusion capabilities.

### Getting Started with Graph API

| Example | Description |
|:--------|:------------|
| @ref graph_cpu_getting_started_cpp | \copybrief graph_cpu_getting_started_cpp_brief |
| @ref graph_sycl_getting_started_cpp | \copybrief graph_sycl_getting_started_cpp_brief |
| @ref graph_gpu_opencl_getting_started_cpp | \copybrief graph_gpu_opencl_getting_started_cpp_brief |

### Advanced Graph API Usage

| Example | Description |
|:--------|:------------|
| @ref graph_cpu_inference_int8_cpp | \copybrief graph_cpu_inference_int8_cpp_brief |
| @ref graph_cpu_single_op_partition_cpp | \copybrief graph_cpu_single_op_partition_cpp_brief |
| @ref graph_sycl_single_op_partition_cpp | \copybrief graph_sycl_single_op_partition_cpp_brief |

## Microkernel (uKernel) API Examples

The oneDNN microkernel API is a low-level abstraction for CPU that provides maximum flexibility
by allowing users to maintain full control over threading logic, blocking logic, and code customization
with minimal overhead.

| Example | Description |
|:--------|:------------|
| @ref cpu_brgemm_example_cpp | \copybrief cpu_brgemm_example_cpp_brief |

## Running Examples

### Prerequisites and Building Examples

Before running examples, ensure:
1. oneDNN is built from source.
   Note that examples are built automatically when building oneDNN
   with `-DONEDNN_BUILD_EXAMPLES=ON` (enabled by default).
2. Environment is set up and oneDNN libraries are in the path.

Refer to @ref dev_guide_build for detailed build instructions.

### Running Examples

Most examples accept an optional engine argument (`cpu` or `gpu`),
and if no argument is provided, example will most likely default to CPU:

**Linux/macOS:**
```bash
# Run on CPU (default)
./examples/getting_started

# Run on CPU explicitly
./examples/getting_started cpu

# Run on GPU (if available)
./examples/getting_started gpu
```

**Windows:**
```cmd
# Run on CPU (default)
examples\getting_started.exe

# Run on CPU explicitly
examples\getting_started.exe cpu

# Run on GPU (if available)
examples\getting_started.exe gpu
```

Examples will output "Example passed on CPU/GPU." upon successful completion
and display an error status with message otherwise.
