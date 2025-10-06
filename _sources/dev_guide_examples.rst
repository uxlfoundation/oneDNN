.. index:: pair: page; Examples and Tutorials
.. _doxid-dev_guide_examples:

Examples and Tutorials
======================

Functional API
~~~~~~~~~~~~~~

================  ========  ====================================================================================================  ==========================================================================  
Topic             Engine    C++ API                                                                                               C API                                                                       
================  ========  ====================================================================================================  ==========================================================================  
Tutorials         CPU/GPU   :ref:`oneDNN API Basic Workflow Tutorial <doxid-getting_started_cpp>`                                                                                                             
                  CPU/GPU   :ref:`Memory Format Propagation <doxid-memory_format_propagation_cpp>`                                                                                                            
                  CPU/GPU   :ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_cpp>`                           :ref:`Reorder between CPU and GPU engines <doxid-cross_engine_reorder_c>`   
                  CPU/GPU   :ref:`Getting started on both CPU and GPU with SYCL extensions API <doxid-sycl_interop_buffer_cpp>`                                                                               
                  GPU       :ref:`Getting started on GPU with OpenCL extensions API <doxid-gpu_opencl_interop_cpp>`                                                                                           
                  CPU/GPU   :ref:`Bnorm u8 by binary post-ops example <doxid-bnorm_u8_via_binary_postops_cpp>`                                                                                                
                  CPU/GPU   :ref:`MatMul with Host Scalar Scale example <doxid-matmul_with_host_scalar_scale_cpp>`                                                                                            
Performance       CPU/GPU   :ref:`Performance Profiling Example <doxid-performance_profiling_cpp>`                                                                                                            
                  CPU/GPU   :ref:`Matrix Multiplication Performance Example <doxid-matmul_perf_cpp>`                                                                                                          
                  CPU/GPU   :ref:`Bnorm u8 by binary post-ops example <doxid-bnorm_u8_via_binary_postops_cpp>`                                                                                                
f32 inference     CPU/GPU   :ref:`CNN f32 inference example <doxid-cnn_inference_f32_cpp>`                                        :ref:`CNN f32 inference example <doxid-cnn_inference_f32_c>`                
                  CPU       :ref:`RNN f32 inference example <doxid-cpu_rnn_inference_f32_cpp>`                                                                                                                
int8 inference    CPU/GPU   :ref:`CNN int8 inference example <doxid-cnn_inference_int8_cpp>`                                                                                                                  
                  CPU       :ref:`RNN int8 inference example <doxid-cpu_rnn_inference_int8_cpp>`                                                                                                              
f32 training      CPU/GPU   :ref:`CNN f32 training example <doxid-cnn_training_f32_cpp>`                                                                                                                      
                  CPU                                                                                                             :ref:`CNN f32 training example <doxid-cpu_cnn_training_f32_c>`              
                  CPU/GPU   :ref:`RNN f32 training example <doxid-rnn_training_f32_cpp>`                                                                                                                      
bf16 training     CPU/GPU   :ref:`CNN bf16 training example <doxid-cnn_training_bf16_cpp>`                                                                                                                    
f8 quantization   CPU/GPU   :ref:`Matrix Multiplication with f8 Quantization <doxid-matmul_f8_quantization_cpp>`                                                                                              
================  ========  ====================================================================================================  ==========================================================================

Graph API
~~~~~~~~~

==========  ========  =========================================================================================================  
Topic       Engine    Example Name                                                                                               
==========  ========  =========================================================================================================  
Tutorials   CPU       :ref:`Getting started on CPU with Graph API <doxid-graph_cpu_getting_started_cpp>`                         
            CPU       :ref:`Convolution int8 inference example with Graph API <doxid-graph_cpu_inference_int8_cpp>`              
            CPU/GPU   :ref:`Getting started with SYCL extensions API and Graph API <doxid-graph_sycl_getting_started_cpp>`       
            CPU       :ref:`Single op partition on CPU <doxid-graph_cpu_single_op_partition_cpp>`                                
            GPU       :ref:`Single op partition on GPU <doxid-graph_sycl_single_op_partition_cpp>`                               
            GPU       :ref:`Getting started with OpenCL extensions and Graph API <doxid-graph_gpu_opencl_getting_started_cpp>`   
==========  ========  =========================================================================================================

