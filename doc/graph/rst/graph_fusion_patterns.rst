Fusion Patterns
###############

.. default-role:: math
.. toctree::
   :maxdepth: 1
   :hidden:

   dev_guide_graph_gated_mlp
   dev_guide_graph_gqa
   dev_guide_graph_sdpa_compressed_kv
   dev_guide_graph_sdpa
   dev_guide_graph_matmul_fusions
   dev_guide_graph_quantized_matmul_fusions
   dev_guide_graph_convolution_fusions
   dev_guide_graph_quantized_convolution_fusions
   dev_guide_graph_convtranspose_fusions
   dev_guide_graph_quantized_convtranspose_fusions
   dev_guide_graph_binary_fusions
   dev_guide_graph_unary_fusions
   dev_guide_graph_interpolate_fusions
   dev_guide_graph_reduction_fusions
   dev_guide_graph_pool_fusions
   dev_guide_graph_norm_fusions
   dev_guide_graph_softmax_fusions
   dev_guide_graph_other_fusions


The following fusion patterns represent subgraphs that the oneDNN Graph API
identifies as candidates for partitions. Users can define computation graphs
according to these patterns, get partitions from the graph, compile the
partitions into compiled partitions, and execute them to obtain results. See
`Graph API Basic Concepts <dev_guide_graph_basic_concepts.html>`_ for more
details about the programming model.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1

   * - Pattern
     - Description
   * - Scaled Dot-Product Attention
     - This pattern is widely used for attention mechanisms in transformer models, e.g., BERT and GPT. Refer to `Scaled Dot-Product Attention (SDPA) <dev_guide_graph_sdpa.html>`_ for more details.
   * - Grouped Query Attention
     - This pattern is widely in LLM models like llama2 70b and llama3 to reduce the memory usage of the kv cache during inference. Refer to `Grouped Query Attention (GQA) <dev_guide_graph_gqa.html>`_ for more details.
   * - Scaled Dot-Product Attention with Compressed Key/Value
     - This pattern is used for memory-efficient attention mechanisms. Refer to `Scaled Dot-Product Attention with Compressed Key/Value <dev_guide_graph_sdpa_compressed_kv.html>`_ for more details.
   * - Gated Multi-Layer Perceptron (Gated-MLP)
     - This pattern is widely used for enhancing feedforward layers in transformer models, e.g., Vision Transformers (ViT). Refer to `Gated Multi-Layer Perceptron (Gated-MLP) <dev_guide_graph_gated_mlp.html>`_ for more details.
   * - MatMul Fusion Patterns
     - This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc. Refer to `MatMul Fusion Patterns <dev_guide_graph_matmul_fusion_patterns.html>`_ for more details.
   * - Quantized MatMul Fusion Patterns
     - This pattern is widely used for efficient matrix multiplication in quantized models. Refer to `Quantized MatMul Fusion Patterns <dev_guide_graph_quantized_matmul_fusion_patterns.html>`_ for more details.
   * - Convolution Fusion Patterns
     - This pattern is widely used in Convolution Neural Networks, e.g., ResNet, ResNext, SSD, etc. Refer to `Convolution Fusion Patterns <dev_guide_graph_convolution_fusion_patterns.html>`_ for more details.
   * - Quantized Convolution Fusion Patterns
     - This pattern is widely used in quantized Convolution Neural Networks. Refer to `Quantized Convolution Fusion Patterns <dev_guide_graph_quantized_convolution_fusion_patterns.html>`_ for more details.
   * - ConvTranspose Fusion Patterns
     - This pattern is widely used for upsampling in Generative Adversarial Networks. Refer to `ConvTranspose Fusion Patterns <dev_guide_graph_convtranspose_fusion_patterns.html>`_ for more details.
   * - Quantized ConvTranspose Fusion Patterns
     - This pattern is widely used in quantized Generative Adversarial Networks. Refer to `Quantized ConvTranspose Fusion Patterns <dev_guide_graph_quantized_convtranspose_fusion_patterns.html>`_ for more details.
   * - Binary Fusion Patterns
     - Fusion Patterns related to binary operations like Add, Divide, Maximum, Minimum, Multiply, Subtract. This pattern is widely used in language models and recommendation models, e.g., BERT, DLRM. Refer to `Binary Fusion Patterns <dev_guide_graph_binary_fusion_patterns.html>`_ for more details.
   * - Unary Fusion Patterns
     - Fusion Patterns related to unary operations like Abs, Clamp, Elu, Exp, GELU, HardSigmoid, HardSwish, LeakyReLU, Log, Mish, Sigmoid, SoftPlus, ReLU, Round, Sqrt, Square, Tanh. This pattern is widely used in Convolution Neural Networks. Refer to `Unary Fusion Patterns <dev_guide_graph_unary_fusion_patterns.html>`_ for more details.
   * - Interpolate Fusion Patterns
     - This pattern is widely used for image processing. Refer to `Interpolate Fusion Patterns <dev_guide_graph_interpolate_fusion_patterns.html>`_ for more details.
   * - Reduction Fusion Patterns
     - Fusion Patterns related to reduction operations like ReduceL1, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum. This pattern is widely used for data processing, for example loss reduction. Refer to `Reduction Fusion Patterns <dev_guide_graph_reduction_fusion_patterns.html>`_ for more details.
   * - Pool Fusion Patterns
     - Fusion Patterns related to pool operations like MaxPool, AvgPool. This pattern is widely used in Convolution Neural Networks. Refer to `Pool Fusion Patterns <dev_guide_graph_pool_fusion_patterns.html>`_ for more details.
   * - Norm Fusion Patterns
     - Fusion Patterns related to norm operations like GroupNorm, LayerNorm, BatchNormInference. This pattern is widely used in Convolution Neural Networks, for example DenseNet. Refer to `Norm Fusion Patterns <dev_guide_graph_norm_fusion_patterns.html>`_ for more details.
   * - SoftMax Fusion Patterns
     - This pattern is widely used in Convolution Neural Networks. Refer to `SoftMax Fusion Patterns <dev_guide_graph_softmax_fusion_patterns.html>`_ for more details.
   * - Other Fusion Patterns
     - Fusion Patterns related to Reorder, StaticReshape, StaticTranspose, Concat, Interpolate, Reciprocal, TypeCast. Refer to `Other Fusion Patterns <dev_guide_graph_other_fusion_patterns.html>`_ for more details.
