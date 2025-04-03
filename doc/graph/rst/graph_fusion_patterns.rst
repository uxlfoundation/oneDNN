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


The following fusion patterns are subgraphs that the oneDNN Graph API
recognizes as candidates for fusion.

.. list-table:: 
   :widths: 75 25
   :header-rows: 1

   * - Pattern
     - Description
   * - Scaled Dot-Product Attention
     - Refer to `Scaled Dot-Product Attention (SDPA) <dev_guide_graph_sdpa.html>`_ for more details.
   * - Grouped Query Attention
     - Refer to `Grouped Query Attention (GQA) <dev_guide_graph_gqa.html>`_ for more details.
   * - Scaled Dot-Product Attention with Compressed Key/Value
     - Refer to `Scaled Dot-Product Attention with Compressed Key/Value <dev_guide_graph_sdpa_compressed_kv.html>`_ for more details.
   * - Gated Multi-Layer Perceptron (Gated-MLP)
     - Refer to `Gated Multi-Layer Perceptron (Gated-MLP) <dev_guide_graph_gated_mlp.html>`_ for more details.
   * - MatMul Fusion Patterns
     - This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc. Refer to `MatMul Fusions <dev_guide_graph_matmul_fusions.html>`_ for more details.
   * - Quantized MatMul Fusion Patterns
     - This pattern is a quantized version of MatMul Fusion Patterns. Refer to `Quantized MatMul Fusions <dev_guide_graph_quantized_matmul_fusions.html>`_ for more details.
   * - Convolution Fusion Patterns
     - This pattern is widely used in Convolution Neural Networks, for example ResNet, ResNext, SSD, etc. Refer to `Convolution Fusions <dev_guide_graph_convolution_fusions.html>`_ for more details.
   * - Quantized Convolution Fusion Patterns
     - This pattern is a quantized version of Convolution Fusion Patterns. Refer to `Quantized Convolution Fusions <dev_guide_graph_quantized_convolution_fusions.html>`_ for more details.
   * - ConvTranspose Fusion Patterns
     - This pattern is widely used in Generative Adversarial Networks. Refer to `ConvTranspose Fusions <dev_guide_graph_convtranspose_fusions.html>`_ for more details.
   * - Quantized ConvTranspose Fusion Patterns
     - This pattern is a quantized version of ConvTranspose Fusion Patterns. Refer to `Quantized ConvTranspose Fusions <dev_guide_graph_quantized_convtranspose_fusions.html>`_ for more details.
   * - Binary Fusion Patterns
     - Fusions related to binary operations like Add, Divide, Maximum, Minimum, Multiply, Subtract. This pattern is widely used in language models and recommendation models, for example BERT, DLRM, etc. Refer to `Binary Fusions <dev_guide_graph_binary_fusions.html>`_ for more details.
   * - Unary Fusion Patterns
     - Fusions related to unary operations like Abs, Clamp, Elu, Exp, GELU, HardSigmoid, HardSwish, LeakyReLU, Log, Mish, Sigmoid, SoftPlus, ReLU, Round, Sqrt, Square, Tanh. This pattern is widely used in Convolution Neural Networks. Refer to `Unary Fusions <dev_guide_graph_unary_fusions.html>`_ for more details.
   * - Interpolate Fusion Patterns
     - This pattern is widely used for image processing. Refer to `Interpolate Fusions <dev_guide_graph_interpolate_fusions.html>`_ for more details.
   * - Reduction Fusion Patterns
     - Fusions related to reduction operations like ReduceL1, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum. This pattern is widely used for data processing, for example loss reduction. Refer to `Reduction Fusions <dev_guide_graph_reduction_fusions.html>`_ for more details.
   * - Pool Fusion Patterns
     - Fusions related to pool operations like MaxPool, AvgPool. This pattern is widely used in Convolution Neural Networks. Refer to `Pool Fusions <dev_guide_graph_pool_fusions.html>`_ for more details.
   * - Norm Fusion Patterns
     - Fusions related to norm operations like GroupNorm, LayerNorm, BatchNormInference. This pattern is widely used in Convolution Neural Networks, for example DenseNet. Refer to `Norm Fusions <dev_guide_graph_norm_fusions.html>`_ for more details.
   * - SoftMax Fusion Patterns
     - This pattern is widely used in Convolution Neural Networks. Refer to `SoftMax Fusions <dev_guide_graph_softmax_fusions.html>`_ for more details.
   * - Other Fusion Patterns
     - Fusions related to Reorder, StaticReshape, StaticTranspose, Concat, Interpolate, Reciprocal, TypeCast. Refer to `Other Fusions <dev_guide_graph_other_fusions.html>`_ for more details.
