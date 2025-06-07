oneAPI Deep Neural Network Library (oneDNN) Developer Guide and Reference
=========================================================================

oneAPI Deep Neural Network Library (oneDNN) is an open-source cross-platform 
performance library of basic building blocks for deep learning applications.

The library is optimized for Intel(R) Architecture Processors, Intel Graphics, 
and Arm(R) 64-bit Architecture (AArch64)-based processors. oneDNN has experimental 
support for the following architectures: NVIDIA* GPU, AMD* GPU, 
OpenPOWER* Power ISA (PPC64), IBMz* (s390x), and RISC-V.

oneDNN is intended for deep learning applications and framework developers 
interested in improving application performance on CPUs and GPUs.

.. toctree::
   :caption: About
   :hidden:
   :maxdepth: 1

   Introduction<self>

.. toctree::
   :caption: Get Started 
   :hidden:
   :maxdepth: 1

   Understand oneDNN API<dev_guide_basic_concepts>
   build_and_link
   page_getting_started_cpp

.. toctree::
   :caption: Developer Guide
   :hidden:
   :maxdepth: 1

   programming_model
   supported_primitives
   graph_extension
   dev_guide_examples
   performance_profiling_and_inspection
   advanced_topics
   ukernels

.. toctree::
   :caption: Developer Reference  
   :hidden:
   :maxdepth: 1

   group_dnnl_api.rst