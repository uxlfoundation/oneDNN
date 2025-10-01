[abseil-cpp](https://github.com/abseil/abseil-cpp) is a hard code requirement
for [OpenXLA](https://github.com/openxla/xla) which delivers an asynchronous
Eigen threadpool runtime to oneDNN.

Gets built only with tests folder when DNNL_CPU_RUNTIME=THREADPOOL is used and
internal build time variable `_DNNL_TEST_THREADPOOL_IMPL` is set to
`EIGEN_ASYNC`.

The state of source copy is at master@a54cb45c from the mentioned repository.
