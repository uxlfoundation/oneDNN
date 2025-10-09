This is a copy-exact of all sources and CMake files from
[abseil-cpp](git@github.com:abseil/abseil-cpp.git). The software is used to
support oneDNN validation of Eigen Threadpool runtime.

The snippet is taken from `lts_2025_08_14` branch, commit `d38452e1` which
denotes `tag: 20250814.1`.

The management of the dependency on abseil-cpp is done through
tests/CMakeLists.txt.

Note: in case something is not compiled, feel free to update
`absl/copts/GENERATED_AbseilCopts.cmake` file for correspondent compiler.
