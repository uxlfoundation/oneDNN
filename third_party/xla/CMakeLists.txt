# XLA doesn't support compilation with GCC natively. If any warnings pop up,
# they should be silenced through the following adjustment of `CMAKE_CXX_FLAGS`.
# if(UNIX)
#     append(CMAKE_CXX_FLAGS "-Wno-sign-compare")
# endif()

set(TARGET_NAME dnnl_xla)
file(GLOB_RECURSE XLA_SOURCES ${PROJECT_SOURCE_DIR}/third_party/xla/*.cc)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/third_party/xla)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/third_party/xla/third_party)

add_library(${TARGET_NAME} STATIC ${XLA_SOURCES})
