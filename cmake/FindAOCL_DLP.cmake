# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

# ----------
# FindAOCL_DLP
# ----------
#
# Finds the AOCL-DLP (Deep Learning Primitives) library
# https://github.com/amd/aocl-dlp
#
# This module defines the following variables:
#
#   AOCL_DLP_FOUND          - True if AOCL-DLP was found
#   AOCL_DLP_INCLUDE_DIRS   - include directories for AOCL-DLP
#   AOCL_DLP_LIBRARIES      - link against this library to use AOCL-DLP
#
# The module will also define two cache variables:
#
#   AOCL_DLP_INCLUDE_DIR    - the AOCL-DLP include directory
#   AOCL_DLP_LIBRARY        - the path to the AOCL-DLP library
#

# Use AOCL_DLP_ROOT_DIR environment variable to find the library and headers
find_path(AOCL_DLP_INCLUDE_DIR
  NAMES aocl_dlp.h
  PATHS ENV AOCL_DLP_ROOT_DIR
  PATH_SUFFIXES include
  )

find_library(AOCL_DLP_LIBRARY
  NAMES aocl-dlp aocl_dlp
  PATHS ENV AOCL_DLP_ROOT_DIR
  PATH_SUFFIXES lib lib64 build
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AOCL_DLP DEFAULT_MSG
  AOCL_DLP_INCLUDE_DIR
  AOCL_DLP_LIBRARY
)

mark_as_advanced(
  AOCL_DLP_LIBRARY
  AOCL_DLP_INCLUDE_DIR
  )

if(AOCL_DLP_FOUND)
  set(AOCL_DLP_INCLUDE_DIRS ${AOCL_DLP_INCLUDE_DIR})
  set(AOCL_DLP_LIBRARIES ${AOCL_DLP_LIBRARY})
endif()
