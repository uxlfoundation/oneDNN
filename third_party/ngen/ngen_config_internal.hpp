/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef NGEN_CONFIG_INTERNAL_HPP
#define NGEN_CONFIG_INTERNAL_HPP

// Drop NGEN_CONFIG define once C++11/14 support dropped
#if (defined(__has_include) && __has_include("ngen_config.hpp")) || defined(NGEN_CONFIG)
#include "ngen_config.hpp"

#ifndef NGEN_ASM_SHOW_FORMATS
#define NGEN_ASM_SHOW_FORMATS 0
#endif

#else
// Default config settings

#ifndef NGEN_NAMESPACE
#define NGEN_NAMESPACE ngen
#endif

#ifndef NGEN_ASM
#define NGEN_ASM
#endif

#ifndef NGEN_DUMP
#define NGEN_DUMP
#endif

#if (__cplusplus >= 202002L || _MSVC_LANG >= 202002L)
#if __has_include(<version>)
#include <version>
#if __cpp_lib_source_location >= 201907L
#define NGEN_ENABLE_SOURCE_LOCATION true
#endif
#endif
#endif

#ifdef NGEN_ENABLE_SOURCE_LOCATION
#define NGEN_DEFAULT_DEBUG_LINE_MAPPING true
#endif

#endif

#ifndef NGEN_IGNORE_DEPRECATE
#ifdef _MSC_VER
#define NGEN_DEPRECATION_WARNING(msg) \
    __pragma(message(msg)) \
    __pragma(message("Define NGEN_IGNORE_DEPRECATE to suppress deprecation warnings."))
#else
#define NGEN_DETAIL_PRAGMA_IMPL(x) _Pragma(#x)
#define NGEN_DETAIL_PRAGMA(x) NGEN_DETAIL_PRAGMA_IMPL(x)
#define NGEN_DEPRECATION_WARNING(msg) \
    NGEN_DETAIL_PRAGMA(GCC warning msg) \
    NGEN_DETAIL_PRAGMA(message "Define NGEN_IGNORE_DEPRECATE to suppress deprecation warnings.")
#endif
#else
#define NGEN_DEPRECATION_WARNING(msg)
#endif

#endif /* header guard */
