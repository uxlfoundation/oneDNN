#ifndef NGEN_DYNAMIC_HPP
#define NGEN_DYNAMIC_HPP

#include "ngen_config_internal.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "windows.h"
#else
#include <dlfcn.h>
#endif

namespace NGEN_NAMESPACE {
namespace dynamic {

inline void *findSymbol(const char *lib, const char *symbol)
{
    // In nGEN usage, the caller has always initialized the runtime library (OCL, L0)
    //   prior to invoking nGEN, and is responsible for the lifetime of this library.
    // Hence we can always rely here on the library being loaded and initialized.

#ifdef _WIN32
    HMODULE handle = GetModuleHandleA(lib);
    if (!handle) return nullptr;
    return reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#else
    void *handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!handle) return nullptr;
    return dlsym(handle, symbol);
#endif
}

} /* namespace dynamic */
} /* namespace NGEN_NAMESPACE */

#endif
