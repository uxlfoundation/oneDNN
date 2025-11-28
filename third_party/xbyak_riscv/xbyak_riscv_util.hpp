/******************************************************************************
* Copyright (C), 2023, KNS Group LLC (YADRO)
*
* Licensed under the 3-Clause BSD License
* You may obtain a copy of the License at https://opensource.org/license/bsd-3-clause/
*******************************************************************************/

#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include "xbyak_riscv_csr.hpp"
#include "xbyak_riscv.hpp"

#if defined(__linux__) && defined(__riscv)
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/utsname.h>
#include <asm/hwcap.h>
#include <unistd.h>
#endif

namespace Xbyak_riscv {

#ifndef COMPAT_HWCAP_ISA_I
#define COMPAT_HWCAP_ISA_I  (1U << ('I' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_M
#define COMPAT_HWCAP_ISA_M  (1U << ('M' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_A
#define COMPAT_HWCAP_ISA_A  (1U << ('A' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_F
#define COMPAT_HWCAP_ISA_F  (1U << ('F' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_D
#define COMPAT_HWCAP_ISA_D  (1U << ('D' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_C
#define COMPAT_HWCAP_ISA_C  (1U << ('C' - 'A'))
#endif

#ifndef COMPAT_HWCAP_ISA_V
#define COMPAT_HWCAP_ISA_V  (1U << ('V' - 'A'))
#endif

enum class RISCVExtension : uint64_t {
    I = COMPAT_HWCAP_ISA_I,
    M = COMPAT_HWCAP_ISA_M,
    A = COMPAT_HWCAP_ISA_A,
    F = COMPAT_HWCAP_ISA_F,
    D = COMPAT_HWCAP_ISA_D,
    C = COMPAT_HWCAP_ISA_C,
    V = COMPAT_HWCAP_ISA_V
};

template <CSR csr>
struct CSRReader : public CodeGenerator {
    // Buffer capacity exactly for 2 instructions.
    static constexpr size_t capacity = 8;

    CSRReader() : CodeGenerator(capacity) {
        csrrs(a0, csr, x0);
        ret();
    }
};

/**
 * Class that detects information about a RISC-V CPU.
 */
class CPU final {
public:
    static const CPU& getInstance() {
        static const CPU cpu;
        return cpu;
    }

    CPU() {
#if defined(__linux__) && defined(__riscv)
        // Set hwcapFeatures with AT_HWCAP value from
        // the Linux auxiliary vector to check for base extensions support.
        hwcapFeatures = getauxval(AT_HWCAP) & (
            COMPAT_HWCAP_ISA_I |
            COMPAT_HWCAP_ISA_M |
            COMPAT_HWCAP_ISA_A |
            COMPAT_HWCAP_ISA_F |
            COMPAT_HWCAP_ISA_D |
            COMPAT_HWCAP_ISA_C |
            COMPAT_HWCAP_ISA_V
        );

        // Set xlen, number of cores, cache info
        xlen = sysconf(_SC_LONG_BIT);
        numCores = sysconf(_SC_NPROCESSORS_ONLN);
        numCores = sysconf(_SC_NPROCESSORS_ONLN);

        dataCacheSize_[0] = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        dataCacheSize_[1] = sysconf(_SC_LEVEL2_CACHE_SIZE);
        dataCacheSize_[2] = sysconf(_SC_LEVEL3_CACHE_SIZE);
        dataCacheSize_[3] = sysconf(_SC_LEVEL4_CACHE_SIZE);

        dataCacheLineSize_[0] = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        dataCacheLineSize_[1] = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
        dataCacheLineSize_[2] = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
        dataCacheLineSize_[3] = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
#endif

        // Set vlen
        if(hasExtension(RISCVExtension::V)) {
            CSRReader<CSR::vlenb> csrReaderGenerator;
            csrReaderGenerator.ready();
            const auto csrReader = csrReaderGenerator.getCode<uint32_t (*)()>();
            vlen = csrReader() * 8 /* bit */;
        }

        // Set flen (bit)
        if (hasExtension(RISCVExtension::D)) {
            flen = 64;
        } else if (hasExtension(RISCVExtension::F)) {
            flen = 32;
        }
    }

    /**
     * Checks if a particular RISC-V extension is available.
     *
     * @param extension The extension to check.
     */
    bool hasExtension(RISCVExtension extension) const {
        return (hwcapFeatures & static_cast<uint64_t>(extension)) != 0;
    }

    /**
     * Get vector register width in bits
    */
    uint32_t getVlen() const {
        return vlen;
    }

    /**
     * Get general purpose register width in bits
    */
    uint32_t getXlen() const {
        return xlen;
    };

    /**
     * Get floating-point register width in bits
    */
    uint32_t getFlen() const {
        return flen;
    }

    uint32_t getNumCores() const {
        return numCores;
    }

    /**
     * Get data cache size in bytes
     * @param lvl Cache level 1..4
    */
    uint32_t getDataCacheSize(uint32_t lvl) const {
        if (lvl == 0 || lvl > maxNumberCacheLevels) XBYAK_RISCV_THROW(ERR_BAD_PARAMETER);
        return dataCacheSize_[lvl - 1];
    }

    /**
     * Get data cache line size in bytes
     * @param lvl Cache level 1..4
    */
    uint32_t getDataCacheLineSize(uint32_t lvl) const {
        if (lvl == 0 || lvl > maxNumberCacheLevels) XBYAK_RISCV_THROW(ERR_BAD_PARAMETER);
        return dataCacheLineSize_[lvl - 1];
    }

private:
    uint64_t hwcapFeatures = 0;
    static constexpr size_t maxNumberCacheLevels = 4;
    uint32_t dataCacheSize_[maxNumberCacheLevels] = {0, 0, 0, 0};
    uint32_t dataCacheLineSize_[maxNumberCacheLevels] = {0, 0, 0, 0};
    uint32_t numCores = 0;
    uint32_t xlen = 0;
    uint32_t vlen = 0;
    uint32_t flen = 0;
};

} // Xbyak_riscv
