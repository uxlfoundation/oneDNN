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
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "xbyak_riscv_csr.hpp"
#include "xbyak_riscv.hpp"

#if defined(__linux__) && defined(__riscv)
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/utsname.h>
#include <asm/hwcap.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Xbyak_riscv {

// Legacy HWCAP constants
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

#if defined(__linux__) && defined(__riscv)
// Definitions for riscv_hwprobe (Linux 6.4+)
#ifndef __NR_riscv_hwprobe
#define __NR_riscv_hwprobe 258
#endif

#ifndef RISCV_HWPROBE_KEY_IMA_EXT_0
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#endif

#ifndef RISCV_HWPROBE_IMA_V
#define RISCV_HWPROBE_IMA_V (1ULL << 2)
#endif

#ifndef RISCV_HWPROBE_EXT_ZVBB
#define RISCV_HWPROBE_EXT_ZVBB (1ULL << 17)
#endif

#ifndef RISCV_HWPROBE_EXT_ZVBC
#define RISCV_HWPROBE_EXT_ZVBC (1ULL << 18)
#endif

#ifndef RISCV_HWPROBE_EXT_ZVKG
#define RISCV_HWPROBE_EXT_ZVKG (1ULL << 20)
#endif

#ifndef RISCV_HWPROBE_EXT_ZVFH
#define RISCV_HWPROBE_EXT_ZVFH (1ULL << 30)
#endif

struct riscv_hwprobe {
    int64_t key;
    uint64_t value;
};
#endif

enum class RISCVExtension : uint64_t {
    // 0-25: Legacy single-letter map (matches HWCAP for convenience)
    I = COMPAT_HWCAP_ISA_I,
    M = COMPAT_HWCAP_ISA_M,
    A = COMPAT_HWCAP_ISA_A,
    F = COMPAT_HWCAP_ISA_F,
    D = COMPAT_HWCAP_ISA_D,
    C = COMPAT_HWCAP_ISA_C,
    V = COMPAT_HWCAP_ISA_V,

    // 26+: Extended Z-extensions
    // Adding new extensions here is safe and conflict-free
    Zvfh = 1ULL << 26,
    Zvbb = 1ULL << 27,
    Zvbc = 1ULL << 28,
    Zvkg = 1ULL << 29
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
        hwcapFeatures = 0;
        xlen = sizeof(void*) * 8; // Fallback if sysconf fails
        
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

        // Try to use riscv_hwprobe to detect Z-extensions
        struct riscv_hwprobe requests[] = {
            {RISCV_HWPROBE_KEY_IMA_EXT_0, 0}
        };
        
        int ret = syscall(__NR_riscv_hwprobe, &requests, sizeof(requests) / sizeof(requests[0]), 0, NULL, 0);

        if (ret == 0) {
            uint64_t v = requests[0].value;
            // Update V support from hwprobe if present
            if (v & RISCV_HWPROBE_IMA_V) hwcapFeatures |= static_cast<uint64_t>(RISCVExtension::V);
            
            // Detect Z-extensions using the table
            for (const auto& entry : getExtensionTable()) {
                if (v & entry.hwprobe_bit) {
                    hwcapFeatures |= static_cast<uint64_t>(entry.id);
                }
            }
        } else {
            // Fallback: If hwprobe failed (Kernel < 6.4), use procfs
            // We only try this if the base Vector extension is present.
            if (hasExtension(RISCVExtension::V)) {
                detect_extensions_procfs();
            }
        }

        // Set xlen, number of cores, cache info
        xlen = sysconf(_SC_LONG_BIT);
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

    /**
     * Helper structure for extension mapping
    */
    struct ExtensionEntry {
        RISCVExtension id;
        uint64_t hwprobe_bit;  // Bit in RISCV_HWPROBE_KEY_IMA_EXT_0
        const char* name;      // String in /proc/cpuinfo "isa" line
    };

    /**
     * Centralized table for all supported Z-extensions
    */
    static const std::vector<ExtensionEntry>& getExtensionTable() {
        static const std::vector<ExtensionEntry> table = {
            { RISCVExtension::Zvfh, RISCV_HWPROBE_EXT_ZVFH, "_zvfh" },
            { RISCVExtension::Zvbb, RISCV_HWPROBE_EXT_ZVBB, "_zvbb" },
            { RISCVExtension::Zvbc, RISCV_HWPROBE_EXT_ZVBC, "_zvbc" },
            { RISCVExtension::Zvkg, RISCV_HWPROBE_EXT_ZVKG, "_zvkg" }
        };
        return table;
    }

    /**
     * Helper to detect all extensions in the table from /proc/cpuinfo
    */
    void detect_extensions_procfs() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        if (!cpuinfo.is_open()) return;

        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("isa") == 0) {
                for (const auto& entry : getExtensionTable()) {
                    if (line.find(entry.name) != std::string::npos) {
                        hwcapFeatures |= static_cast<uint64_t>(entry.id);
                    }
                }
                // We only parse the first valid 'isa' line we find.
                break; 
            }
        }
    }
};

} // Xbyak_riscv
