//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <thread>

extern int l1_cache_size;
extern int l2_cache_size;
extern int force_cpu;

#ifdef __linux__
#include <sys/auxv.h>

/* Get HWCAP bits from asm/hwcap.h */
#include <asm/hwcap.h>
#endif // __linux__

/* Make sure the bits we care about are defined, just in case asm/hwcap.h is
 * out of date (or for bare metal mode) */
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP      (1 << 10)
#endif

#ifndef HWCAP_CPUID
#define HWCAP_CPUID        (1 << 11)
#endif

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP      (1 << 20)
#endif

#ifndef HWCAP_SVE
#define HWCAP_SVE          (1 << 22)
#endif

#ifndef HWCAP_ASIMDFHM
#define HWCAP_ASIMDFHM     (1 << 23)
#endif

#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2         (1 << 1)
#endif

#ifndef HWCAP2_SVEI8MM
#define HWCAP2_SVEI8MM      (1 << 9)
#endif

#ifndef HWCAP2_SVEF32MM
#define HWCAP2_SVEF32MM    (1 << 10)
#endif

#ifndef HWCAP2_SVEBF16
#define HWCAP2_SVEBF16     (1 << 12)
#endif

#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM        (1 << 13)
#endif

#ifndef HWCAP2_BF16
#define HWCAP2_BF16        (1 << 14)
#endif

#ifndef HWCAP2_SME
#define HWCAP2_SME         (1 << 23)
#endif

#ifndef HWCAP2_SME_F64F64
#define HWCAP2_SME_F64F64  (1 << 25)
#endif

#ifndef HWCAP2_SME_I8I32
#define HWCAP2_SME_I8I32   (1 << 26)
#endif

#ifndef HWCAP2_SME_F16F32
#define HWCAP2_SME_F16F32  (1 << 27)
#endif

#ifndef HWCAP2_SME_B16F32
#define HWCAP2_SME_B16F32  (1 << 28)
#endif

#ifndef HWCAP2_SME_F32F32
#define HWCAP2_SME_F32F32  (1 << 29)
#endif

#ifndef HWCAP2_SME2
#define HWCAP2_SME2        (1UL << 37)
#endif

#ifndef HWCAP2_SME_B16B16
#define HWCAP2_SME_B16B16  (1UL << 41)
#endif

#ifndef HWCAP2_SME_F16F16
#define HWCAP2_SME_F16F16  (1UL << 42)
#endif

#ifndef HWCAP2_SME_F8F32
#define HWCAP2_SME_F8F32   (1UL << 59)
#endif

//unsigned int get_cpu_impl();

/*
 * As the android standard library is missing some C++11 features, use this
 * method to extract values from std::strings
 */
inline unsigned long string_to_value(const std::string &str, int base=0) {
    return strtoul(str.c_str(), nullptr, base);
}

/*
 * Use sysctl to check for feature support.
 */
#ifdef __APPLE__
static bool sysctl_feature_supported(const char *name)
{
    int    value = 0;
    size_t len   = sizeof(value);

    if (sysctlbyname(name, &value, &len, nullptr, 0) == 0)
    {
        // sysctl exists, interpret non-zero as “supported”
        return value != 0;
    }
    return false;
}
#endif

/* CPU models - we only need to detect CPUs we have
 * microarchitecture-specific code for.
 *
 * Architecture features are detected via HWCAPs.
 */
enum class CPUModel {
    GENERIC    = 0x0001,
    A53        = 0x0010,
    A55r0      = 0x0011,
    A55r1      = 0x0012,
    A35        = 0x0013,
    A510       = 0x0014,
    X1         = 0x0020,
    A73        = 0x0021,
    V1         = 0x0030,
    A64FX      = 0x1001
};

class CPUInfo
{
private:
    struct PerCPUData {
        CPUModel  model      = CPUModel::GENERIC;
        uint32_t  midr       = 0;
        bool      model_set  = false;
    };

    std::vector<PerCPUData> _percpu;

#ifndef BARE_METAL
    bool _cpuid   = false;
#endif
    bool _fp16    = false;
    bool _dotprod = false;
    bool _fhm     = false;
    bool _sve     = false;
    bool _sve2    = false;
    bool _svei8mm = false;
    bool _svef32mm = false;
    bool _svebf16 = false;
    bool _i8mm = false;
    bool _bf16 = false;
    bool _sme = false;
    bool _sme2 = false;
    bool _sme_f64f64 = false;
    bool _sme_i8i32 = false;
    bool _sme_f16f32 = false;
    bool _sme_b16f32 = false;
    bool _sme_f32f32 = false;
    bool _sme_f16f16 = false;
    bool _sme_b16b16 = false;
    bool _sme_f8f32 = false;
    bool _f8f32mm = false;

    unsigned int L1_cache_size = 32768;
    unsigned int L2_cache_size = 262144;

    void check_dot_allowlist(const unsigned long midr) {
#if 0
        /* Pre-4.15 kernels don't have the ASIMDDP bit.
         *
         * Although the CPUID bit allows us to read the feature register
         * directly, the kernel quite sensibly masks this to only show
         * features known by it to be safe to show to userspace.  As a
         * result, pre-4.15 kernels won't show the relevant bit in the
         * feature registers either.
         *
         * So for now, use a allowlist of CPUs known to support the feature.
         */
        if (!_dotprod) {
            const unsigned int dotprod_allowlist_masks[]  = { 0xfff0fff0, 0xff00fff0, 0xfff0fff0, 0xfff0fff0, 0xff00fff0, 0xff00ffc0, 0xff00ff00, 0 };
            const unsigned int dotprod_allowlist_values[] = { 0x4110d050, 0x4100d060, 0x4110d0a0, 0x4120d0a0, 0x4100d0b0, 0x4100d0c0, 0x4100d400, 0 };

            for (int i=0;dotprod_allowlist_values[i];i++) {
                if ((midr & dotprod_allowlist_masks[i]) == dotprod_allowlist_values[i]) {
                    _dotprod = true;
                    break;
                }
            }
        }
#endif // __aarch64__
    }

    /* Convert an MIDR register value to a CPUModel enum value. */
    CPUModel midr_to_model(const unsigned int midr) {
        CPUModel model;

        // Unpack variant and CPU ID
        int variant = (midr >> 20) & 0xF;
        int cpunum = (midr >> 4) & 0xFFF;
        int implementer = (midr >> 24) & 0xFF;

        /* Only CPUs we have code paths for are detected.  All other CPUs
         * can be safely classed as "GENERIC"
         */
        switch((implementer << 12) | cpunum) {
            case 0x46001:
                model = CPUModel::A64FX;
                break;

            case 0x41d03:
                model = CPUModel::A53;
                break;

            case 0x41d04:
                model = CPUModel::A35;
                break;

            case 0x41d05:
                if (variant) {
                    model = CPUModel::A55r1;
                } else {
                    model = CPUModel::A55r0;
                }
                break;

            case 0x41d80:
            case 0x41d46:
                model = CPUModel::A510;
                break;

            case 0x41d09:
                model = CPUModel::A73;
                break;

            case 0x41d44:
                model = CPUModel::X1;
                break;

            case 0x41d40:
                model = CPUModel::V1;
                break;

            default:
                model = CPUModel::GENERIC;
                break;
        }

        return model;
    }

    /* If the CPUID capability is present, MIDR information is provided in
       /sys.  Use that to populate the CPU model table.  */
    void populate_models_cpuid() {
        for (unsigned long int i=0; i<_percpu.size(); i++) {
            std::stringstream str;
            str << "/sys/devices/system/cpu/cpu" << i << "/regs/identification/midr_el1";
            std::ifstream file;

            file.open(str.str(), std::ios::in);

            if (file.is_open()) {
                std::string line;

                if (bool(getline(file, line))) {
                    const unsigned long midr = string_to_value(line, 16);

                    _percpu[i].midr      = (midr & 0xffffffff);
                    _percpu[i].model     = midr_to_model(_percpu[i].midr);
                    _percpu[i].model_set = true;

                    /* Check if this CPU should be allowlisted for dot product support. */
                    check_dot_allowlist(midr & 0xffffffff);
                }
            }
        }
    }

    void populate_models_baremetal() {
#ifdef __aarch64__
        // Assume single CPU in bare metal mode.  Just read the ID register and feature bits directly.
        uint64_t fr0, pfr0, fr1, midr, pfr1, fpfr0;

        __asm __volatile (
            "MRS  %0, ID_AA64ISAR0_EL1\n"
            "MRS  %1, ID_AA64PFR0_EL1\n"
            "MRS  %2, midr_el1\n"
            "MRS  %3, ID_AA64ISAR1_EL1\n"
            "MRS  %4, ID_AA64PFR1_EL1\n"
            "MRS  %5, s3_0_c0_c4_7\n"
            : "=r" (fr0), "=r" (pfr0), "=r" (midr), "=r" (fr1), "=r" (pfr1), "=r" (fpfr0)
        );

//        printf("baremetal detect: ISAR0 = %lx, ISAR1 = %lx, PFR0 = %lx\n",fr0,fr1,pfr0);

        if ((fr0 >> 44) & 0xf) {
            _dotprod = true;
        }

        if ((fr0 >> 48) & 0xf) {
            _fhm = true;
        }

        if ((pfr0 >> 16) & 0xf) {
            _fp16 = true;
        }

        if ((pfr0 >> 32) & 0xf) {
            uint64_t svefr0;
            _sve = true;
            __asm __volatile (
                ".inst 0xd5380483 // mrs     x3, id_aa64zfr0_el1\n"
                "MOV  %0, X3"
                : "=r" (svefr0)
                :
                : "x3"
            );
//            printf("SVE detect: %lx\n",svefr0);
            if (svefr0 & 0xF) {
                _sve2 = true;
            }
            if ((svefr0 >> 20) & 0xF) {
                _svebf16 = true;
                _bf16 = true; // Hack for Klein EBM
            }
            if ((svefr0 >> 44) & 0xF) {
                _svei8mm = true;
            }
            if ((svefr0 >> 52) & 0xF) {
                _svef32mm = true;
            }
        }

        if ((fr1 >> 44) & 0xF) {
            _bf16 = true;
        }

        if ((fr1 >> 52) & 0xF) {
            _i8mm = true;
        }

        if ((fpfr0 >> 27) & 0x1) {
            _f8f32mm = true;
        }

        // SME detection - find version and supported ops.
        uint64_t sme_field = (pfr1 >> 24) & 0xF;
        if (sme_field) {
            uint64_t smefr0;
            __asm __volatile (
                ".inst 0xd53804a3 // mrs     x3, id_aa64smfr0_el1\n"
                "MOV  %0, X3"
                : "=r" (smefr0)
                :
                : "x3"
            );
            _sme = true;
            if (sme_field > 1) {
                _sme2 = true;
            }

            // Bit 32: F32F32
            if (smefr0 & (1ULL << 32)) {
                _sme_f32f32 = true;
            }

            // Bit 34: B16F32
            if (smefr0 & (1ULL << 34)) {
                _sme_b16f32 = true;
            }

            // Bit 35: F16F32
            if (smefr0 & (1ULL << 35)) {
                _sme_f16f32 = true;
            }

            // Bits 36-39: I8I32
            if (((smefr0 >> 36) & 0xF) == 0xF) {
                _sme_i8i32 = true;
            }

            // Bit 40: F8F32
            if (smefr0 & (1UL << 40)) {
                _sme_f8f32 = true;
            }

            // Bit 42: FP16FP16 ops
            if (smefr0 & (1ULL << 42)) {
                _sme_f16f16 = true;
            }

            // Bit 43: BF16BF16 ops
            if (smefr0 & (1ULL << 43)) {
                _sme_b16b16 = true;
            }

            // Bit 48: F64F64 ops
            if (smefr0 & (1ULL << 48)) {
                _sme_f64f64 = true;
            }
        }

        midr &= 0xffffffff;

        _percpu[0].midr = midr;
        _percpu[0].model = midr_to_model(midr);
        _percpu[0].model_set = true;
#endif // aarch64
    }

    /* If "long-form" cpuinfo is present, parse that to populate models. */
    void populate_models_cpuinfo() {
        std::regex   proc_regex("^processor.*(\\d+)$");
        std::regex   imp_regex("^CPU implementer.*0x(..)$");
        std::regex   var_regex("^CPU variant.*0x(.)$");
        std::regex   part_regex("^CPU part.*0x(...)$");
        std::regex   rev_regex("^CPU revision.*(\\d+)$");

        std::ifstream file;
        file.open("/proc/cpuinfo", std::ios::in);

        if (file.is_open()) {
            std::string line;
            int midr=0;
            int curcpu=-1;

            while(bool(getline(file, line))) {
                std::smatch match;

                if (std::regex_match(line, match, proc_regex)) {
                    std::string id = match[1];
                    int newcpu=string_to_value(id, 0);

                    if (curcpu >= 0 && midr==0) {
                        // Matched a new CPU ID without any description of the previous one - looks like old format.
                        return;
                    }

                    if (curcpu >= 0) {
                        _percpu[curcpu].midr      = midr;
                        _percpu[curcpu].model     = midr_to_model(midr);
                        _percpu[curcpu].model_set = true;

                        /* Check if this CPU should be allowlisted for dot product support. */
                        check_dot_allowlist(midr);
//                        printf("CPU %d: %x\n",curcpu,midr);
                    }

                    midr=0;
                    curcpu=newcpu;

                    continue;
                }

                if (std::regex_match(line, match, imp_regex)) {
                    int impv = string_to_value(match[1], 16);
                    midr |= (impv << 24);
                    continue;
                }

                if (std::regex_match(line, match, var_regex)) {
                    int varv = string_to_value(match[1], 16);
                    midr |= (varv << 20);
                    continue;
                }

                if (std::regex_match(line, match, part_regex)) {
                    int partv = string_to_value(match[1], 16);
                    midr |= (partv << 4);
                    continue;
                }

                if (std::regex_match(line, match, rev_regex)) {
                    int regv = string_to_value(match[1], 10);
                    midr |= (regv);
                    midr |= (0xf << 16);
                    continue;
                }
            }

            if (curcpu >= 0) {
                _percpu[curcpu].midr      = midr;
                _percpu[curcpu].model     = midr_to_model(midr);
                _percpu[curcpu].model_set = true;

//                printf("CPU %d: %x\n",curcpu,midr);
            }
        }
    }

    /* Identify the maximum valid CPUID in the system.  This reads
     * /sys/devices/system/cpu/present to get the information.  */
    int get_max_cpus() {
        int max_cpus = 1;

#ifndef BARE_METAL
        std::ifstream CPUspresent;
        CPUspresent.open("/sys/devices/system/cpu/present", std::ios::in);
        bool success = false;

        if (CPUspresent.is_open()) {
            std::string line;

            if (bool(getline(CPUspresent, line))) {
                /* The content of this file is a list of ranges or single values, e.g.
                 * 0-5, or 1-3,5,7 or similar.  As we are interested in the
                 * max valid ID, we just need to find the last valid
                 * delimiter ('-' or ',') and parse the integer immediately after that.
                 */
                auto startfrom=line.begin();

                for (auto i=line.begin(); i<line.end(); ++i) {
                    if (*i=='-' || *i==',') {
                        startfrom=i+1;
                    }
                }

                line.erase(line.begin(), startfrom);

                max_cpus = string_to_value(line, 10) + 1;
                success = true;
            }
        }

        // Return std::thread::hardware_concurrency() as a fallback.
        if (!success) {
            max_cpus = std::thread::hardware_concurrency();
        }
#endif // !BARE_METAL

        return max_cpus;
    }

public:
    CPUInfo() {
        _percpu.resize(get_max_cpus());

#ifdef __APPLE__
        _fp16       = sysctl_feature_supported("hw.optional.arm.FEAT_FP16");
        _dotprod    = sysctl_feature_supported("hw.optional.arm.FEAT_DotProd");
        _fhm        = sysctl_feature_supported("hw.optional.arm.FEAT_FHM");
        _i8mm       = sysctl_feature_supported("hw.optional.arm.FEAT_I8MM");
        _bf16       = sysctl_feature_supported("hw.optional.arm.FEAT_BF16");
        _sme        = sysctl_feature_supported("hw.optional.arm.FEAT_SME");
        _sme2       = sysctl_feature_supported("hw.optional.arm.FEAT_SME2");
        _sme_f64f64 = sysctl_feature_supported("hw.optional.arm.FEAT_SME_F64F64");
        _sme_i8i32  = sysctl_feature_supported("hw.optional.arm.SME_I8I32");
        _sme_f16f32 = sysctl_feature_supported("hw.optional.arm.SME_F16F32");
        _sme_b16f32 = sysctl_feature_supported("hw.optional.arm.SME_B16F32");
        _sme_f32f32 = sysctl_feature_supported("hw.optional.arm.SME_F32F32");
        _sme_f16f16 = sysctl_feature_supported("hw.optional.arm.FEAT_SME_F16F16");
        _sme_b16b16 = sysctl_feature_supported("hw.optional.arm.FEAT_SME_B16B16");
        _sme_f8f32  = false;

        _f8f32mm = false;

        return;
#endif // __APPLE__

#ifndef BARE_METAL
#ifdef __linux__
        unsigned long hwcaps = getauxval(AT_HWCAP);
        unsigned long hwcaps2 = getauxval(AT_HWCAP2);
#else
        unsigned long hwcaps = 0;
        unsigned long hwcaps2 = 0;
#endif // __linux__

#ifdef __aarch64__
        // These are AArch64 specific HWCAP flags.
        if (hwcaps & HWCAP_CPUID) {
            _cpuid = true;
        }

        if (hwcaps & HWCAP_ASIMDHP) {
            _fp16 = true;
        }

        if (hwcaps & HWCAP_ASIMDDP) {
            _dotprod = true;
        }

        if (hwcaps & HWCAP_ASIMDFHM) {
            _fhm = true;
        }

        if (hwcaps & HWCAP_SVE) {
            _sve = true;
        }

        if (hwcaps2 & HWCAP2_SVE2) {
            _sve2 = true;
        }

        if (hwcaps2 & HWCAP2_SVEI8MM) {
            _svei8mm = true;
        }

        if (hwcaps2 & HWCAP2_SVEF32MM) {
            _svef32mm = true;
        }

        if (hwcaps2 & HWCAP2_SVEBF16) {
            _svebf16 = true;
        }

        if (hwcaps2 & HWCAP2_I8MM) {
            _i8mm = true;
        }

        if (hwcaps2 & HWCAP2_BF16) {
            _bf16 = true;
        }

        if (hwcaps2 & HWCAP2_SME) {
            _sme = true;
        }

        if (hwcaps2 & HWCAP2_SME_F64F64) {
            _sme_f64f64 = true;
        }

        if (hwcaps2 & HWCAP2_SME_I8I32) {
            _sme_i8i32 = true;
        }

        if (hwcaps2 & HWCAP2_SME_F16F32) {
            _sme_f16f32 = true;
        }

        if (hwcaps2 & HWCAP2_SME_B16F32) {
            _sme_b16f32 = true;
        }

        if (hwcaps2 & HWCAP2_SME_F32F32) {
            _sme_f32f32 = true;
        }

        if (hwcaps2 & HWCAP2_SME2) {
            _sme2 = true;
        }

        if (hwcaps2 & HWCAP2_SME_B16B16) {
            _sme_b16b16 = true;
        }

        if (hwcaps2 & HWCAP2_SME_F16F16) {
            _sme_f16f16 = true;
        }

        if (hwcaps2 & HWCAP2_SME_F8F32) {
            _sme_f8f32 = true;
        }
#endif // __aarch64__

        if (_cpuid) {
            populate_models_cpuid();
        } else {
            populate_models_cpuinfo();
        }
#else
        populate_models_baremetal();
#endif // BARE_METAL
    }

    void set_cpu_model(unsigned long cpuid, CPUModel model) {
        if (_percpu.size() > cpuid) {
            _percpu[cpuid].model     = model;
            _percpu[cpuid].model_set = true;
        }
    }

#define FORCE_FEAT_IMPL(x) \
    if (!strcmp(feature, #x)) { \
        _##x = value; \
        return true; \
    }

    bool force_feature(const char *feature, bool value) {
        FORCE_FEAT_IMPL(fp16);
        FORCE_FEAT_IMPL(dotprod);
        FORCE_FEAT_IMPL(fhm);
        FORCE_FEAT_IMPL(sve);
        FORCE_FEAT_IMPL(sve2);
        FORCE_FEAT_IMPL(bf16);
        FORCE_FEAT_IMPL(i8mm);
        FORCE_FEAT_IMPL(svebf16);
        FORCE_FEAT_IMPL(svei8mm);
        FORCE_FEAT_IMPL(svef32mm);
        FORCE_FEAT_IMPL(sme);
        FORCE_FEAT_IMPL(sme_f64f64);
        FORCE_FEAT_IMPL(sme_i8i32);
        FORCE_FEAT_IMPL(sme_f16f32);
        FORCE_FEAT_IMPL(sme_b16f32);
        FORCE_FEAT_IMPL(sme_f32f32);
        FORCE_FEAT_IMPL(sme2);
        FORCE_FEAT_IMPL(sme_b16b16);
        FORCE_FEAT_IMPL(sme_f16f16);
        FORCE_FEAT_IMPL(sme_f8f32);
        FORCE_FEAT_IMPL(f8f32mm);

        return false;
    }

#undef FORCE_FEAT_IMPL

    void midr_override(const unsigned long midr) {
        check_dot_allowlist(midr);

        for(size_t i=0; i<_percpu.size(); i++) {
            _percpu[i].midr      = midr;
            _percpu[i].model     = midr_to_model(midr);
            _percpu[i].model_set = true;
        }
    }

    bool has_fp16() const {
        return _fp16;
    }

    bool has_dotprod() const {
        return _dotprod;
    }

    bool has_fhm() const {
        return _fhm;
    }

    bool has_sve() const {
        return _sve;
    }

    bool has_sve2() const {
        return _sve && _sve2;
    }

    bool has_svei8mm() const {
        return _sve && _svei8mm;
    }

    bool has_svef32mm() const {
        return _sve && _svef32mm;
    }

    bool has_svebf16() const {
        return _sve && _svebf16;
    }

    bool has_i8mm() const {
        return _i8mm;
    }

    bool has_bf16() const {
        return _bf16;
    }

    bool has_sme() const {
        return _sme;
    }

    bool has_sme_f64f64() const {
        return _sme && _sme_f64f64;
    }

    bool has_sme_i8i32() const {
        return _sme && _sme_i8i32;
    }

    bool has_sme_f16f32() const {
        return _sme && _sme_f16f32;
    }

    bool has_sme_b16f32() const {
        return _sme && _sme_b16f32;
    }

    bool has_sme_f32f32() const {
        return _sme && _sme_f32f32;
    }

    bool has_sme2() const {
        return _sme && _sme2;
    }

    bool has_sme2_f16f16() const {
        return _sme2 && _sme_f16f16;
    }

    bool has_sme2_b16b16() const {
        return _sme2 && _sme_b16b16;
    }

    bool has_sme2_f8f32() const {
        return _sme2 && _sme_f8f32;
    }

    bool has_f8f32mm() const {
        return _f8f32mm;
    }

    CPUModel get_cpu_model(unsigned long cpuid) const {
        if (cpuid < _percpu.size()) {
            return _percpu[cpuid].model;
        }

        return CPUModel::GENERIC;
    }

    CPUModel get_cpu_model() const {
#ifdef __linux__
        return get_cpu_model(sched_getcpu());
#else
        return get_cpu_model(0);
#endif
    }

    unsigned int get_L1_cache_size() const {
        return L1_cache_size;
    }

    void set_L1_cache_size(unsigned int size) {
        L1_cache_size = size;
    }

    unsigned int get_L2_cache_size() const {
        return L2_cache_size;
    }

    void set_L2_cache_size(unsigned int size) {
        L2_cache_size = size;
    }
};

CPUInfo *get_CPUInfo();
