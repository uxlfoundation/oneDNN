/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <sys/types.h>

#include "../common/c_types_map.hpp"
#include "../common/dnnl_thread.hpp"
#include "../common/type_helpers.hpp"
#include "../common/utils.hpp"

#include "my_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace my_utils {

using namespace dnnl::impl::utils;

int read_stack(const char *title, int ithr) {
#define MAXCHARL 4096
#ifndef _MSC_VER
    pid_t pid = getpid();
#else
    unsigned int pid = 0;
#endif
    printf("pid = %d\n", (int32_t)pid);
    fflush(nullptr);
    char path_buf[0x100] = {};
#ifdef _WIN32
    sprintf_s(path_buf, "/proc/%u/stack", pid);
#else
    sprintf(path_buf, "/proc/%u/stack", pid);
#endif

    FILE *fp;
    fp = fopen(path_buf, "r");
    if (fp == nullptr) {
        printf("Could not open file %s", path_buf);
        return 1;
    }

    char str[MAXCHARL];
    while (fgets(str, MAXCHARL, fp) != nullptr)
        printf("%s ithr = %d : %s", title, ithr, str);
    fclose(fp);
    return 0;
}

int read_pagemap(const char *title, int ithr, size_t virt_addr,
        unsigned long virt_size) {
#define PAGEMAP_ENTRY 8
#define GET_BIT(X, Y) ((X) & ((uint64_t)1 << (Y))) >> (Y)
#define GET_PFN(X) ((X) & 0x7FFFFFFFFFFFFF)

    const int __endian_bit = 1;
#define is_bigendian() ((*(char *)&__endian_bit) == 0)

#ifndef _MSC_VER
    pid_t pid = getpid();
#else
    unsigned int pid = 0;
#endif
    printf("%s ithr = %d pid = %d\n", title, ithr, (int32_t)pid);
    fflush(nullptr);

    int i, c, status;
    uint64_t read_val, file_offset;
    char path_buf[0x100] = {};
#ifdef _WIN32
    sprintf_s(path_buf, "/proc/%u/pagemap", pid);
#else
    sprintf(path_buf, "/proc/%u/pagemap", pid);
#endif

    FILE *f;
    //        char *end;

    printf("%s ithr = %d Big endian? %d\n", title, ithr, is_bigendian());
    f = fopen(path_buf, "rb");
    if (!f) {
        printf("%s ithr = %d Error! Cannot open %s\n", title, ithr, path_buf);
        return -1;
    }

    // Shifting by virt-addr-offset number of bytes
    // and multiplying by the size of an address (the size of an entry in pagemap
    // file)
#ifndef _MSC_VER
    int page_size = getpagesize();
#else
    int page_size = 4096;
#endif
    printf("%s ithr = %d Page_size: 0x%d, Entry_size: 0x%d\n", title, ithr,
            page_size, PAGEMAP_ENTRY);

    auto last_addr = virt_addr + virt_size;
    size_t curr_addr;
    int tot_swapped = 0;
    int tot_not_present = 0;
    for (curr_addr = virt_addr; curr_addr < last_addr; curr_addr += page_size) {
        file_offset = curr_addr / page_size * PAGEMAP_ENTRY;
        //      printf("Reading %s at 0x%llx\n", path_buf, (unsigned long long)file_offset);
        status = fseek(f, file_offset, SEEK_SET);
        if (status) {
            perror("Failed to do fseek!");
            return -1;
        }
        errno = 0;
        read_val = 0;
        unsigned char c_buf[PAGEMAP_ENTRY];
        for (i = 0; i < PAGEMAP_ENTRY; i++) {
            c = getc(f);
            if (c == EOF) {
                printf("\n%s ithr = %d Reached end of the file\n", title, ithr);
                return 0;
            }
            if (is_bigendian())
                c_buf[i] = c;
            else
                c_buf[PAGEMAP_ENTRY - i - 1] = c;
            //          printf("[%d]0x%x ", i, c);
        }
        for (i = 0; i < PAGEMAP_ENTRY; i++) {
            // printf("%d ",c_buf[i]);
            read_val = (read_val << 8) + c_buf[i];
        }
        //      printf("\n");
        //            printf(" Result: 0x%llx,", (unsigned long long)read_val);
        // if(GET_BIT(read_val, 63))
        if (GET_BIT(read_val, 63)) {
            printf("%s ithr = %d Vaddr: 0x%zx,PFN: 0x%llx,", title, ithr,
                    curr_addr, (unsigned long long)GET_PFN(read_val));
        } else {
            printf("%s ithr = %d Vaddr: 0x%zx, Page not present,", title, ithr,
                    curr_addr);
            tot_not_present++;
        }
        if (GET_BIT(read_val, 62)) {
            printf(" Page swapped");
            tot_swapped++;
        }
        printf("\n");
    }
    printf("Total: not presnt = %d swapped = %d\n", tot_not_present,
            tot_swapped);
    fclose(f);
    return 0;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bool get_env_num(const char *vname, int &value) {
    bool status = false;
    char *pString;
#ifdef _WIN32
    size_t len;
    auto err = _dupenv_s(&pString, &len, vname);
    if (err) return false;
#else
    pString = ::getenv(vname);
#endif

    if (pString == nullptr) return false;
    printf("\n!! mkldnn: The environment variable %s is: %s\n", vname, pString);
    int x = -1;
    x = atoi(pString);
    if (x != -1) {
        value = x;
        status = true;
    }
    return status;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

int is_debugged() {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "TracerPid:", 10) == 0) {
            int tracer_pid = atoi(line + 10);
            fclose(f);
            return tracer_pid != 0;
        }
    }

    fclose(f);
    return 0;
}

} // namespace my_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
