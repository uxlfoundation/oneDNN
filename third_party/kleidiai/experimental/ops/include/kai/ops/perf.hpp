//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/* Prototypes from perf.c */

void start_counter(int fd);
long long get_counter(int fd);
long long stop_counter(int fd);
int open_instruction_counter(void);
int open_cycle_counter(void);
