#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

for f in *.hpp; do
    if [ "$f" != "list.hpp" ]; then
        printf '#include "%s"\n' "$f"
    fi
done
