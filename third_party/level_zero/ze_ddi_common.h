/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ze_ddi_common.h
 * @version v1.13-r1.13.1
 *
 */
#ifndef _ZE_DDI_COMMON_H
#define _ZE_DDI_COMMON_H
#if defined(__cplusplus)
#pragma once
#endif
#include "ze_ddi.h"
#include "zet_ddi.h"
#include "zes_ddi.h"

#if defined(__cplusplus)
extern "C" {
#endif

/// @brief Handle with pointers to Dispatch Tables allocated by the driver at the beginning of every L0 object handle.
typedef struct _ze_handle_t
{
    ze_dditable_driver_t *pCore; // [in] pointer to _ze_dditable_t_ object related to this handle
    zet_dditable_driver_t *pTools; // [in] pointer to _zet_dditable_t_ object related to this handle
    zes_dditable_driver_t *pSysman; // [in] pointer to _zes_dditable_t_ object related to this handle
} ze_handle_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_DDI_COMMON_H