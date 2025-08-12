/*
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ze_loader.h
 */

#ifndef _ZE_LOADER_H
#define _ZE_LOADER_H
#if defined(__cplusplus)
#pragma once
#endif

#include "../ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

 typedef struct _zel_version {
    int major;
    int minor;
    int patch; 
  } zel_version_t; 

 //Ex component string "ze_tracing", "ze_validation", etc 
#define ZEL_COMPONENT_STRING_SIZE 64 

 typedef struct zel_component_version {
    char component_name[ZEL_COMPONENT_STRING_SIZE];
    ze_api_version_t spec_version;
    zel_version_t component_lib_version;
} zel_component_version_t; 


ZE_APIEXPORT ze_result_t ZE_APICALL
zelLoaderGetVersions(
   size_t *num_elems,                     //Pointer to num versions to get.  
   zel_component_version_t *versions);    //Pointer to array of versions. If set to NULL, num_elems is returned

typedef enum _zel_handle_type_t {
   ZEL_HANDLE_DRIVER,
   ZEL_HANDLE_DEVICE,
   ZEL_HANDLE_CONTEXT,
   ZEL_HANDLE_COMMAND_QUEUE,
   ZEL_HANDLE_COMMAND_LIST,
   ZEL_HANDLE_FENCE,
   ZEL_HANDLE_EVENT_POOL,
   ZEL_HANDLE_EVENT,
   ZEL_HANDLE_IMAGE,
   ZEL_HANDLE_MODULE,
   ZEL_HANDLE_MODULE_BUILD_LOG,
   ZEL_HANDLE_KERNEL,
   ZEL_HANDLE_SAMPLER,
   ZEL_HANDLE_PHYSICAL_MEM
} zel_handle_type_t;

//Translates Loader Handles to Driver Handles if loader handle intercept is enabled.
//If handle intercept is not enabled handleOut is set to handleIn  
ZE_APIEXPORT ze_result_t ZE_APICALL
zelLoaderTranslateHandle(
   zel_handle_type_t handleType,   //Handle Type
   void *handleIn,                  //Input: handle to translate from loader handle to driver handle
   void **handleOut);                //Output: Pointer to handleOut is set to driver handle if successful

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for handling calls to released drivers in teardown.
///
ZE_DLLEXPORT ze_result_t ZE_APICALL
zelSetDriverTeardown();

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for informing the loader to delay teardown of its context until the call to zelLoaderContextTeardown().Only applies during static loader usage.
/// NOTE: This function is a work around for legacy stacks that use L0 apis after the application is already in teardown. Unless you need to use the L0 apis during teardown, do not use this function. 
ZE_DLLEXPORT void ZE_APICALL
zelSetDelayLoaderContextTeardown();

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for explicitly tearing down the loader's context. Only applies during static loader usage.
/// NOTE: This function is a work around for legacy stacks that use L0 apis after the application is already in teardown. Unless you need to use the L0 apis during teardown, do not use this function. 
ZE_DLLEXPORT void ZE_APICALL
zelLoaderContextTeardown();

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for Enabling the Tracing Layer During Runtime.
///
ZE_DLLEXPORT ze_result_t ZE_APICALL
zelEnableTracingLayer();

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for Checking if the Loader is torndown.
///
ZE_DLLEXPORT bool ZE_APICALL
zelCheckIsLoaderInTearDown();

typedef void (*zel_loader_teardown_callback_t)();
typedef void (*zel_application_teardown_callback_t)(uint32_t index);

/**
 * @brief Registers a teardown callback to be invoked during loader teardown.
 *
 * This function allows the application to register a callback function that will be called
 * when the loader is being torn down. The loader will also provide its own callback function
 * and assign an index to the registered callback.
 *
 * The application_callback is required to be a function that takes no arguments and returns void.
 * In addition, the application_callback should be thread-safe and not block to prevent deadlocking the
 * loader teardown process.
 *
 * For example, the application_callback used by the static loader is:
 *  void staticLoaderTeardownCallback() {
 *    loaderTeardownCallbackReceived = true;
 *  }
 * The application_callback should provide a simple notification to the application that the loader is being torn down.
 *
 * @param[in] application_callback  Application's callback function to be called during loader teardown.
 * @param[out] loader_callback      Pointer to the loader's callback function.
 * @param[out] index                Index assigned to the registered callback.
 *
 * @return
 *     - ZE_RESULT_SUCCESS if the callback was successfully registered.
 *     - Appropriate error code otherwise.
 */
ZE_DLLEXPORT ze_result_t ZE_APICALL
zelRegisterTeardownCallback(
   zel_loader_teardown_callback_t application_callback, // [in] Application's callback function to be called during loader teardown
   zel_application_teardown_callback_t *loader_callback, // [out] Pointer to the loader's callback function
   uint32_t *index // [out] Index assigned to the registered callback
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for Disabling the Tracing Layer During Runtime.
///
ZE_DLLEXPORT ze_result_t ZE_APICALL
zelDisableTracingLayer();

#if defined(__cplusplus)
} // extern "C"
#endif
#endif //_ZE_LOADER_H