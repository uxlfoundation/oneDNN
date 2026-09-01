#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

# Introduce the functions write_basic_package_version_file(...) and
# configure_package_config_file(...) into the current workspace
include(CMakePackageConfigHelpers)

# Set filenames and installation destinations
set(KLEIDIAI_OPS_CONFIG_FILE ${PROJECT_NAME}Config.cmake)
set(KLEIDIAI_OPS_CONFIG_VERSION_FILE ${PROJECT_NAME}ConfigVersion.cmake)
set(KLEIDIAI_OPS_TARGETS_NAME ${PROJECT_NAME}Targets)
set(KLEIDIAI_OPS_TARGETS_FILE "${CMAKE_CURRENT_BINARY_DIR}/${KLEIDIAI_OPS_TARGETS_NAME}.cmake")
set(KLEIDIAI_OPS_CONFIG_TEMPLATE_FILE "cmake/${KLEIDIAI_OPS_CONFIG_FILE}.in")
set(KLEIDIAI_OPS_CONFIG_OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${KLEIDIAI_OPS_CONFIG_FILE}")
set(KLEIDIAI_OPS_CONFIG_VERSION_OUTPUT_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/${KLEIDIAI_OPS_CONFIG_VERSION_FILE}")

# Install libraries
install(
    TARGETS ${KLEIDIAI_OPS_TARGETS_TO_INSTALL}
    EXPORT ${KLEIDIAI_OPS_TARGETS_NAME}
    RUNTIME DESTINATION ${KLEIDIAI_OPS_BIN_INSTALL_DIR}
    LIBRARY DESTINATION ${KLEIDIAI_OPS_LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${KLEIDIAI_OPS_LIB_INSTALL_DIR})

# Add all targets to the build-tree export set
export(
    EXPORT ${KLEIDIAI_OPS_TARGETS_NAME}
    NAMESPACE ${PROJECT_NAME}::
    FILE "${KLEIDIAI_OPS_TARGETS_FILE}")

# Install the exported targets to allow other projects to find this project with find_package()
install(
    EXPORT ${KLEIDIAI_OPS_TARGETS_NAME}
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${KLEIDIAI_OPS_CONFIG_INSTALL_DIR}")

# Install header files
#
# NOTE: The absence of a trailing slash is *important*; it means the folder itself is copied (as
# opposed to just the contents of the folder).
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/kai"
    DESTINATION "${KLEIDIAI_OPS_INCLUDE_INSTALL_DIR}"
    FILES_MATCHING
    PATTERN "*.hpp")

# Generate the *Config.cmake file for find_package()
configure_package_config_file(
    "${KLEIDIAI_OPS_CONFIG_TEMPLATE_FILE}" "${KLEIDIAI_OPS_CONFIG_OUTPUT_FILE}"
    INSTALL_DESTINATION ${KLEIDIAI_OPS_CONFIG_INSTALL_DIR})

# TODO: Agree on version compatibility. Semver? Or 'AnyNewerVersion' Configure *ConfigVersion.cmake
# which exports the version info
write_basic_package_version_file(
    "${KLEIDIAI_OPS_CONFIG_VERSION_OUTPUT_FILE}"
    VERSION ${${PROJECT_NAME}_VERSION}
    COMPATIBILITY SameMajorVersion)

# Install the generated *Config.cmake for find_package()
install(FILES "${KLEIDIAI_OPS_CONFIG_OUTPUT_FILE}" "${KLEIDIAI_OPS_CONFIG_VERSION_OUTPUT_FILE}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# -------------------------------------------------------------------------------------------------
