cmake_minimum_required(VERSION 3.12)
project(csparse VERSION 1.2.0)

set(HEADER_FILES
    csparse.h
    ldl.h
    UFconfig.h)

set(SOURCE_FILES
    csparse.c
    ldl.c)

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/CSparse>")

if(WIN32)
    # remove warnings about deprecation (CRT,etc)
    target_compile_options(${PROJECT_NAME} PRIVATE "/wd4996")
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME CSparse
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "extlibs/CSparse"
    )
