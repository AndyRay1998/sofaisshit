cmake_minimum_required(VERSION 3.12)

project(CImg)

install(FILES
        CImg.h
        DESTINATION include/CImg
        COMPONENT CImg_headers)

install(DIRECTORY
        plugins
        DESTINATION include/CImg
        COMPONENT CImg_headers)


include(CMakePackageConfigHelpers)

### CImgConfig.cmake
## Build tree
set(CIMG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
configure_file(CImgConfig.cmake.in ${CMAKE_BINARY_DIR}/lib/cmake/CImgConfig.cmake)

## Install tree
set(CIMG_INCLUDE_DIR include)
configure_file(CImgConfig.cmake.in InstalledCImgConfig.cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InstalledCImgConfig.cmake
        DESTINATION lib/cmake/CImg
        RENAME CImgConfig.cmake)
