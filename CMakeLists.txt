cmake_minimum_required(VERSION 2.8.2)

# project specification
project(ttl)

# module include path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# included files
include(CTest)
include(DownloadProject)
include(GTest)

gtest()

include_directories(include)

add_subdirectory(include)
add_subdirectory(test)

