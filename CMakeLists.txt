cmake_minimum_required(VERSION 2.8.2)

# project specification
project(ttl)

# module include path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  
# included files
include(CTest)
include(DownloadProject)
include(GTest)

gtest()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(test)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX})

CHECK_SYMBOL_EXISTS(LAPACKE_dgetrf mkl.h HAVE_MKL)
CHECK_SYMBOL_EXISTS(LAPACKE_dgetrf lapacke.h HAVE_LAPACKE)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/include/ttl/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ttl/config.h)
