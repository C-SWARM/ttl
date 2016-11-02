cmake_minimum_required(VERSION 2.8.2)

# project specification
project(ttl CXX)

# module include path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

# -----------------------------------------------------------------------------
# Included functionality.
# -----------------------------------------------------------------------------
include(CTest)
include(DownloadProject)
include(GTest)
include(CheckIncludeFiles)

# -----------------------------------------------------------------------------
# Configure the gtest framework
# -----------------------------------------------------------------------------
gtest()

# -----------------------------------------------------------------------------
# Figure out what form of lapack we're going to include and link to.
# -----------------------------------------------------------------------------
CHECK_INCLUDE_FILES(mkl.h HAVE_MKL)
option(ENABLE_MKL "Use Intel's mkl.h header instead of lapacke.h" ${HAVE_MKL})
set(MKL_DEFS -DMKL_ILP64 CACHE STRING "Defines to set when including mkl")
set(MKL_LIBS -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential CACHE STRING
             "Libraries to link to set when including mkl")
if (ENABLE_MKL)
  add_definitions(${MKL_DEFS})
  set(LAPACK_LIBS ${MKL_LIBS})
else (ENABLE_MKL)
  set(LAPACK_LIBS -llapacke)
endif (ENABLE_MKL)

# -----------------------------------------------------------------------------
# Make sure that all of the include directories we need are actually included.
# -----------------------------------------------------------------------------
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(test)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX})

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ttl/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/ttl/config.h)
