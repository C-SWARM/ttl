# Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
# University of Notre Dame
# Indiana University
# University of Washington
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
cmake_minimum_required(VERSION 2.8.2)

# project specification
project(ttl CXX)

# -----------------------------------------------------------------------------
# Configure-time project options.
# -----------------------------------------------------------------------------
option(TTL_WITH_LAPACK "use an external lapack implementation for solve/inverse"
  OFF)

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
if (TTL_WITH_LAPACK)
  CHECK_INCLUDE_FILES(mkl.h TTL_HAVE_MKL_H)
  include(FindLAPACK)
endif (TTL_WITH_LAPACK)

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
