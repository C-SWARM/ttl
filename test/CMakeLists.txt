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

# Trivial examples and unit tests using gtest and gmock
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_executable(identity Identity.cpp)
add_executable(index Index.cpp)
add_executable(tensor Tensor.cpp)
add_executable(library Determinant.cpp Inverse.cpp Transpose.cpp)
add_executable(bind Bind.cpp)
add_executable(index_map IndexMap.cpp)
add_executable(operators Operators.cpp)
add_executable(scalar Scalar.cpp)
add_executable(trees Trees.cpp)
add_executable(epsilon Epsilon.cpp)
add_executable(delta Delta.cpp)

target_link_libraries(identity gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(tensor gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(library gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(bind gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(index_map gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(operators gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(scalar gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(trees gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(epsilon gtest_main ${LAPACK_LIBRARIES})
target_link_libraries(delta gtest_main ${LAPACK_LIBRARIES})


add_test(NAME identity_test COMMAND identity)
add_test(NAME index_test COMMAND index)
add_test(NAME tensor_test COMMAND tensor)
add_test(NAME library_test COMMAND library)
add_test(NAME bind_test COMMAND bind)
add_test(NAME index_map_test COMMAND index_map)
add_test(NAME operators_test COMMAND operators)
add_test(NAME scalar_test COMMAND scalar)
add_test(NAME trees_test COMMAND trees)
add_test(NAME epsilon_test COMMAND epsilon)
add_test(NAME delta_test COMMAND delta)
