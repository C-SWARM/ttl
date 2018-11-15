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

include(GTest)

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

add_test(NAME bind_test COMMAND bind)
add_test(NAME delta_test COMMAND delta)
add_test(NAME epsilon_test COMMAND epsilon)
add_test(NAME identity_test COMMAND identity)
add_test(NAME index_test COMMAND index)
add_test(NAME index_map_test COMMAND index_map)
add_test(NAME library_test COMMAND library)
add_test(NAME operators_test COMMAND operators)
add_test(NAME scalar_test COMMAND scalar)
add_test(NAME tensor_test COMMAND tensor)
add_test(NAME trees_test COMMAND trees)

#Cuda tests

# completed and execute as expected (mostly)

add_executable(cuda_bind Bind.cu)
target_link_libraries(cuda_bind gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_bind_test COMMAND cuda_bind)

add_executable(cuda_delta Delta.cu)
target_link_libraries(cuda_delta gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_delta_test COMMAND cuda_delta)

add_executable(cuda_determinant Determinant.cu)
target_link_libraries(cuda_determinant gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_determinant_test COMMAND cuda_determinant)

add_executable(cuda_epsilon Epsilon.cu)
target_link_libraries(cuda_epsilon gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_epsilon_test COMMAND cuda_epsilon)

add_executable(cuda_identity Identity.cu)
target_link_libraries(cuda_identity gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_identity_test COMMAND cuda_identity)

add_executable(cuda_index Index.cu)
add_test(NAME cuda_index_test COMMAND cuda_index)

add_executable(cuda_index_map IndexMap.cu)
target_link_libraries(cuda_index_map gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_index_map_test COMMAND cuda_index_map)

add_executable(cuda_init Init.cu)
target_link_libraries(cuda_init gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_init_test COMMAND cuda_init)

add_executable(cuda_operators Operators.cu)
target_link_libraries(cuda_operators gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_operators_test COMMAND cuda_operators)

add_executable(cuda_scalar Scalar.cu)
target_link_libraries(cuda_scalar gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_scalar_test COMMAND cuda_scalar)

add_executable(cuda_tensor Tensor.cu)
target_link_libraries(cuda_tensor gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_tensor_test COMMAND cuda_tensor)

add_executable(cuda_transpose Transpose.cu)
target_link_libraries(cuda_transpose gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_transpose_test COMMAND cuda_transpose)

add_executable(cuda_trees Trees.cu)
target_link_libraries(cuda_trees gtest_main ${LAPACK_LIBRARIES})
add_test(NAME cuda_trees COMMAND cuda_trees)



# in progress

#add_executable(cuda_inverse Inverse.cu)
#target_link_libraries(cuda_inverse gtest_main ${LAPACK_LIBRARIES})
#add_test(NAME cuda_inverse_test COMMAND cuda_inverse)



# not tested - Vector.cpp and Quantum.cpp aren't tested in serial/aren't in the original cmakefile, so I've left them as-is

# add_executable(cuda_vector Vector.cu)
# target_link_libraries(cuda_vector gtest_main ${LAPACK_LIBRARIES})
# add_test(NAME cuda_vector COMMAND cuda_vector)

# add_executable(cuda_quantum Quantum.cu)
# target_link_libraries(cuda_quantum gtest_main ${LAPACK_LIBRARIES})
# add_test(NAME cuda_quantum_test COMMAND cuda_quantum)

