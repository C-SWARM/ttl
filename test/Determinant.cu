// -------------------------------------------------------------------*- C++ -*-
// Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
// University of Notre Dame
// Indiana University
// University of Washington
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------
#include <ttl/ttl.h>
#include <gtest/gtest.h>
// updated from "const" to "static constexpr"
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;

#define cudaCheckErrors(msg)                                \
  do {                                                      \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",    \
              msg, cudaGetErrorString(__err),               \
              __FILE__, __LINE__);                          \
      fprintf(stderr, "*** FAILED - ABORTING\n");           \
      exit(1);                                              \
    }                                                       \
  } while (0)

__global__ 
void test_basic_2_2(int *d){
    ttl::Tensor<2,2,int> A = {1,2,3,4};
    *d = ttl::det(A);
}

TEST(Determinant, Basic_2_2) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_basic_2_2<<<1,1>>>(d);
    
// control for race conditions
    cudaDeviceSynchronize();
    
// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*d, -2);

// garbage
    cudaFree(d);
}

__global__
void test_external_2_2(int *d){
    int a[4];
    const ttl::Tensor<2,2,int*> A = {a, {1,2,3,4}};
    *d = ttl::det(A);
}

TEST(Determinant, External_2_2) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_external_2_2<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*d, -2);

// garbage
    cudaFree(d);
}

__global__
void test_R_Value_2_2(int *d){
    *d = ttl::det(ttl::Tensor<2,2,int>{1,2,3,4});
}

TEST(Determinant, RValue_2_2) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_R_Value_2_2<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*d, -2);

// garbage
    cudaFree(d);
}

__global__
void test_expression_r_val_2_2(int *d){
    ttl::Tensor<2,2,int> A = {1,2,3,4};
    *d = ttl::det(A(i,j));
}

TEST(Determinant, ExpressionRValue_2_2) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_expression_r_val_2_2<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*d, -2);

// garbage
    cudaFree(d);
}

__global__
void test_expression_2_2(int *d){
    ttl::Tensor<2,2,int> A = {1,2,3,4};
    auto e = A(i,j);
    *d = ttl::det(e);
}

TEST(Determinant, Expression_2_2) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_expression_2_2<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results  
    EXPECT_EQ(*d, -2);

// garbage
    cudaFree(d);
}

__global__
void test_expression_3_3(int *d){
    ttl::Tensor<2,3,int> A = {1,2,3,
                                4,5,6,
                                7,8,10};
    *d = ttl::det(A);
}

TEST(Determinant, 3_3) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_expression_3_3<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*d, -3);

// garbage
    cudaFree(d);
}
