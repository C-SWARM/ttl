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

static constexpr ttl::Index<'i'> i;

  __global__
void test_scalar_tensor(double *a) {
    ttl::Tensor<0, 2, double> B = {1.2};
    *a = B[0];
}

TEST(Tensor, ScalarTensor) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    double *a;

//initialize memory on device and host
    cudaMallocManaged(&a, 1*sizeof(double));

// launch kernel
    test_scalar_tensor<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

//   validate results/do test
    EXPECT_EQ(*a, 1.2);

//garbage collect
    cudaFree(a);
}

// // __global__
// // void test_scalar_tensor(ttl::Tensor<0, 2, double> *a) {
// //     ttl::Tensor<0, 2, double> A = {1.2};
// //     a = A;
// // }

// // TEST(Tensor, ScalarTensor) {
// //     //Check for available device
// //         int nDevices;
// //         cudaGetDeviceCount(&nDevices);
// //         assert(nDevices > 0);

// //     //instantiate data pointers
// //         ttl::Tensor<0, 2, double> *a = new ttl::Tensor<0, 2, double>;

// //     //initialize memory on device and host
// //         cudaMallocManaged(&a, 1*sizeof(ttl::Tensor<0, 2, double>));

// //     //   allocate host tensor
// //         ttl::Tensor<0, 2, int*> A = {a};

// //     // launch kernel
// //         test_scalar_tensor<<<1,1>>>(a);

// //     // control for race conditions
// //         cudaDeviceSynchronize();

// //     // check errors
// //         cudaCheckErrors("failed");

// //     //   validate results/do test
// //         EXPECT_EQ(A[0], 1.2);

// //     //garbage collect
// //         cudaFree(a);
// // }

__global__
void test_scalar_assign(double *a) {
  ttl::Tensor<0, 2, double> A;
  A[0] = 1.4;

  *a = A[0];
}

TEST(Tensor, ScalarAssign) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    double *a;

//initialize memory on device and host
    cudaMallocManaged(&a, 1*sizeof(double));

// launch kernel
    test_scalar_assign<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// validate results/do test
    EXPECT_EQ(*a, 1.4);

// garbage collect
    cudaFree(a);
}

__global__
void test_scalar_expression(double *a) {
  ttl::Tensor<0, 2, double> A = {1.2};
  *a = A();
}

TEST(Tensor, ScalarExpression) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    double *a;

//initialize memory on device and host
    cudaMallocManaged(&a, 1*sizeof(double));

// launch kernel
    test_scalar_expression<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// validate results/do test
    EXPECT_EQ(*a, 1.2);
  
//garbage collect
    cudaFree(a);
}

__global__
void test_scalar_contraction(int *d) {
    ttl::Tensor<1, 2, int> A = {1, 2}, B = {5, 6};
    auto out = A(i)*B(i);
    *d = out;
}

TEST(Tensor, ScalarContraction) {
//Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

//instantiate data pointers
    int *d;

//initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

// launch kernel
    test_scalar_contraction<<<1,1>>>(d);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// validate results/do test
    EXPECT_EQ(*d, 17);
  
//garbage collect 
    cudaFree(d);
}
