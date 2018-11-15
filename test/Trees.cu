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
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;
static constexpr ttl::Index<'m'> m;

__global__
void test_lazy_tensor_product(int *a,int *b,int *c) {
    ttl::Tensor<2,2,int*> A = {a,{1, 2, 3, 4}}, B = {b,{2, 3, 4, 5}};
    ttl::Tensor<2,2,int*> C = {c};
    auto t0 = A(i,k);
    auto t1 = B(k,j);
    C(i,j) = t0 * t1;
}

TEST(Trees, LazyTensorProduct) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b, *c;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&c, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> C = {c};

// launch kernel
    test_lazy_tensor_product<<<1,1>>>(a,b,c);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(C[0][0], 10);
    EXPECT_EQ(C[0][1], 13);
    EXPECT_EQ(C[1][0], 22);
    EXPECT_EQ(C[1][1], 29);

// garbage collection
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

__global__
void test_lazy_tensor_product_complex(int *a,int *b,int *c,int *d,int *e) {
    ttl::Tensor<2,2,int*> A = {a,{1, 2, 3, 4}}, B = {b,{2, 3, 4, 5}},  C = {c,{3, 4, 5, 6}}, D = {d,{4, 5, 6, 7}};
    ttl::Tensor<2,2,int*> E = {e};
    auto t0 = A(i,j) * B(j,k);
    auto t1 = C(k,l) * D(l,m);
    E(i,m) = t0 * t1;
}

TEST(Trees, LazyTensorProductComplex) {
  
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b, *c, *d, *e;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&c, 4*sizeof(int));
    cudaMallocManaged(&d, 4*sizeof(int));
    cudaMallocManaged(&e, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> E = {e};

// launch kernel
    test_lazy_tensor_product_complex<<<1,1>>>(a,b,c,d,e);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(E[0][0], 1088);
    EXPECT_EQ(E[0][1], 1301);
    EXPECT_EQ(E[1][0], 2416);
    EXPECT_EQ(E[1][1], 2889);

// garbage collection
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(e);
}
