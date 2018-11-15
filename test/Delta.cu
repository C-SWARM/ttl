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

__global__
void test_delta_1_1(int *d){
  ttl::Tensor<1,1,int*> D {d};
  D = ttl::delta(i);
}

TEST(Delta, 1_1) {
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *d;

  //initialize memory on device and host
    cudaMallocManaged(&d, 1*sizeof(int));

  // intialize verifying tensor
    ttl::Tensor<1, 1, int*> D = {d};

  // launch kernel
  test_delta_1_1<<<1,1>>>(d);
    
  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  // verify results
    EXPECT_EQ(D(0), 1);

  // garbage collection
    cudaFree(d);
}

__global__
void test_delta_2_2(int *d){
  ttl::Tensor<2,2,int*> D {d};
  D = ttl::delta(i,j);
}

TEST(Delta, 2_2) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *d;

//initialize memory on device and host
  cudaMallocManaged(&d, 4*sizeof(int));

// intialize verifying tensor
  ttl::Tensor<2, 2, int*> D = {d};

// launch kernel
  test_delta_2_2<<<1,1>>>(d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_EQ(D(0,0), 1);
  EXPECT_EQ(D(1,0), 0);
  EXPECT_EQ(D(0,1), 0);
  EXPECT_EQ(D(1,1), 1);

// garbage
  cudaFree(d);
}

__global__
void test_delta_2_3(int *d){
  ttl::Tensor<2,3,int*> D {d};
  D = ttl::delta(i,j);
}

TEST(Delta, 2_3) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *d;

//initialize memory on device and host
  cudaMallocManaged(&d, 4*sizeof(int));

// intialize verifying tensor
  ttl::Tensor<2, 3, int*> D = {d};

// launch kernel
  test_delta_2_3<<<1,1>>>(d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(D(i,j), (i == j) ? 1 : 0);
    }
  }

// garbage
  cudaFree(d);
}

__global__
void test_delta_3_2(int *d){
  ttl::Tensor<3,2,int*> D {d};
  D = ttl::delta(i,j,k);
}

TEST(Delta, 3_2) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *d;

//initialize memory on device and host
  cudaMallocManaged(&d, 8*sizeof(int));

// intialize verifying tensor
  ttl::Tensor<3, 2, int*> D = {d};

// launch kernel
  test_delta_3_2<<<1,1>>>(d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(D(i,j,k), (i == j && j == k) ? 1 : 0);
      }
    }
  }

// garbage
  cudaFree(d);
}

__global__
void test_delta_4_3(int *d){
  ttl::Tensor<4,3,int*> D {d};
  D = ttl::delta(i,j,k,l);
}

TEST(Delta, 4_3) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *d;

//initialize memory on device and host
  cudaMallocManaged(&d, 81*sizeof(int));

// intialize verifying tensor
  ttl::Tensor<4, 3, int*> D = {d};

// launch kernel
  test_delta_4_3<<<1,1>>>(d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          EXPECT_EQ(D(i,j,k,l), (i == j && j == k && k == l) ? 1 : 0);
        }
      }
    }
  }

// garbage
  cudaFree(d);
}

__global__
void test_delta_widen(int *d){
  ttl::Tensor<2,2,int*> D {d};
  D = 3u*ttl::delta(i,j);
}

TEST(Delta, Widen) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *d;

//initialize memory on device and host
  cudaMallocManaged(&d, 4*sizeof(int));

// intialize verifying tensor
  ttl::Tensor<2, 2, int*> D = {d};

// launch kernel
  test_delta_widen<<<1,1>>>(d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(D(i,j), (i == j) ? 3 : 0);
    }
  }
  
// garbage
  cudaFree(d);
}

__global__
void test_expression(double *a, double *d){
  *d = 3.14;
  ttl::Tensor<2,3,double*> A {a};
  A = *d * ttl::delta(i,j);
}

TEST(Delta, Expression) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  double *a;
  double *d;

//initialize memory on device and host
  cudaMallocManaged(&a, 9*sizeof(double));
  cudaMallocManaged(&d, 9*sizeof(double));

// intialize verifying tensor
  ttl::Tensor<2, 3, double*> A = {a};

// launch kernel
  test_expression<<<1,1>>>(a,d);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m){
      EXPECT_EQ(A(m,n), (m==n) ? *d : 0.0);
    }
  }
}

__global__
void test_expression2(int *a, int *b, int *c, int *d, double *e){
  ttl::Tensor<2,3,int*> A = {a,{1,2,3,4,5,6,7,8}};

  ttl::Tensor<2,3,int*> B = {b},
                        C = {c},
                        D = {d};
                        
  ttl::Tensor<2,3,double*> E = {e};

  B = A(i,j) * ttl::delta(j,k);
  C = A(i,j) * ttl::delta(k,j).to(j,k);
  D = ((ttl::delta(i,k)*ttl::delta(j,l)).to(i,j,k,l))*A(k,l);
  
  E = ttl::delta<3>(i,j)*ttl::delta<3>(j,k);
}

TEST(Delta, Expression2) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;
  int *b;
  int *c;
  int *d;
  double *e;

//initialize memory on device and host
  cudaMallocManaged(&a, 9*sizeof(int));
  cudaMallocManaged(&b, 9*sizeof(int));
  cudaMallocManaged(&c, 9*sizeof(int));
  cudaMallocManaged(&d, 9*sizeof(int));
  cudaMallocManaged(&e, 9*sizeof(double));

// intialize verifying tensor
  ttl::Tensor<2, 3, int*> A = {a};
  ttl::Tensor<2, 3, int*> B = {b};
  ttl::Tensor<2, 3, int*> C = {c};
  ttl::Tensor<2, 3, int*> D = {d};
  ttl::Tensor<2, 3, double*> E = {e};

// launch kernel
  test_expression2<<<1,1>>>(a,b,c,d,e);
    
// control for race conditions
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

// verify results
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(B(m,n), A(m,n));
      EXPECT_EQ(C(m,n), A(m,n));
      EXPECT_EQ(D(m,n), A(m,n));
    }
  }
  
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      if (n == m) EXPECT_EQ(E(n,m), 1);
      else EXPECT_EQ(E(n,m), 0);
    }
  }

// garbage
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(e);
}
