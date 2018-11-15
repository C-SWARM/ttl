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

const int e[] = {0,1,2,3,4,5,6,7};
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

static const ttl::Tensor<2,2,int> B = {0,1,2,3};
static const ttl::Tensor<2,2,const int> C = {0,1,2,3};
static const ttl::Tensor<2,2,const int*> E(e);

__global__
void test_init_r_value(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};
  ttl::Tensor<2,2,int*> A {a};
  A = 2 * B(i,j);
}

TEST(Bind, InitializeRValue) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_init_r_value<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));

// garbage collection
    cudaFree(a);
}

__global__
void test_init_l_value(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};
  auto e = 2 * B(i,j);
  ttl::Tensor<2,2,int> C = e;
  
  ttl::Tensor<2,2,int*> A {a};
  A = C;
}

TEST(Bind, InitializeLValue) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_init_l_value<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));

// garbage
  cudaFree(a);
}

__global__
void test_assign(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = B(i,j);
}

TEST(Bind, Assign) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(1));
  EXPECT_EQ(B.get(2), A.get(2));
  EXPECT_EQ(B.get(3), A.get(3));

// garbage
  cudaFree(a);
}

__global__
void test_r_val_expression(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = 2*B(i,j);
}

TEST(Bind, AssignRValueExpression) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_r_val_expression<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));

// garbage
  cudaFree(a);
}

__global__
void test_l_val_expression(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};
  auto b = 2 * B(i,j);

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = b;
}

TEST(Bind, AssignLValueExpression) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_l_val_expression<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));
}

__global__
void test_accumulate(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  ttl::Tensor<2,2,int> Z = {};
  Z(i,j) += B(i,j);
  
  ttl::Tensor<2,2,int*> A {a};
  A = Z;
}

TEST(Bind, Accumulate) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_accumulate<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(1));
  EXPECT_EQ(B.get(2), A.get(2));
  EXPECT_EQ(B.get(3), A.get(3));
}

__global__
void test_assign_from_const(int *a) {
  const ttl::Tensor<2,2,const int> C = {0,1,2,3};
  ttl::Tensor<2,2,int> Z;
  Z(i,j) = C(i,j);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignFromConst) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign_from_const<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(C.get(0), A.get(0));
  EXPECT_EQ(C.get(1), A.get(1));
  EXPECT_EQ(C.get(2), A.get(2));
  EXPECT_EQ(C.get(3), A.get(3));

// garbage
  cudaFree(a);
}

__global__
void test_assign_from_external(int *a) {
  const int e[] = {0,1,2,3,4,5,6,7};
  const ttl::Tensor<2,2,const int*> E(e);

  ttl::Tensor<2,2,int> Z;
  Z(i,j) = E(i,j);

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignFromExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign_from_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(e[0], A.get(0));
  EXPECT_EQ(e[1], A.get(1));
  EXPECT_EQ(e[2], A.get(2));
  EXPECT_EQ(e[3], A.get(3));

// garbage
  cudaFree(a);  
}

__global__
void test_assign_to_external(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z(i,j) = B(i,j);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignToExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  test_assign_to_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(a[0], B.get(0));
  EXPECT_EQ(a[1], B.get(1));
  EXPECT_EQ(a[2], B.get(2));
  EXPECT_EQ(a[3], B.get(3));

// garbage
  cudaFree(a);  
}

__global__
void test_assign_external(int *a) {
  const int e[] = {0,1,2,3,4,5,6,7};
  const ttl::Tensor<2,2,const int*> E(e);

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z(i,j) = E(i,j);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  test_assign_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[1]);
  EXPECT_EQ(a[2], e[2]);
  EXPECT_EQ(a[3], e[3]);

// garbage
  cudaFree(a);  
}

__global__
void test_assign_permute(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  ttl::Tensor<2,2,int> Z;
  Z(i,j) = B(j,i);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignPermute) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign_permute<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(2));
  EXPECT_EQ(B.get(2), A.get(1));
  EXPECT_EQ(B.get(3), A.get(3));

// garbage
  cudaFree(a);  
}

__global__
void test_assign_permute_from_const(int *a) {
  const ttl::Tensor<2,2,const int> C = {0,1,2,3};

  ttl::Tensor<2,2,int> Z;
  Z(i,j) = C(j,i);

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignPermuteFromConst) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign_permute_from_const<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(C.get(0), A.get(0));
  EXPECT_EQ(C.get(1), A.get(2));
  EXPECT_EQ(C.get(2), A.get(1));
  EXPECT_EQ(C.get(3), A.get(3));

// garbage
  cudaFree(a);    
}

__global__
void test_assign_permute_from_external(int *a) {
  const int e[] = {0,1,2,3,4,5,6,7};
  const ttl::Tensor<2,2,const int*> E(e);

  ttl::Tensor<2,2,int> Z;
  Z(i,j) = E(j,i);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignPermuteFromExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  test_assign_permute_from_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(e[0], A.get(0));
  EXPECT_EQ(e[1], A.get(2));
  EXPECT_EQ(e[2], A.get(1));
  EXPECT_EQ(e[3], A.get(3));

// garbage
  cudaFree(a);  
}

__global__
void test_assign_permute_to_external(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z(i,j) = B(j,i);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignPermuteToExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  test_assign_permute_to_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(a[0], B.get(0));
  EXPECT_EQ(a[1], B.get(2));
  EXPECT_EQ(a[2], B.get(1));
  EXPECT_EQ(a[3], B.get(3));

// garbage
  cudaFree(a);  
}

__global__
void test_assign_permute_external(int *a) {
  const int e[] = {0,1,2,3,4,5,6,7};
  const ttl::Tensor<2,2,const int*> E(e);

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z(i,j) = E(j,i);
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, AssignPermuteExternal) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  test_assign_permute_external<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[2]);
  EXPECT_EQ(a[2], e[1]);
  EXPECT_EQ(a[3], e[3]);

// garbage
  cudaFree(a);  
}

__global__
void testExternalInitializeRValue(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  ttl::Tensor<2,2,int*> {a, 2 * B(i,j)};
}

TEST(Bind, ExternalInitializeRValue) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testExternalInitializeRValue<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);

// garbage
  cudaFree(a);  
}

__global__
void testExternalInitializeLValue(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};
  auto e = 2 * B(i,j);
  ttl::Tensor<2,2,int*> {a, e};
}

TEST(Bind, ExternalInitializeLValue) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testExternalInitializeLValue<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);

// garbage
  cudaFree(a);  
}

__global__
void testExternalAssignRValueExpression(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z = 2 * B(i,j);

  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, ExternalAssignRValueExpression) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testExternalAssignRValueExpression<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);

//garbage
  cudaFree(a);
}

__global__
void testExternalAssignLValueExpression(int *a) {
  const ttl::Tensor<2,2,int> B = {0,1,2,3};
  auto e = 2 * B(i,j);

  int z[4];
  ttl::Tensor<2,2,int*> Z(z);
  Z = e;
  
  ttl::Tensor<2,2,int*> A {a};
  A(i,j) = Z(i,j);
}

TEST(Bind, ExternalAssignLValueExpression) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testExternalAssignLValueExpression<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);

//garbage
  cudaFree(a);
}

__global__
void testTrace2x2(int *a) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  *a = A(i,i);
}

TEST(Bind, Trace2x2) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 1*sizeof(int)); 

// launch kernel
  testTrace2x2<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, 5);

//garbage
  cudaFree(a);
}

__global__
void testTrace2x3(int *a) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8,9};
  *a = A(i,i);
}

TEST(Bind, Trace2x3) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 1*sizeof(int)); 

// launch kernel
  testTrace2x3<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, 15);

//garbage
  cudaFree(a);  
}

__global__
void testTrace3x2(int *a, int *z) {
  ttl::Tensor<3,2,int*> Z = {z,{1,2,3,4,5,6,7,8}};
  *a = Z(i,i,i);
}

TEST(Bind, Trace3x2) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a; 
  int *z; 
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 1*sizeof(int)); 
  cudaMallocManaged(&z, 8*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<3, 2, int*> Z = {z};

// launch kernel
  testTrace3x2<<<1,1>>>(a,z);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, Z.get(0) + Z.get(7));

//garbage
  cudaFree(a); 
  cudaFree(z);
}

__global__
void testParallelContract(int *a, int *z) {
  ttl::Tensor<4,2,int*> Z = {z,{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}};
  *a = Z(i,i,j,j);
}

TEST(Bind, ParallelContract) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
  int *z;
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 1*sizeof(int)); 
  cudaMallocManaged(&z, 16*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<4, 2, int*> Z = {z};

// launch kernel
  testParallelContract<<<1,1>>>(a,z);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, Z.get(0) + Z.get(3) + Z.get(12) + Z.get(15));

//garbage
  cudaFree(a); 
  cudaFree(z); 
}

__global__
void testSequentialContract(int *a, int *z) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  ttl::Tensor<2,2,int*> Z = {z,A(i,i,j,k)};
  *a = Z(j,j);
}

TEST(Bind, SequentialContract) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
  int *z;
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 1*sizeof(int)); 
  cudaMallocManaged(&z, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> Z = {z};

// launch kernel
  testSequentialContract<<<1,1>>>(a,z);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, Z.get(0) + Z.get(3));
 
//garbage
  cudaFree(a); 
  cudaFree(z);
}

__global__
void testProjectionRead1(int *a) {
  ttl::Tensor<1,2,int*> A {a};
  A = {0,1};
}

TEST(Bind, ProjectionRead1) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 2*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<1, 2, int*> A = {a};

// launch kernel
  testProjectionRead1<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A(0), A[0]);
  EXPECT_EQ(A(1), A[1]);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionRead2(int *a) {
  ttl::Tensor<2,2,int*> A {a,{0,1,2,3}};
}

TEST(Bind, ProjectionRead2) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  testProjectionRead2<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A(0,0), A[0][0]);
  EXPECT_EQ(A(0,1), A[0][1]);
  EXPECT_EQ(A(1,0), A[1][0]);
  EXPECT_EQ(A(1,1), A[1][1]);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionRead2_1(int *a) {
  ttl::Tensor<2,2,int> V = {0,1,2,3};
  ttl::Tensor<1,2,int*> A {a};
  A = V(1,i);
}

TEST(Bind, ProjectionRead2_1) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 2*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<1, 2, int*> A = {a};

// launch kernel
  testProjectionRead2_1<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A(0), 2);
  EXPECT_EQ(A(1), 3);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionRead3(int *a, int *scalarPtr, int *z, int *y, int *x) {
  ttl::Tensor<3,2,int*> A = {a,{0,1,2,3,4,5,6,7}};
  *scalarPtr = A(0,1,0);

  ttl::Tensor<2,2,int*> Z {z};
  Z = A(i,1,j);

  ttl::Tensor<1,2,int*> Y {y};
  Y = A(1,i,0);

  ttl::Tensor<1,2,int*> X {x};
  X = A(i,1,1);
}

TEST(Bind, ProjectionRead3) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;
  int *scalarPtr;  
  int *z;
  int *y;
  int *x;
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 8*sizeof(int)); 
  cudaMallocManaged(&scalarPtr, 1*sizeof(int)); 
  cudaMallocManaged(&z, 4*sizeof(int)); 
  cudaMallocManaged(&y, 2*sizeof(int)); 
  cudaMallocManaged(&x, 2*sizeof(int)); 

// launch storage Tensors
  ttl::Tensor<3, 2, int*> A {a};
  ttl::Tensor<2, 2, int*> Z {z};
  ttl::Tensor<1, 2, int*> Y {y};
  ttl::Tensor<1, 2, int*> X {x};

// launch kernel
  testProjectionRead3<<<1,1>>>(a,scalarPtr,z,y,x);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*scalarPtr, A[0][1][0]);

  EXPECT_EQ(Z(0,0), A(0,1,0));
  EXPECT_EQ(Z(0,1), A(0,1,1));
  EXPECT_EQ(Z(1,0), A(1,1,0));
  EXPECT_EQ(Z(1,1), A(1,1,1));
  
  EXPECT_EQ(Y(0), A(1,0,0));
  EXPECT_EQ(Y(1), A(1,1,0));

  EXPECT_EQ(X(0), A(0,1,1));
  EXPECT_EQ(X(1), A(1,1,1));

//garbage
  cudaFree(a); 
  cudaFree(scalarPtr); 
  cudaFree(z); 
  cudaFree(y); 
  cudaFree(x); 
}

__global__
void testProjectionWrite(int *a) {
  ttl::Tensor<1,2,int> Z;
  Z(0) = 0;
  Z(1) = 1;

  ttl::Tensor<1,2,int*> A{a};
  A = Z;
}

TEST(Bind, ProjectionWrite) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 2*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<1, 2, int*> A = {a};

// launch kernel
  testProjectionWrite<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionWrite2(int *a) {
  ttl::Tensor<2,2,int> Z;
  Z(0,0) = 0;
  Z(0,1) = 1;
  Z(1,0) = 2;
  Z(1,1) = 3;
  
  ttl::Tensor<2,2,int*> A {a};
  A = Z;
}

TEST(Bind, ProjectionWrite2) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  testProjectionWrite2<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionWriteVector(int *a1,int *a2) {
  ttl::Tensor<2,2,int> Z = {};
  ttl::Tensor<1,2,int> v = {1,2};

  ttl::Tensor<2,2,int*> A1 {a1};
  ttl::Tensor<2,2,int*> A2 {a2};
  
  Z(i,0) = v(i);
  A1 = Z;

  Z(1,i) = v(i);
  A2 = Z;
}

TEST(Bind, ProjectionWriteVector) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a1;  
  int *a2;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a1, 4*sizeof(int)); 
  cudaMallocManaged(&a2, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A1 = {a1};
  ttl::Tensor<2, 2, int*> A2 = {a2};

// launch kernel
  testProjectionWriteVector<<<1,1>>>(a1,a2);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A1[0][0], 1);
  EXPECT_EQ(A1[1][0], 2);

  
  EXPECT_EQ(A2[1][0], 1);
  EXPECT_EQ(A2[1][1], 2);

//garbage
  cudaFree(a1); 
  cudaFree(a2); 
}

__global__
void testProjectionWriteMatrix(int *a) {
  ttl::Tensor<3,2,int> Z = {};
  ttl::Tensor<2,2,int> M = {1,2,3,4};

  Z(i,0,j) = M(i,j);
  
  ttl::Tensor<3,2,int*> A {a};
  A = Z;
}

TEST(Bind, ProjectionWriteMatrix) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 8*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<3, 2, int*> A = {a};

// launch kernel
  testProjectionWriteMatrix<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A[0][0][0], 1);
  EXPECT_EQ(A[0][0][1], 2);
  EXPECT_EQ(A[1][0][0], 3);
  EXPECT_EQ(A[1][0][1], 4);

//garbage
  cudaFree(a); 
}

__global__
void testProjectionProduct(int *a) {
  ttl::Tensor<2,2,int> Z = {}, Y = {1,2,3,4};
  ttl::Tensor<3,2,int> X = {1,2,3,4,5,6,7,8};

  Z(i,0) = Y(j,i)*X(1,j,0);

  ttl::Tensor<2,2,int*> A {a};
  A = Z;
}

TEST(Bind, ProjectionProduct) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
  testProjectionProduct<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(A[0][0], 26);
  EXPECT_EQ(A[1][0], 38);

//garbage
  cudaFree(a); 
}

__global__
void testCurry(int *a) {
  ttl::Tensor<1,2,int> A = {1,2};
  auto f = A(j);
  
  *a = f(0);
  *(a+1) = f(1);
}

TEST(Bind, Curry) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testCurry<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*(a), 1);
  EXPECT_EQ(*(a+1), 2);

//garbage
  cudaFree(a); 
}

__global__
void testPermuteSubTree(int *a) {
  ttl::Tensor<2,2,int> A = {1,2,3,4},
                       B = {1,3,2,4};
  
  auto f = A(i,j).to(j,i);

  *a = f(0,0);
  *(a+1) = f(0,1);
  *(a+2) = f(1,0);
  *(a+3) = f(1,1);


}

TEST(Bind, PermuteSubtree) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *a;  

// comparison tensor
  ttl::Tensor<2,2,int> B = {1,3,2,4};
    
//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int)); 

// launch kernel
  testPermuteSubTree<<<1,1>>>(a);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values  
  EXPECT_EQ(*a, B(0,0));
  EXPECT_EQ(*(a+1), B(0,1));
  EXPECT_EQ(*(a+2), B(1,0));
  EXPECT_EQ(*(a+3), B(1,1));

//garbage
  cudaFree(a); 
}

