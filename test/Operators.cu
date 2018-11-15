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

static constexpr double E = 2.72;
static constexpr double PI = 3.14;

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;

__global__
void test_negate(int *a, int *b) {
  ttl::Tensor<2, 2, int*> A = {a};
  ttl::Tensor<2, 2, int*> B = {b, {0,1,2,3}};
  A(i,j) = -B(i,j);
}

TEST(UnaryOp, Negate) {
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *a, *b;

  //initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

    // assert(A);
    // assert(B);
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

  // launch kernel
    test_negate<<<1,1>>>(a,b);
    
  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");
  
  // verify results
    EXPECT_EQ(A[0][0], -B[0][0]);
    EXPECT_EQ(A[0][1], -B[0][1]);
    EXPECT_EQ(A[1][0], -B[1][0]);
    EXPECT_EQ(A[1][1], -B[1][1]);
  
  //garbage collect A and B
  cudaFree(a);
  cudaFree(b);
}

__global__
void test_multiply_rhs(double *a, double *b) {
  ttl::Tensor<2, 2, double*> A = {a};
  ttl::Tensor<2, 2, double*> B = {b};
  B.fill(E);
  A(i,j) = PI * B(i,j);
}

TEST(ScalarOp, MultiplyRHS) {
  
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *a, *b;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));
    cudaMallocManaged(&b, 4*sizeof(double));

  // launch storage Tensor
    ttl::Tensor<2, 2, double*> A = {a};

  // launch kernel
    test_multiply_rhs<<<1,1>>>(a,b);

  // control for race conditions  
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  //compare values
    EXPECT_EQ(A[0][0], PI * E);
    EXPECT_EQ(A[0][1], PI * E);
    EXPECT_EQ(A[1][0], PI * E);
    EXPECT_EQ(A[1][1], PI * E);
  
  //garbage collection
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_multiply_lhs(double *a, double *b) {
  ttl::Tensor<2, 2, double*> A = {a};
  ttl::Tensor<2, 2, double*> B = {b};
  B.fill(E);
  A(i,j) = B(i,j) * PI ;
}

TEST(ScalarOp, MultiplyLHS) {

  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *a, *b;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));
    cudaMallocManaged(&b, 4*sizeof(double));

  // launch storage Tensor
    ttl::Tensor<2, 2, double*> A = {a};
  
  // launch kernel
    test_multiply_lhs<<<1,1>>>(a,b);

  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  //compare values
    EXPECT_EQ(A[0][0], PI * E);
    EXPECT_EQ(A[0][1], PI * E);
    EXPECT_EQ(A[1][0], PI * E);
    EXPECT_EQ(A[1][1], PI * E);

  // garbage collection
  cudaFree(a);
  cudaFree(b);
}

__global__
void test_divide(double *a, double *b) {
  ttl::Tensor<2, 2, double*> A = {a};
  ttl::Tensor<2, 2, double*> B = {b};
  B.fill(E);
  A(i,j) = B(i,j) / PI ;
}

TEST(ScalarOp, Divide) {

  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *a, *b;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));
    cudaMallocManaged(&b, 4*sizeof(double));

  // launch storage Tensor
    ttl::Tensor<2, 2, double*> A = {a};

  // launch kernel
    test_divide<<<1,1>>>(a,b);
  
  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  // compare values
    EXPECT_EQ(A[0][0], E / PI);
    EXPECT_EQ(A[0][1], E / PI);
    EXPECT_EQ(A[1][0], E / PI);
    EXPECT_EQ(A[1][1], E / PI);

  // garbage collection
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_modulo(int *a, int *b) {
  ttl::Tensor<2, 2, int*> A = {a};
  ttl::Tensor<2, 2, int*> B = {b,{0,1,2,3}};
  A(i,j) = B(i,j) % 3;
}

TEST(ScalarOp, Modulo) {

  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *a, *b;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

  // launch storage Tensor
    ttl::Tensor<2, 2, int*> A = {a};

  // launch kernel
    test_modulo<<<1,1>>>(a,b);

  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  //compare values
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 0);

  //garbage collection
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_binaryOP_add(int *a,int *b, int *c) {
  const ttl::Tensor<2, 2, int*> A = {a,{0,1,2,3}}, B = {b,{1,2,3,4}};
  ttl::Tensor<2, 2, int*> C = {c};
  C(i,j) = A(i,j) + B(i,j);
}

TEST(BinaryOp, Add) {
  
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *c,*b,*a;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&c, 4*sizeof(int));

  // launch cpu Tensor
  ttl::Tensor<2, 2, int*> C = {c};

  // launch kernel
     test_binaryOP_add<<<1,1>>>(a,b,c);

  // control for race conditions
    cudaDeviceSynchronize();
    
  // check errors
    cudaCheckErrors("failed");

  // compare answers
    EXPECT_EQ(C[0][0], 1);
    EXPECT_EQ(C[0][1], 3);
    EXPECT_EQ(C[1][0], 5);
    EXPECT_EQ(C[1][1], 7);

  // garbage collection
    cudaFree(a);
    cudaFree(b);  
    cudaFree(c);
}

__global__ 
void test_binaryOP_subtract(int *a, int *b, int *c){
  const ttl::Tensor<2, 2, int*> A = {a,{0,1,2,3}}, B = {b,{1,2,3,4}};
  ttl::Tensor<2, 2, int*> C = {c};
  C(i,j) = A(i,j) - B(i,j);
}

TEST(BinaryOp, Subtract) {
  
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *c,*b,*a;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&c, 4*sizeof(int));

  // launch cpu Tensor
    ttl::Tensor<2, 2, int*> C = {c};

  // launch kernel
    test_binaryOP_subtract<<<1,1>>>(a,b,c);

  // control for race conditions
    cudaDeviceSynchronize();
  
  // check errors
    cudaCheckErrors("failed");
  
  // run tests
    EXPECT_EQ(C[0][0], -1);
    EXPECT_EQ(C[0][1], -1);
    EXPECT_EQ(C[1][0], -1);
    EXPECT_EQ(C[1][1], -1);

  // garbage collection
    cudaFree(a);
    cudaFree(b);  
    cudaFree(c);
}

__global__
void test_tensor_product_multiply(int *a, int *b, int *c){
  const ttl::Tensor<2, 2, int*> A = {a,{1,2,3,4}}, B = {b,{2,3,4,5}};
  ttl::Tensor<2, 2, int*> C = {c};
  C(i,j) = A(i,k) * B(k,j);
}

TEST(TensorProduct, Multiply) {
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    int *c,*b,*a;

  //initialize unified memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&c, 4*sizeof(int));

  // launch cpu Tensor
  ttl::Tensor<2, 2, int*> C = {c};

  // launch kernel
  test_tensor_product_multiply<<<1,1>>>(a,b,c);

  // control for race conditions
  cudaDeviceSynchronize();

  // check errors
  cudaCheckErrors("failed");

  EXPECT_EQ(C[0][0], 10);
  EXPECT_EQ(C[0][1], 13);
  EXPECT_EQ(C[1][0], 22);
  EXPECT_EQ(C[1][1], 29);

  // garbage collection
    cudaFree(a);
    cudaFree(b);  
    cudaFree(c);
}

// Throwing an ambiguity in template argument list - 

// error: more than one partial specialization matches the template argument list of class "ttl::Tensor<0, 2, int *>"
//    "ttl::Tensor<R, D, S *>"
//    "ttl::Tensor<0, D, S>"

__global__
void test_tensor_product_inner(int *c){
  const ttl::Tensor<2, 2, int> A = {1,2,3,4}, B = {2,3,4,5};
  ttl::Tensor<0,2,int> D;
  D() = A(i,j) * B(i,j);
  *c = D();
}

TEST(TensorProduct, Inner) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *c;

//initialize unified memory on device and host
  cudaMallocManaged(&c, 1*sizeof(int));

// launch kernel
  test_tensor_product_inner<<<1,1>>>(c);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_EQ(*c, 40);

// garbage collection
  cudaFree(c);
}

__global__
void test_tensor_product_outer(int *a, int *b, int *c){
  const ttl::Tensor<2, 2, int*> A = {a,{1,2,3,4}}, B = {b,{2,3,4,5}};
  ttl::Tensor<4, 2, int*> C = {c};
  C(i,j,k,l) = A(i,j) * B(k,l);
}

TEST(TensorProduct, Outer) {
//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *c,*b,*a;

//initialize unified memory on device and host
  cudaMallocManaged(&a, 4*sizeof(int));
  cudaMallocManaged(&b, 4*sizeof(int));
  cudaMallocManaged(&c, 4*sizeof(int));

// launch cpu Tensor
  ttl::Tensor<4, 2, int*> C = {c};

// launch kernel
  test_tensor_product_outer<<<1,1>>>(a,b,c);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_EQ(C[0][0][0][0], 2);
  EXPECT_EQ(C[0][0][0][1], 3);
  EXPECT_EQ(C[0][0][1][0], 4);
  EXPECT_EQ(C[0][0][1][1], 5);
  EXPECT_EQ(C[0][1][0][0], 4);
  EXPECT_EQ(C[0][1][0][1], 6);
  EXPECT_EQ(C[0][1][1][0], 8);
  EXPECT_EQ(C[0][1][1][1], 10);
  EXPECT_EQ(C[1][0][0][0], 6);
  EXPECT_EQ(C[1][0][0][1], 9);
  EXPECT_EQ(C[1][0][1][0], 12);
  EXPECT_EQ(C[1][0][1][1], 15);
  EXPECT_EQ(C[1][1][0][0], 8);
  EXPECT_EQ(C[1][1][0][1], 12);
  EXPECT_EQ(C[1][1][1][0], 16);
  EXPECT_EQ(C[1][1][1][1], 20);

// garbage collection
  cudaFree(a);
  cudaFree(b);  
  cudaFree(c);
}

__global__
void test_zero_construct(double *z){
  ttl::Tensor<2,3,double> A = ttl::zero(i,j);
  
  ttl::Tensor<2,3,double*> Z {z};
  Z = A;
}

TEST(Zero, Construct) {

  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *z;

  //initialize unified memory on device and host
    cudaMallocManaged(&z, 4*sizeof(double));

  // launch cpu Tensor
    ttl::Tensor<2, 3, double*> Z = {z};

  // launch kernel
    test_zero_construct<<<1,1>>>(z);

  // control for race conditions
    cudaDeviceSynchronize();

  // check errors
    cudaCheckErrors("failed");

  // verify results
    EXPECT_DOUBLE_EQ(Z(0,0), 0.);
    EXPECT_DOUBLE_EQ(Z(0,1), 0.);
    EXPECT_DOUBLE_EQ(Z(1,0), 0.);
    EXPECT_DOUBLE_EQ(Z(1,1), 0.);
  
  // garbage collection
    cudaFree(z);
}

__global__
void test_zero_assign(double *z){
  ttl::Tensor<2,3,double*> Z = {z};
  Z = ttl::zero(i,j);
}

TEST(Zero, Assign) {
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *z;

  //initialize unified memory on device and host
    cudaMallocManaged(&z, 9*sizeof(double));

  // launch cpu Tensor
    ttl::Tensor<2, 3, double*> Z = {z};

  // launch kernel
    test_zero_assign<<<1,1>>>(z);

  // control for race conditions
    cudaDeviceSynchronize();

  // check errors
    cudaCheckErrors("failed");

  // verify results
    EXPECT_DOUBLE_EQ(Z(0,0), 0.);
    EXPECT_DOUBLE_EQ(Z(0,1), 0.);
    EXPECT_DOUBLE_EQ(Z(1,0), 0.);
    EXPECT_DOUBLE_EQ(Z(1,1), 0.);

  // garbage collection
    cudaFree(z);
}

__global__
void test_zero_assign_expression(double *z){
  ttl::Tensor<2,3,double> A;
  A(i,j) = ttl::zero(i,j);

  ttl::Tensor<2,3,double*> Z = {z};
  Z = A;
}

TEST(Zero, AssignExpression) {

//Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  double *z;

//initialize unified memory on device and host
  cudaMallocManaged(&z, 4*sizeof(double));

// launch cpu Tensor
  ttl::Tensor<2, 2, double*> Z = {z};

// launch kernel
  test_zero_assign_expression<<<1,1>>>(z);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_DOUBLE_EQ(Z(0,0), 0.);
  EXPECT_DOUBLE_EQ(Z(0,1), 0.);
  EXPECT_DOUBLE_EQ(Z(1,0), 0.);
  EXPECT_DOUBLE_EQ(Z(1,1), 0.);

// garbage collection
  cudaFree(z);
}

__global__
void test_zero_assign_product(double *z){
  ttl::Tensor<2,3,double*> Z {z};
  ttl::Tensor<2,3,double> A {1,2,3,4,5,6,7,8,9};
  Z(i,k) = ttl::zero(i,j) * A(j,k);
}

TEST(Zero, AssignProduct) {
  //Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

  //instantiate data pointers
    double *z;

  //initialize unified memory on device and host
    cudaMallocManaged(&z, 9*sizeof(double));

  // launch cpu Tensor
    ttl::Tensor<2, 3, double*> Z = {z};

  // launch kernel
    test_zero_assign_product<<<1,1>>>(z);

  // control for race conditions
    cudaDeviceSynchronize();

  // check errors
    cudaCheckErrors("failed");

  // verify results
    EXPECT_DOUBLE_EQ(Z(0,0), 0.);
    EXPECT_DOUBLE_EQ(Z(0,1), 0.);
    EXPECT_DOUBLE_EQ(Z(1,0), 0.);
    EXPECT_DOUBLE_EQ(Z(1,1), 0.);

  // garbage collection
    cudaFree(z);
}
