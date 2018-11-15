// -------------------------------------------------------------------*- C++ -*-
// Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
// University of Notre Dame
// Indiana University
// University of Washington
//	Alexander
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


__global__
void test_ctorDefault(double *a){
    ttl::Tensor<2, 2, double> A;
    if(sizeof(A) == 4*sizeof(double)){
        *a = 0.;
    }
}

TEST(Tensor, CtorDefault) {
    // Check for available device
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        assert(nDevices > 0);
    
    // instantiate data pointers
        double b = 1., *a = &b;
    
    // initialize memory on device and host
        cudaMallocManaged(&a, 4*sizeof(double));
    
    // launch kernel
        test_ctorDefault<<<1,1>>>(a);
    
    // control for race conditions
        cudaDeviceSynchronize();
    
    // check errors
        cudaCheckErrors("failed");
    
    // verify results
        EXPECT_EQ(*a, 0.);
    
    // garbage collection
        cudaFree(a);
    }

__global__
void test_linear_indexing(double *a) {
    ttl::Tensor<2, 2, double*> A = {a};
    A.get(0) = 0.0;
    A.get(1) = 1.0;
    A.get(2) = E;
    A.get(3) = PI;
}

TEST(Tensor, LinearIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_linear_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A.get(0), 0.0);
    EXPECT_EQ(A.get(1), 1.0);
    EXPECT_EQ(A.get(2), E);
    EXPECT_EQ(A.get(3), PI);

// garbage collection
    cudaFree(a);
}

__global__
void test_array_indexing(double *a) {
    ttl::Tensor<1, 2, double*> A = {a,{0.0, 1.0}};
}

TEST(Tensor, ArrayIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 2*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<1, 2, double*> A = {a};

// launch kernel
    test_array_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0], 0.0);
    EXPECT_EQ(A[1], 1.0);

// garbage collection
    cudaFree(a);
}

__global__
void test_extern_array_indexing(double *a) {
    double b[2] = {0.0, 1.0};
    ttl::Tensor<1, 2, double*> B{b};
    ttl::Tensor<1, 2, double*> A{a};
    A = B;
}

TEST(Tensor, ExternArrayIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 2*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<1, 2, double*> A = {a};

// launch kernel
    test_extern_array_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0], 0.0);
    EXPECT_EQ(A[1], 1.0);

// garbage collection
    cudaFree(a);
}


// $ Why do the values for the host side tensor update?
__global__
void test_const_array_indexing(double *a) {
    const ttl::Tensor<1, 2, double*> A = {a,{0.0, 1.0}};
}

TEST(Tensor, ConstArrayIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 2*sizeof(double));

// instantiate cpu side tensor
    const ttl::Tensor<1, 2, double*> A = {a};

// launch kernel
    test_const_array_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0], 0.0);
    EXPECT_EQ(A[1], 1.0);
}

__global__
void test_array_index_assignment(double *a) {
    ttl::Tensor<1, 2, double*> A = {a};
    A[0] = 0.0;
    A[1] = 1.0;  
}

TEST(Tensor, ArrayIndexAssignment) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 2*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_array_index_assignment<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A.get(0), 0.0);
    EXPECT_EQ(A.get(1), 1.0);

// garbage collection
    cudaFree(a);
}

__global__
void test_2D_array_indexing(double *a) {
    ttl::Tensor<2, 2, double*> A = {a,{ 0.0, 1.0, E, PI }};
}

TEST(Tensor, 2DArrayIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_2D_array_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0.0);
    EXPECT_EQ(A[0][1], 1.0);
    EXPECT_EQ(A[1][0], E);
    EXPECT_EQ(A[1][1], PI);

// garbage collection
    cudaFree(a);
}

__global__
void test_2D_array_index_assignment(double *a) {
    ttl::Tensor<2, 2, double*> A = {a};
    A[0][0] = PI;
    A[0][1] = E;
    A[1][0] = 1.0;
    A[1][1] = 0.0;
}

TEST(Tensor, 2DArrayIndexAssignment) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_2D_array_index_assignment<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A.get(0), PI);
    EXPECT_EQ(A.get(1), E);
    EXPECT_EQ(A.get(2), 1.0);
    EXPECT_EQ(A.get(3), 0.0);

// garbage collection
    cudaFree(a);
}

__global__
void Ctor(int *a) {
    ttl::Tensor<2, 2, int> B = {0, 1, 2, 3};
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}

TEST(Tensor, Ctor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    Ctor<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage collection
    cudaFree(a);
}

__global__
void CtorZero(int *a) {
    ttl::Tensor<2, 2, int*> A = {a,{}};
}

TEST(Tensor, CtorZero) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    CtorZero<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 0);
    EXPECT_EQ(A[1][0], 0);
    EXPECT_EQ(A[1][1], 0);

// garbage collection
    cudaFree(a);
}

__global__
void CtorZeroSuffix(int *a) {
    ttl::Tensor<2, 2, int*> A = {a,{0,1,2}};
}

TEST(Tensor, CtorZeroSuffix) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    CtorZeroSuffix<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 0);

// garbage collection
    cudaFree(a);
}

__global__
void ctor_ignore_overflow(int *a, int *b) {
    ttl::Tensor<2, 2, int*> B = {b,{}};
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3, 4, 5, 6, 7}};   
}

TEST(Tensor, CtorIgnoreOverflow) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    ctor_ignore_overflow<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);
    EXPECT_EQ(B[0][0], 0);
    EXPECT_EQ(B[0][1], 0);
    EXPECT_EQ(B[1][0], 0);
    EXPECT_EQ(B[1][1], 0);

// garbage collection
    cudaFree(a);
    cudaFree(b);
}


// unhappy with solution
__global__
void CtorWiden(double *a) {
    ttl::Tensor<1, 3, double> B = {int(1), float(E), PI};
    ttl::Tensor<1, 3, double*> A {a};
    A = B;
}

TEST(Tensor, CtorWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 3*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<1, 3, double*> A = {a};

// launch kernel
    CtorWiden<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0], 1.0);
    EXPECT_EQ(A[1], float(E));
    EXPECT_EQ(A[2], PI);

// garbage collection
    cudaFree(a);
}

__global__
void const_ctor(int *a) {
    const ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
}

TEST(Tensor, ConstCtor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    const_ctor<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage collection
    cudaFree(a);
}

//  This kind of exchange of values is very poor - needs work - need to write another constructor to
// allow construction with external buffer.

__global__
void test_zero_const(int *a) {
    ttl::Tensor<2, 2, const int> B = {};
    ttl::Tensor<2, 2, int*> A = {a};
    A = B;
}

TEST(Tensor, ZeroConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_zero_const<<<1,1>>>(a);

// instantiate cpu side tensor
    ttl::Tensor<2, 2, const int*> A = {a};

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 0);
    EXPECT_EQ(A[1][0], 0);
    EXPECT_EQ(A[1][1], 0);

// garbage collection
    cudaFree(a);
}

__global__
void test_const_ctor_zero(int *a) {
    const ttl::Tensor<2, 2, int*> A = {a,{}};
}

TEST(Tensor, ConstCtorZero) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    const ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_const_ctor_zero<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 0);
    EXPECT_EQ(A[1][0], 0);
    EXPECT_EQ(A[1][1], 0);

// garbage collection
    cudaFree(a);
}

// $ Found a solution, don't like it. Requires dummy reporter ojbect which doesn't capture all behavior.
// error: argument of type "const int *" is incompatible with parameter of type "void *" Attached to cudaFree

__global__
void test_ctor_const(int *a) {
    ttl::Tensor<2, 2, const int> B = {0, 1, 2, 3};
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}

TEST(Tensor, CtorConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_ctor_const<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results   
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage collection
    cudaFree(a);
}

// $ Found a solution, don't like it. Requires dummy reporter object which does not capture all behavior

// error: argument of type "const int *" is incompatible with parameter of type "void *" Attached to cudaFree

__global__
void test_const_ctor_const(int *a) {
    const ttl::Tensor<2, 2, const int> B = {0, 1, 2, 3};
    ttl::Tensor<2, 2, int*> A = {a};
    A = B;
}

TEST(Tensor, ConstCtorConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_const_ctor_const<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify output
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage collection
    cudaFree(a);
}

// $ Found a solution, don't like it. Requires dummy reporter object which does not capture all behavior
// // error: argument of type "const int *" is incompatible with parameter of type "void *" Attached to cudaFree

__global__
void test_const_ctor_zero_const(int *a) {
    const ttl::Tensor<2, 2, const int> B = {};
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}

TEST(Tensor, ConstCtorZeroConst) { 
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_const_ctor_zero_const<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify output
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 0);
    EXPECT_EQ(A[1][0], 0);
    EXPECT_EQ(A[1][1], 0);

// garbage collection
    cudaFree(a);
}

// $
// don't know how to do the direct assignment - did an indirect mapping

__global__
void test_copy_ctor(int *a, int *b ) {
    ttl::Tensor<2, 2, int*> A = {a, {0, 1, 2, 3}};
    ttl::Tensor<2, 2, int*> B {b};
    ttl::Tensor<2, 2, int> C = A;
    B = C;
}

TEST(Tensor, CopyCtor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};


// launch kernel
    test_copy_ctor<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify result
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage collection
    cudaFree(a);   
    cudaFree(b); 
}

__global__
void test_copy_ctor_widen(int *a, float *b ) {
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, float*> B {b};
    B = A;
}

TEST(Tensor, CopyCtorWiden) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;
    float *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(float));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, float*> B = {b};

// launch kernel
    test_copy_ctor_widen<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify value
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage collection
    cudaFree(a);
    cudaFree(b);
}


// $ dislike - proves the operations work, but seems very inelegant
__global__
void test_copy_ctor_from_const(int *a, int *b ) {
    const ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
    // B = A;
}

TEST(Tensor, CopyCtorFromConst) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;
    int *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    test_copy_ctor_from_const<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage collection
    cudaFree(a);
    cudaFree(b);
}


// error: no operator "=" matches these operands
// operand types are: const ttl::Tensor<2, 2, int *> = ttl::Tensor<2, 2, int *>

__global__
void test_copy_ctor_to_const(int *a, int *b){
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    const ttl::Tensor<2, 2, int> C = A;
    ttl::Tensor<2, 2, int*> B = {b};
    B = C;
}

TEST(Tensor, CopyCtorToConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_copy_ctor_to_const<<<1,1>>>(a,b);

// update with const tensor on cpu
    const ttl::Tensor<2, 2, int*> B = {b};

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_ctor_from_const_to_const(int *a, int *b) {
    const ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    const ttl::Tensor<2, 2, int> C = A;
    ttl::Tensor<2, 2, int*> B {b};
    B = C;
}

TEST(Tensor, CopyCtorFromConstToConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// launch kernel
    test_copy_ctor_from_const_to_const<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// instantiate cpu side tensor
    const ttl::Tensor<2, 2, int*> A = {a};
    const ttl::Tensor<2, 2, int*> B = {b};

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_cstor_from_const_data(int *a, int *b){
    ttl::Tensor<2, 2, int*> A {a};
    ttl::Tensor<2, 2, int*> B {b};
    ttl::Tensor<2, 2, const int> C = {0, 1, 2, 3};
    A = C;
    B = A;

//   ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
//   ttl::Tensor<2, 2, int> B = A;
}
    
TEST(Tensor, CopyCtorFromConstData) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, const int*> A = {a};
    ttl::Tensor<2, 2, const int*> B = {b};

// launch kernel
    test_copy_cstor_from_const_data<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_cstor_to_const_data(int *a, int *b){
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, const int> C = A;
    ttl::Tensor<2, 2, int*> B {b};
    B = C;
    
//   ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
//   ttl::Tensor<2, 2, const int> B = A;
}

TEST(Tensor, CopyCtorToConstData) {


// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
test_copy_cstor_to_const_data<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

//   garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_move_ctor(int *a){
    ttl::Tensor<2, 2, int> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
//   ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
}

TEST(Tensor, MoveCtor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_move_ctor<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage
    cudaFree(a);
}

__global__
void test_move_ctor_widen(double *a){
    ttl::Tensor<2, 2, double> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
    ttl::Tensor<2, 2, double*> A {a};
    A = B;
}

TEST(Tensor, MoveCtorWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_move_ctor_widen<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0.0);
    EXPECT_EQ(A[0][1], 1.0);
    EXPECT_EQ(A[1][0], 2.0);
    EXPECT_EQ(A[1][1], 3.0);

// garbage collection
    cudaFree(a);
}

__global__
void test_move_ctor_to_const(int *a){
    const ttl::Tensor<2, 2, int> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}


TEST(Tensor, MoveCtorToConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_move_ctor_to_const<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage
    cudaFree(a);
}

__global__
void test_move_ctor_from_const_data(int *a){
    ttl::Tensor<2, 2, int*> A {a};
    A = std::move(ttl::Tensor<2, 2, const int>{0, 1, 2, 3}); 
}

TEST(Tensor, MoveCtorFromConstData) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_move_ctor_from_const_data<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage
    cudaFree(a);
}

__global__
void test_move_ctor_to_const_data(int *a){
    ttl::Tensor<2, 2, const int> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}

TEST(Tensor, MoveCtorToConstData) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_move_ctor_to_const_data<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage
    cudaFree(a);
}

__global__
void test_assign(int *a){
    ttl::Tensor<2, 2, int*> A {a};
    A = {0, 1, 2, 3};
}

TEST(Tensor, Assign) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_assign<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage
    cudaFree(a);
}

__global__
void test_assign_zero(int *a){
    ttl::Tensor<2, 2, int*> A {a};
    A = {};
}

TEST(Tensor, AssignZero) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_assign_zero<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);

  // garbage
    cudaFree(a);
}

__global__
void test_assign_zero_suffix(int *a){
    ttl::Tensor<2, 2, int*> A {a};
    A = {0,1,2};
}

TEST(Tensor, AssignZeroSuffix) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_assign_zero_suffix<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// compare values
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 0);

  // garbage collection
    cudaFree(a);
}

__global__
void test_assign_zignore_overflow(int *a, int *b){
    ttl::Tensor<2, 2, int*> B = {b,{}};
    ttl::Tensor<2, 2, int*> A {a};
    A = {0, 1, 2, 3, 4, 5, 6, 7};
}

TEST(Tensor, AssignIgnoreOverflow) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel    
    test_assign_zignore_overflow<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);
    EXPECT_EQ(B[0][0], 0);
    EXPECT_EQ(B[0][1], 0);
    EXPECT_EQ(B[1][0], 0);
    EXPECT_EQ(B[1][1], 0);
// gabrage
    cudaFree(a);
    cudaFree(b);
}

// I don't like this solution

__global__
void test_assign_widen(double *a){
    ttl::Tensor<1, 3, double*> A {a};
    A = {int(1), float(E), PI};
}

TEST(Tensor, AssignWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<1, 3, double*> A = {a};

// launch kernel
    test_assign_widen<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0], 1.0);
    EXPECT_EQ(A[1], float(E));
    EXPECT_EQ(A[2], PI);

// garbage collect
    cudaFree(a);
}

__global__
void test_copy(int *a, int *b){
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
}

TEST(Tensor, Copy) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    test_copy<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// cleanup
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_widen(int *a, float *b){
    ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, float*> B {b};
    B = A;
}

TEST(Tensor, CopyWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;
    float *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, float*> B = {b};

// launch kernel
    test_copy_widen<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage collection
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_from_const(int *a, int *b){
    const ttl::Tensor<2, 2, int*> A = {a,{0, 1, 2, 3}};
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
}

TEST(Tensor, CopyFromConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    test_copy_from_const<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_copy_from_const_data(int *b){
    ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
}

TEST(Tensor, CopyFromConstData) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *b;

// initialize memory on device and host
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> B = {b};
    ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};

// launch kernel
    test_copy_from_const_data<<<1,1>>>(b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(B[0][0], A[0][0]);
    EXPECT_EQ(B[0][1], A[0][1]);
    EXPECT_EQ(B[1][0], A[1][0]);
    EXPECT_EQ(B[1][1], A[1][1]);

// garbage  
    cudaFree(b);
}

__global__
void test_move(int *a){
    ttl::Tensor<2, 2, int> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
    ttl::Tensor<2, 2, int*> A {a}; 
    A = B;
}

TEST(Tensor, Move) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A {a};

// launch kernel
    test_move<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage collect
    cudaFree(a);
}

__global__
void test_move_widen(double *a){
    ttl::Tensor<2, 2, double> B = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});

    ttl::Tensor<2, 2, double*> A {a}; 
    A = B;
}

TEST(Tensor, MoveWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_move_widen<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0.0);
    EXPECT_EQ(A[0][1], 1.0);
    EXPECT_EQ(A[1][0], 2.0);
    EXPECT_EQ(A[1][1], 3.0);

// clear
    cudaFree(a);
}

__global__
void test_move_from_const_data(int *a){
    ttl::Tensor<2, 2, int*> A {a};
    A = std::move(ttl::Tensor<2, 2, const int>{0, 1, 2, 3});
}

TEST(Tensor, MoveFromConstData) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_move_from_const_data<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 0);
    EXPECT_EQ(A[0][1], 1);
    EXPECT_EQ(A[1][0], 2);
    EXPECT_EQ(A[1][1], 3);

// garbage  
    cudaFree(a);
}

__global__
void test_fill(double *a){
    ttl::Tensor<2, 2, double*> A {a};
    A.fill(E);
}

TEST(Tensor, Fill) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_fill<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], E);
    EXPECT_EQ(A[0][1], E);
    EXPECT_EQ(A[1][0], E);
    EXPECT_EQ(A[1][1], E);

// cleanup
    cudaFree(a);
}

__global__
void test_fill_widen(double *a){
    ttl::Tensor<2, 2, double*> A {a};
    A.fill(2);
}

TEST(Tensor, FillWiden) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, double*> A = {a};

// launch kernel
    test_fill_widen<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 2.0);
    EXPECT_EQ(A[0][1], 2.0);
    EXPECT_EQ(A[1][0], 2.0);
    EXPECT_EQ(A[1][1], 2.0);

// cleanup
    cudaFree(a);
}

__global__
void test_external_ctor(int *z){
// internal operations
    ttl::Tensor<2, 2, int*> A2(z);
    ttl::Tensor<2, 2, const int*> B2(z);
    const ttl::Tensor<2, 2, int*> C2(z);
    const ttl::Tensor<2, 2, const int*> D2(z);
}

TEST(ExternalTensor, Ctor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int a[4] = {0,1,2,3};
    int *z = a;

// initialize memory on device and host
    cudaMallocManaged(&z, 4*sizeof(int));

// launch kernel
    test_external_ctor<<<1,1>>>(z);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

//   ttl::Tensor<2, 2, const int*> B(a);
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

//   const ttl::Tensor<2, 2, int*> C(a);
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

//   const ttl::Tensor<2, 2, const int*> D(a);
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

//  garbage
    cudaFree(z);
}

__global__
void test_ctor_list(int* a){
    ttl::Tensor<2, 2, int*> {a, {0, 1, 2, 3}};
}

TEST(Tensor, CtorList) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data and pointers
    int *a = new int[4];

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_ctor_list<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

// cleanup
    cudaFree(a);
}

// invalid device pointer at the cuda error check line

__global__
void test_ctor_pointer(int* a){
    int b[8] = {0,1,2,3,4,5,6,7};
    ttl::Tensor<2, 2, int*> B(&b[2]);
    ttl::Tensor<2, 2, int*> A {a};
    A = B;
}

TEST(ExternalTensor, CtorPointer) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 8*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};

// launch kernel
    test_ctor_pointer<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A[0][0], 2);
    EXPECT_EQ(A[0][1], 3);
    EXPECT_EQ(A[1][0], 4);
    EXPECT_EQ(A[1][1], 5);

// garbage
    cudaFree(a);
}

// no test in original .cpp file

__global__
void test_external_tensor_ctor_const(int* a, int *b){
    const int z[4] = {0,1,2,3};
    ttl::Tensor<2,2,const int*> C(z);
    const ttl::Tensor<2,2,const int*> D(z);

    ttl::Tensor<2,2,int*> A {a};
    ttl::Tensor<2,2,int*> B {b};
    A = C;
    B = D;
}

TEST(ExternalTensor, CtorConst) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;
    int *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> A = {a};
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    test_external_tensor_ctor_const<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(A(0,0),B(0,0));
    EXPECT_EQ(A(1,0),B(1,0));
    EXPECT_EQ(A(0,1),B(0,1));
    EXPECT_EQ(A(1,1),B(1,1));

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_externalTensor_array_indexing(double *a){
    ttl::Tensor<2, 2, double*> A {a};
    A[0][0] = 0.0;
    A[0][1] = 1.0;
    A[1][0] = E;
    A[1][1] = PI;  
}

TEST(ExternalTensor, ArrayIndexing) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a = new double[4];

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// launch kernel
    test_externalTensor_array_indexing<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a[0], 0.0);
    EXPECT_EQ(a[1], 1.0);
    EXPECT_EQ(a[2], E);
    EXPECT_EQ(a[3], PI);

// garbage
    cudaFree(a);
}

__global__
void test_externalTensor_copy_ctor(int *b){
    int a[4] = {0,1,2,3};
    ttl::Tensor<2, 2, int*> A {a};
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
}

TEST(ExternalTensor, CopyCtor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int a[4] = {0,1,2,3};

    int *b;

// initialize memory on device and host
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, int*> B = {b};

// launch kernel
    test_externalTensor_copy_ctor<<<1,1>>>(b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(B[0][0], a[0]);
    EXPECT_EQ(B[0][1], a[1]);
    EXPECT_EQ(B[1][0], a[2]);
    EXPECT_EQ(B[1][1], a[3]);

// garbage
    cudaFree(b);
}

__global__
void test_external_tensor_copy_ctor_const(const int *a, int *b){
    const int z[4] = {0,1,2,3};

// non const tensor
    ttl::Tensor<2, 2, const int*> M(z);
    ttl::Tensor<2, 2, const int*> N = M;
    
    ttl::Tensor<2, 2, int*> A {b};
    A = N;

// const type tensor
    const ttl::Tensor<2, 2, const int*> J(z);
    const ttl::Tensor<2, 2, const int*> K = J;

    ttl::Tensor<2, 2, int*> B {b};
    B = K;
}

TEST(ExternalTensor, CopyCtorConst) {
    // Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a, *b;

    const int z[4] = {0,1,2,3};

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// launch kernel
    test_external_tensor_copy_ctor_const<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// instantiate cpu side tensor
    ttl::Tensor<2, 2, const int*> A = {a};
    ttl::Tensor<2, 2, const int*> B = {b};

// verify results on non-const tensor
    EXPECT_EQ(A[0][0], z[0]);
    EXPECT_EQ(A[0][1], z[1]);
    EXPECT_EQ(A[1][0], z[2]);
    EXPECT_EQ(A[1][1], z[3]);

//   verify results on const tensor
    EXPECT_EQ(B[0][0], z[0]);
    EXPECT_EQ(B[0][1], z[1]);
    EXPECT_EQ(B[1][0], z[2]);
    EXPECT_EQ(B[1][1], z[3]);

// garbage
    cudaFree(a);
    cudaFree(b);
}

__global__
void test_ex_tensor_move_ctor(int *a){
    ttl::Tensor<2, 2, int*> A = std::move(ttl::Tensor<2, 2, int*>(a));
    A[0][0] = 0;
    A[0][1] = 1;
    A[1][0] = 2;
    A[1][1] = 3;
}

TEST(ExternalTensor, MoveCtor) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a_h = new int[4];

// initialize memory on device and host
    cudaMallocManaged(&a_h, 4*sizeof(int));

// launch kernel
    test_ex_tensor_move_ctor<<<1,1>>>(a_h);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a_h[0], 0);
    EXPECT_EQ(a_h[1], 1);
    EXPECT_EQ(a_h[2], 2);
    EXPECT_EQ(a_h[3], 3);

// garbage
    cudaFree(a_h);
}

__global__
void test_copy_ctor_external(int *b, const int *z){
    const ttl::Tensor<2, 2, const int*> A(z);
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
    B[1][1] = 0;
}

TEST(Tensor, CopyCtorExternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *b;
    
    const int a[4] = {0,1,2,3};
    const int *z = a;

// initialize memory on device and host
    cudaMallocManaged(&b, 4*sizeof(int));
    cudaMallocManaged(&z, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, const int*> B = {b};

// launch kernel
    test_copy_ctor_external<<<1,1>>>(b,z);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(B[0][0], 0);
    EXPECT_EQ(B[0][1], 1);
    EXPECT_EQ(B[1][0], 2);
    EXPECT_EQ(B[1][1], 0);
    EXPECT_EQ(a[3], 3);

// garbage
    cudaFree(b);
}

__global__
void test_move_ctor_extern(int * a){
    ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, int*>(a));
    A[0][0] = 0;
    A[0][1] = 1;
    A[1][0] = 2;
    A[1][1] = 3;
}

TEST(Tensor, MoveCtorExternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int a[4] = {0,1,2,3};
    int *z = a;

// initialize memory on device and host
    cudaMallocManaged(&z, 4*sizeof(int));

// launch kernel
    test_move_ctor_extern<<<1,1>>>(z);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 3);

// garbage
    cudaFree(z);
}

__global__
void test_assign_external(const int *a, int *b){
    ttl::Tensor<2, 2, const int*> A(a);
    ttl::Tensor<2, 2, int*> B {b};
    B = A;
    B[1][1] = 0;
}

TEST(Tensor, AssignExternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    const int z[4] = {0,1,2,3};
    const int *a = z;
    int *b;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));
    cudaMallocManaged(&b, 4*sizeof(int));

// instantiate cpu side tensor
    ttl::Tensor<2, 2, const int*> B = {b};

// launch kernel
    test_assign_external<<<1,1>>>(a,b);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(B[0][0], 0);
    EXPECT_EQ(B[0][1], 1);
    EXPECT_EQ(B[1][0], 2);
    EXPECT_EQ(B[1][1], 0);
    EXPECT_EQ(z[3], 3);
}

// no idea why the array won't update
__global__
void test_move_external(int *a){
    int b[4] = {0,1,2,3};
    ttl::Tensor<2, 2, int> A;
    A = std::move(ttl::Tensor<2, 2, int*>(b));
    A[0][0] = 0;
    A[0][1] = 1;
    A[1][0] = 2;
    A[1][1] = 3;

    *a = A[0][0];
    *(a+1) = A[0][1];
    *(a+2) = A[1][0];
    *(a+3) = A[1][1];
}

TEST(Tensor, MoveExternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_move_external<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*a, 0);
    EXPECT_EQ(*(a+1), 1);
    EXPECT_EQ(*(a+2), 2);
    EXPECT_EQ(*(a+3), 3);

// garbage
    cudaFree(a);
}

__global__
void test_external_assign(int *z){
    ttl::Tensor<2, 2, int*> A(z);
    A = {0,1,2,3};
}

TEST(ExternalTensor, Assign) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *z = new int[4];

// initialize memory on device and host
    cudaMallocManaged(&z, 4*sizeof(int));

// launch kernel
    test_external_assign<<<1,1>>>(z);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(z[0], 0);
    EXPECT_EQ(z[1], 1);
    EXPECT_EQ(z[2], 2);
    EXPECT_EQ(z[3], 3);

// garbage
    cudaFree(z);
}

__global__
void test_external_assign_zero_suffix(int *z){
    ttl::Tensor<2, 2, int*> A(z);
    A = {0,1,2};
}

TEST(ExternalTensor, AssignZeroSuffix) {

// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a = new int[4];

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_external_assign_zero_suffix<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
    EXPECT_EQ(a[2], 2);
    EXPECT_EQ(a[3], 0);
}

__global__
void test_assign_ignore_overflow(int *a){
    ttl::Tensor<2, 2, int*> A(a);
    A = {0, 1, 2, 3, 4, 5, 6, 7};
}

TEST(ExternalTensor, AssignIgnoreOverflow) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 8*sizeof(int));

// launch kernel
    test_assign_ignore_overflow<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
for (int ii = 0; ii < 8; ii++){
    if (ii > 3){
        EXPECT_EQ(*(a+ii), 0);
    }
    else{
        EXPECT_EQ(*(a+ii), ii);
    }
}

// garbage
    cudaFree(a);
}

__global__
void test_external_assign_internal(int *a){
    ttl::Tensor<2, 2, int> B = {0,1,2,3};
    ttl::Tensor<2, 2, int*> A(a);
    A = B;
}

TEST(ExternalTensor, AssignInternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_external_assign_internal<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
    EXPECT_EQ(*a, 0);
    EXPECT_EQ(*(a+1), 1);
    EXPECT_EQ(*(a+2), 2);
    EXPECT_EQ(*(a+3), 3);

// garbage
    cudaFree(a);
}

__global__
void test_external_move_internal(int *a){
    ttl::Tensor<2, 2, int*> A(a);
    A = std::move(ttl::Tensor<2, 2, int>{0,1,2,3});
}

TEST(ExternalTensor, MoveInternal) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    int *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(int));

// launch kernel
    test_external_move_internal<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
for(int ii = 0; ii < 4; ii++){
    EXPECT_EQ(*(a+ii), ii);
}

// garbage  
    cudaFree(a);
}

__global__
void test_external_fill(double *a){
    ttl::Tensor<2, 2, double*> A(a);
    A.fill(E);
}

TEST(ExternalTensor, Fill) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// launch kernel
    test_external_fill<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
for(int ii = 0; ii < 4; ii++){
    EXPECT_EQ(*(a+ii), E);
}

// garbage
    cudaFree(a);
}

__global__
void test_external_fill_r_value(double *a){
    ttl::Tensor<2, 2, double*>(a).fill(E);
}

TEST(ExternalTensor, FillRvalue) {
// Check for available device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    assert(nDevices > 0);

// instantiate data pointers
    double *a;

// initialize memory on device and host
    cudaMallocManaged(&a, 4*sizeof(double));

// launch kernel
    test_external_fill_r_value<<<1,1>>>(a);

// control for race conditions
    cudaDeviceSynchronize();

// check errors
    cudaCheckErrors("failed");

// verify results
for(int ii = 0; ii < 4; ii++){
    EXPECT_EQ(*(a+ii), E);
}


// garbage
    cudaFree(a);
}
