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

__global__
void test_2_2(int *e){
  auto d = ttl::epsilon<2>(i,j);

  ttl::Tensor<2,2,int*> E {e};
  E = d;
}

TEST(Epsilon, 2_2) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *e;  
    
//initialize unified memory on device and host
  cudaMallocManaged(&e, 4*sizeof(int)); 

//temp tensor store
  ttl::Tensor<2,2,int*> E {e};

// launch kernel
  test_2_2<<<1,1>>>(e);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(E(0,0), 0); EXPECT_EQ(E(0,1), 1);
  EXPECT_EQ(E(1,0), -1); EXPECT_EQ(E(1,1), 0);  

// garbage
  cudaFree(e);
}

__global__
void test_3_3(int *e){
  auto f = ttl::epsilon<3>(i,j,k);

  ttl::Tensor<3,3,int*> E{e};
  E = f;
}

TEST(Epsilon, 3_3) {
// check device availability
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

//instantiate data pointers
  int *e; 
    
//initialize unified memory on device and host
  cudaMallocManaged(&e, 27*sizeof(int)); 

// launch storage Tensor
  ttl::Tensor<3, 3, int*> E = {e};

// launch kernel
  test_3_3<<<1,1>>>(e);

// control for race conditions  
  cudaDeviceSynchronize();
    
// check errors
  cudaCheckErrors("failed");

//compare values
  EXPECT_EQ(E(0,0,0), 0); EXPECT_EQ(E(0,0,1),  0); EXPECT_EQ(E(0,0,2), 0);
  EXPECT_EQ(E(0,1,0), 0); EXPECT_EQ(E(0,1,1),  0); EXPECT_EQ(E(0,1,2), 1);
  EXPECT_EQ(E(0,2,0), 0); EXPECT_EQ(E(0,2,1), -1); EXPECT_EQ(E(0,2,2), 0);

  EXPECT_EQ(E(1,0,0), 0); EXPECT_EQ(E(1,0,1), 0); EXPECT_EQ(E(1,0,2), -1);
  EXPECT_EQ(E(1,1,0), 0); EXPECT_EQ(E(1,1,1), 0); EXPECT_EQ(E(1,1,2),  0);
  EXPECT_EQ(E(1,2,0), 1); EXPECT_EQ(E(1,2,1), 0); EXPECT_EQ(E(1,2,2),  0);

  EXPECT_EQ(E(2,0,0),  0); EXPECT_EQ(E(2,0,1), 1); EXPECT_EQ(E(2,0,2), 0);
  EXPECT_EQ(E(2,1,0), -1); EXPECT_EQ(E(2,1,1), 0); EXPECT_EQ(E(2,1,2), 0);
  EXPECT_EQ(E(2,2,0),  0); EXPECT_EQ(E(2,2,1), 0); EXPECT_EQ(E(2,2,2), 0);

// garbage
  cudaFree(e);  
}