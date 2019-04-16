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
#include <ttl2/ttl.hpp>
#include <gtest/gtest.h>

// const int e[] = {0,1,2,3,4,5,6,7};
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

static const ttl::Tensor<2,2,int> B = {0,1,2,3};
static const ttl::Tensor<2,2,const int> C = {0,1,2,3};
// static const ttl::Tensor<2,2,const int*> E(e);

TEST(Bind, InitializeRValue) {
  ttl::Tensor<2,2,int> A = 2 * B(i,j);
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(Bind, InitializeLValue) {
  auto e = 2 * B(i,j);
  ttl::Tensor<2,2,int> A = e;
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(Bind, Assign) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j);
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Bind, AssignRValueExpression) {
  ttl::Tensor<2,2,int> A;
  A = 2 * B(i,j);
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(Bind, AssignLValueExpression) {
  auto b = 2 * B(i,j);
  ttl::Tensor<2,2,int> A;
  A = b;
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(Bind, Accumulate) {
  ttl::Tensor<2,2,int> A = {};
  A(i,j) += B(i,j);
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Bind, AssignFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(i,j);
  EXPECT_EQ(C[0], A[0]);
  EXPECT_EQ(C[1], A[1]);
  EXPECT_EQ(C[2], A[2]);
  EXPECT_EQ(C[3], A[3]);
}

// TEST(Bind, AssignFromExternal) {
//   ttl::Tensor<2,2,int> A;
//   A(i,j) = E(i,j);
//   EXPECT_EQ(e[0], A[0]);
//   EXPECT_EQ(e[1], A[1]);
//   EXPECT_EQ(e[2], A[2]);
//   EXPECT_EQ(e[3], A[3]);
// }

// TEST(Bind, AssignToExternal) {
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A(i,j) = B(i,j);
//   EXPECT_EQ(a[0], B[0]);
//   EXPECT_EQ(a[1], B[1]);
//   EXPECT_EQ(a[2], B[2]);
//   EXPECT_EQ(a[3], B[3]);
// }

// TEST(Bind, AssignExternal) {
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A(i,j) = E(i,j);
//   EXPECT_EQ(a[0], e[0]);
//   EXPECT_EQ(a[1], e[1]);
//   EXPECT_EQ(a[2], e[2]);
//   EXPECT_EQ(a[3], e[3]);
// }

// TEST(Bind, AssignPermute) {
//   ttl::Tensor<2,2,int> A;
//   A(i,j) = B(j,i);
//   EXPECT_EQ(B[0], A[0]);
//   EXPECT_EQ(B[1], A[2]);
//   EXPECT_EQ(B[2], A[1]);
//   EXPECT_EQ(B[3], A[3]);
// }

// TEST(Bind, AssignPermuteFromConst) {
//   ttl::Tensor<2,2,int> A;
//   A(i,j) = C(j,i);
//   EXPECT_EQ(C[0], A[0]);
//   EXPECT_EQ(C[1], A[2]);
//   EXPECT_EQ(C[2], A[1]);
//   EXPECT_EQ(C[3], A[3]);
// }

// TEST(Bind, AssignPermuteFromExternal) {
//   ttl::Tensor<2,2,int> A;
//   A(i,j) = E(j,i);
//   EXPECT_EQ(e[0], A[0]);
//   EXPECT_EQ(e[1], A[2]);
//   EXPECT_EQ(e[2], A[1]);
//   EXPECT_EQ(e[3], A[3]);
// }

// TEST(Bind, AssignPermuteToExternal) {
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A(i,j) = B(j,i);
//   EXPECT_EQ(a[0], B[0]);
//   EXPECT_EQ(a[1], B[2]);
//   EXPECT_EQ(a[2], B[1]);
//   EXPECT_EQ(a[3], B[3]);
// }

// TEST(Bind, AssignPermuteExternal) {
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A(i,j) = E(j,i);
//   EXPECT_EQ(a[0], e[0]);
//   EXPECT_EQ(a[1], e[2]);
//   EXPECT_EQ(a[2], e[1]);
//   EXPECT_EQ(a[3], e[3]);
// }

// TEST(Bind, ExternalInitializeRValue) {
//   int a[4];
//   ttl::Tensor<2,2,int*> {a, 2 * B(i,j)};
//   EXPECT_EQ(2 * B[0], a[0]);
//   EXPECT_EQ(2 * B[1], a[1]);
//   EXPECT_EQ(2 * B[2], a[2]);
//   EXPECT_EQ(2 * B[3], a[3]);
// }

// TEST(Bind, ExternalInitializeLValue) {
//   auto e = 2 * B(i,j);
//   int a[4];
//   ttl::Tensor<2,2,int*> {a, e};
//   EXPECT_EQ(2 * B[0], a[0]);
//   EXPECT_EQ(2 * B[1], a[1]);
//   EXPECT_EQ(2 * B[2], a[2]);
//   EXPECT_EQ(2 * B[3], a[3]);
// }

// TEST(Bind, ExternalAssignRValueExpression) {
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A = 2 * B(i,j);
//   EXPECT_EQ(2 * B[0], a[0]);
//   EXPECT_EQ(2 * B[1], a[1]);
//   EXPECT_EQ(2 * B[2], a[2]);
//   EXPECT_EQ(2 * B[3], a[3]);
// }

// TEST(Bind, ExternalAssignLValueExpression) {
//   auto e = 2 * B(i,j);
//   int a[4];
//   ttl::Tensor<2,2,int*> A(a);
//   A = e;
//   EXPECT_EQ(2 * B[0], a[0]);
//   EXPECT_EQ(2 * B[1], a[1]);
//   EXPECT_EQ(2 * B[2], a[2]);
//   EXPECT_EQ(2 * B[3], a[3]);
// }

TEST(Bind, Trace2x2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  int t = A(i,i);
  EXPECT_EQ(t, 5);
}

TEST(Bind, Trace2x3) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8,9};
  auto t = A(i,i);
  EXPECT_EQ(t, 15);
}

TEST(Bind, Trace3x2) {
  ttl::Tensor<3,2,int> A = {1,2,3,4,5,6,7,8};
  auto t = A(i,i,i);
  EXPECT_EQ(t, A[0] + A[7]);
}

TEST(Bind, ParallelContract) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  auto t = A(i,i,j,j);
  EXPECT_EQ(t, A[0] + A[3] + A[12] + A[15]);
}

TEST(Bind, SequentialContract) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  ttl::Tensor<2,2,int> B = A(i,i,j,k);
  auto t = B(j,j);
  EXPECT_EQ(t, B[0] + B[3]);
}

TEST(Bind, ProjectionRead1) {
  ttl::Tensor<1,2,int> A = {0,1};
  EXPECT_EQ(A(0), A[0]);
  EXPECT_EQ(A(1), A[1]);
}

TEST(Bind, ProjectionRead2) {
  ttl::Tensor<2,2,int> A = {0,1,2,3};
  EXPECT_EQ(A(0,0), 0);
  EXPECT_EQ(A(0,1), 1);
  EXPECT_EQ(A(1,0), 2);
  EXPECT_EQ(A(1,1), 3);
}


TEST(Bind, ProjectionRead2_1) {
  ttl::Tensor<2,2,int> A = {0,1,2,3};
  ttl::Tensor<1,2,int> v = A(1,i);
  EXPECT_EQ(v(0), 2);
  EXPECT_EQ(v(1), 3);
}

TEST(Bind, ProjectionRead3) {
  ttl::Tensor<3,2,int> A = {0,1,2,3,4,5,6,7};
  int d = A(0,1,0);
  EXPECT_EQ(d, 2);
  ttl::Tensor<2,2,int> B = A(i,1,j);
  EXPECT_EQ(B(0,0), A(0,1,0));
  EXPECT_EQ(B(0,1), A(0,1,1));
  EXPECT_EQ(B(1,0), A(1,1,0));
  EXPECT_EQ(B(1,1), A(1,1,1));
  ttl::Tensor<1,2,int> v = A(1,i,0);
  EXPECT_EQ(v(0), A(1,0,0));
  EXPECT_EQ(v(1), A(1,1,0));
  v = A(i,1,1);
  EXPECT_EQ(v(0), A(0,1,1));
  EXPECT_EQ(v(1), A(1,1,1));
}

TEST(Bind, ProjectionWrite) {
  ttl::Tensor<1,2,int> A;
  A(0) = 0;
  A(1) = 1;
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
}

TEST(Bind, ProjectionWrite2) {
  ttl::Tensor<2,2,int> A;
  A(0,0) = 0;
  A(0,1) = 1;
  A(1,0) = 2;
  A(1,1) = 3;
  EXPECT_EQ(A(0,0), 0);
  EXPECT_EQ(A(0,1), 1);
  EXPECT_EQ(A(1,0), 2);
  EXPECT_EQ(A(1,1), 3);
}

TEST(Bind, ProjectionWriteVector) {
  ttl::Tensor<2,2,int> A = {};
  ttl::Tensor<1,2,int> v = {1,2};

  A(i,0) = v(i);
  EXPECT_EQ(A(0,0), 1);
  EXPECT_EQ(A(1,0), 2);

  A(1,i) = v(i);
  EXPECT_EQ(A(1,0), 1);
  EXPECT_EQ(A(1,1), 2);
}

TEST(Bind, ProjectionWriteMatrix) {
  ttl::Tensor<3,2,int> A = {};
  ttl::Tensor<2,2,int> M = {1,2,3,4};

  A(i,0,j) = M(i,j);
  EXPECT_EQ(A(0,0,0), 1);
  EXPECT_EQ(A(0,0,1), 2);
  EXPECT_EQ(A(1,0,0), 3);
  EXPECT_EQ(A(1,0,1), 4);
}

TEST(Bind, ProjectionProduct) {
  ttl::Tensor<2,2,int> A = {}, B={1,2,3,4};
  ttl::Tensor<3,2,int> C={1,2,3,4,5,6,7,8};

  A(i,0) = B(j,i)*C(1,j,0);
  EXPECT_EQ(A(0,0), 26);
  EXPECT_EQ(A(1,0), 38);
}

TEST(Bind, Curry) {
  ttl::Tensor<1,2,int> A = {1,2};
  auto f = A(j);
  EXPECT_EQ(f(0), 1);
  EXPECT_EQ(f(1), 2);
}

// TEST(Bind, PermuteSubtree) {
//   ttl::Tensor<2,2,int> A = {1,2,3,4},
//                        B = {1,3,2,4};
//   auto f = A(i,j).to(j,i);
//   EXPECT_EQ(f(0,0), B(0,0));
//   EXPECT_EQ(f(0,1), B(0,1));
//   EXPECT_EQ(f(1,0), B(1,0));
//   EXPECT_EQ(f(1,1), B(1,1));
// }
