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

static constexpr double E = 2.72;
static constexpr double PI = 3.14;

TEST(Tensor, CtorDefault) {
  ttl::Tensor<2, 2, double> A;
  (void)A;
}

TEST(Tensor, LinearIndexing) {
  ttl::Tensor<2, 2, double> A;
  A.get(0) = 0.0;
  A.get(1) = 1.0;
  A.get(2) = E;
  A.get(3) = PI;
  EXPECT_EQ(A.get(0), 0.0);
  EXPECT_EQ(A.get(1), 1.0);
  EXPECT_EQ(A.get(2), E);
  EXPECT_EQ(A.get(3), PI);
}

TEST(Tensor, ArrayIndexing) {
  ttl::Tensor<1, 2, double> A = {0.0, 1.0};
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
}

TEST(Tensor, ExternArrayIndexing) {
  double a[2] = {0.0, 1.0};
  ttl::Tensor<1, 2, double*> A{a};
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
}

TEST(Tensor, ConstArrayIndexing) {
  const ttl::Tensor<1, 2, double> A = {0.0, 1.0};
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
}

TEST(Tensor, ArrayIndexAssignment) {
  ttl::Tensor<1, 2, double> A;
  A[0] = 0.0;
  A[1] = 1.0;
  EXPECT_EQ(A.get(0), 0.0);
  EXPECT_EQ(A.get(1), 1.0);
}

TEST(Tensor, 2DArrayIndexing) {
  ttl::Tensor<2, 2, double> A = { 0.0, 1.0, E, PI };
  EXPECT_EQ(A[0][0], 0.0);
  EXPECT_EQ(A[0][1], 1.0);
  EXPECT_EQ(A[1][0], E);
  EXPECT_EQ(A[1][1], PI);
}

TEST(Tensor, 2DArrayIndexAssignment) {
  ttl::Tensor<2, 2, double> A;
  A[0][0] = PI;
  A[0][1] = E;
  A[1][0] = 1.0;
  A[1][1] = 0.0;
  EXPECT_EQ(A.get(0), PI);
  EXPECT_EQ(A.get(1), E);
  EXPECT_EQ(A.get(2), 1.0);
  EXPECT_EQ(A.get(3), 0.0);
}

TEST(Tensor, Ctor) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, CtorZero) {
  ttl::Tensor<2, 2, int> A = {};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, CtorZeroSuffix) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, CtorIgnoreOverflow) {
  ttl::Tensor<2, 2, int> B = {};
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
  EXPECT_EQ(B[0][0], 0);
  EXPECT_EQ(B[0][1], 0);
  EXPECT_EQ(B[1][0], 0);
  EXPECT_EQ(B[1][1], 0);
}

TEST(Tensor, CtorWiden) {
  ttl::Tensor<1, 3, double> A = {int(1), float(E), PI};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], float(E));
  EXPECT_EQ(A[2], PI);
}

TEST(Tensor, ConstCtor) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, ZeroConst) {
  ttl::Tensor<2, 2, const int> A = {};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, ConstCtorZero) {
  const ttl::Tensor<2, 2, int> A = {};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, CtorConst) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, ConstCtorConst) {
  const ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, ConstCtorZeroConst) {
  const ttl::Tensor<2, 2, const int> A = {};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, CopyCtor) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorWiden) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, float> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorFromConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorToConst) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  const ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorFromConstToConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  const ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorFromConstData) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyCtorToConstData) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, const int> B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, MoveCtor) {
  ttl::Tensor<2, 2, int> A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, MoveCtorWiden) {
  ttl::Tensor<2, 2, double> A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0.0);
  EXPECT_EQ(A[0][1], 1.0);
  EXPECT_EQ(A[1][0], 2.0);
  EXPECT_EQ(A[1][1], 3.0);
}

TEST(Tensor, MoveCtorToConst) {
  const ttl::Tensor<2, 2, int> A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, MoveCtorFromConstData) {
  ttl::Tensor<2, 2, int> A = ttl::Tensor<2, 2, const int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, MoveCtorToConstData) {
  ttl::Tensor<2, 2, const int> A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, Assign) {
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, AssignZero) {
  ttl::Tensor<2, 2, int> A;
  A = {};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 0);
  EXPECT_EQ(A[1][0], 0);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, AssignZeroSuffix) {
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 0);
}

TEST(Tensor, AssignIgnoreOverflow) {
  ttl::Tensor<2, 2, int> B = {};
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
  EXPECT_EQ(B[0][0], 0);
  EXPECT_EQ(B[0][1], 0);
  EXPECT_EQ(B[1][0], 0);
  EXPECT_EQ(B[1][1], 0);
}

TEST(Tensor, AssignWiden) {
  ttl::Tensor<1, 3, double> A;
  A = {int(1), float(E), PI};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], float(E));
  EXPECT_EQ(A[2], PI);
}

TEST(Tensor, Copy) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyWiden) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, float> B;
  B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyFromConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, CopyFromConstData) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(Tensor, Move) {
  ttl::Tensor<2, 2, int> A;
  A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, MoveWiden) {
  ttl::Tensor<2, 2, double> A;
  A = ttl::Tensor<2, 2, int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0.0);
  EXPECT_EQ(A[0][1], 1.0);
  EXPECT_EQ(A[1][0], 2.0);
  EXPECT_EQ(A[1][1], 3.0);
}

TEST(Tensor, MoveFromConstData) {
  ttl::Tensor<2, 2, int> A;
  A = ttl::Tensor<2, 2, const int>{0, 1, 2, 3};
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 3);
}

TEST(Tensor, Fill) {
  ttl::Tensor<2, 2, double> A;
  A.fill(E);
  EXPECT_EQ(A[0][0], E);
  EXPECT_EQ(A[0][1], E);
  EXPECT_EQ(A[1][0], E);
  EXPECT_EQ(A[1][1], E);
}

TEST(Tensor, FillWiden) {
  ttl::Tensor<2, 2, double> A;
  A.fill(2);
  EXPECT_EQ(A[0][0], 2.0);
  EXPECT_EQ(A[0][1], 2.0);
  EXPECT_EQ(A[1][0], 2.0);
  EXPECT_EQ(A[1][1], 2.0);
}

TEST(ExternalTensor, Ctor) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int*> A(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  ttl::Tensor<2, 2, const int*> B(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  const ttl::Tensor<2, 2, int*> C(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  const ttl::Tensor<2, 2, const int*> D(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, CtorList) {
  int a[4];
  ttl::Tensor<2, 2, int*> {a, {0, 1, 2, 3}};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, CtorPointer) {
  int a[8] = {0,1,2,3,4,5,6,7};
  ttl::Tensor<2, 2, int*> A(&a[2]);
  EXPECT_EQ(A[0][0], 2);
  EXPECT_EQ(A[0][1], 3);
  EXPECT_EQ(A[1][0], 4);
  EXPECT_EQ(A[1][1], 5);
}

TEST(ExternalTensor, CtorConst) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2,2,const int*> A(a);
  const ttl::Tensor<2,2,const int*> B(a);
}

TEST(ExternalTensor, ArrayIndexing) {
  double a[4];
  ttl::Tensor<2, 2, double*> A(a);
  A[0][0] = 0.0;
  A[0][1] = 1.0;
  A[1][0] = E;
  A[1][1] = PI;
  EXPECT_EQ(a[0], 0.0);
  EXPECT_EQ(a[1], 1.0);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], PI);
}

TEST(ExternalTensor, CopyCtor) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int*> A(a);
  ttl::Tensor<2, 2, int*> B = A;
  EXPECT_EQ(B[0][0], a[0]);
  EXPECT_EQ(B[0][1], a[1]);
  EXPECT_EQ(B[1][0], a[2]);
  EXPECT_EQ(B[1][1], a[3]);
}

TEST(ExternalTensor, CopyCtorConst) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, const int*> B = A;
  EXPECT_EQ(B[0][0], a[0]);
  EXPECT_EQ(B[0][1], a[1]);
  EXPECT_EQ(B[1][0], a[2]);
  EXPECT_EQ(B[1][1], a[3]);

  const ttl::Tensor<2, 2, const int*> C(a);
  const ttl::Tensor<2, 2, const int*> D = C;
  EXPECT_EQ(D[0][0], a[0]);
  EXPECT_EQ(D[0][1], a[1]);
  EXPECT_EQ(D[1][0], a[2]);
  EXPECT_EQ(D[1][1], a[3]);
}

TEST(ExternalTensor, MoveCtor) {
  int a[4];
  ttl::Tensor<2, 2, int*> A = ttl::Tensor<2, 2, int*>(a);
  A[0][0] = 0;
  A[0][1] = 1;
  A[1][0] = 2;
  A[1][1] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, CopyCtorExternal) {
  const int a[4] = {0,1,2,3};
  const ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, int> B = A;
  B[1][1] = 0;
  EXPECT_EQ(B[0][0], 0);
  EXPECT_EQ(B[0][1], 1);
  EXPECT_EQ(B[1][0], 2);
  EXPECT_EQ(B[1][1], 0);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, MoveCtorExternal) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int> A = ttl::Tensor<2, 2, int*>(a);
  A[0][0] = 0;
  A[0][1] = 1;
  A[1][0] = 2;
  A[1][1] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, AssignExternal) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, int> B;
  B = A;
  B[1][1] = 0;
  EXPECT_EQ(B[0][0], 0);
  EXPECT_EQ(B[0][1], 1);
  EXPECT_EQ(B[1][0], 2);
  EXPECT_EQ(B[1][1], 0);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, MoveExternal) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int> A;
  A = ttl::Tensor<2, 2, int*>(a);
  A[0][0] = 0;
  A[0][1] = 1;
  A[1][0] = 2;
  A[1][1] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, Assign) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = {0,1,2,3};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, AssignZeroSuffix) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = {0, 1, 2};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 0);
}

TEST(ExternalTensor, AssignIgnoreOverflow) {
  int a[8] = {};
  ttl::Tensor<2, 2, int*> A(a);
  A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
  EXPECT_EQ(a[4], 0);
  EXPECT_EQ(a[5], 0);
  EXPECT_EQ(a[6], 0);
  EXPECT_EQ(a[7], 0);
}

TEST(ExternalTensor, AssignInternal) {
  ttl::Tensor<2, 2, int> B = {0,1,2,3};
  int a[4] ;
  ttl::Tensor<2, 2, int*> A(a);
  A = B;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, MoveInternal) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = ttl::Tensor<2, 2, int>{0,1,2,3};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, Fill) {
  double a[4];
  ttl::Tensor<2, 2, double*> A(a);
  A.fill(E);
  EXPECT_EQ(a[0], E);
  EXPECT_EQ(a[1], E);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], E);
}

TEST(ExternalTensor, FillRvalue) {
  double a[4];
  ttl::Tensor<2, 2, double*>(a).fill(E);
  EXPECT_EQ(a[0], E);
  EXPECT_EQ(a[1], E);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], E);
}
