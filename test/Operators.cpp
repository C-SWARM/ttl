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

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;

TEST(UnaryOp, Negate) {
  ttl::Tensor<2, 2, int> A, B = {0,1,2,3};
  A(i,j) = -B(i,j);
  EXPECT_EQ(A[0][0], -B[0][0]);
  EXPECT_EQ(A[0][1], -B[0][1]);
  EXPECT_EQ(A[1][0], -B[1][0]);
  EXPECT_EQ(A[1][1], -B[1][1]);
}

TEST(ScalarOp, MultiplyRHS) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = PI * B(i,j);
  EXPECT_EQ(A[0][0], PI * E);
  EXPECT_EQ(A[0][1], PI * E);
  EXPECT_EQ(A[1][0], PI * E);
  EXPECT_EQ(A[1][1], PI * E);
}

TEST(ScalarOp, MultiplyLHS) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = B(i,j) * PI;
  EXPECT_EQ(A[0][0], PI * E);
  EXPECT_EQ(A[0][1], PI * E);
  EXPECT_EQ(A[1][0], PI * E);
  EXPECT_EQ(A[1][1], PI * E);
}

TEST(ScalarOp, Divide) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = B(i,j) / PI;
  EXPECT_EQ(A[0][0], E / PI);
  EXPECT_EQ(A[0][1], E / PI);
  EXPECT_EQ(A[1][0], E / PI);
  EXPECT_EQ(A[1][1], E / PI);
}

TEST(ScalarOp, Modulo) {
  ttl::Tensor<2, 2, int> A, B = {0,1,2,3};
  A(i,j) = B(i,j) % 3;
  EXPECT_EQ(A[0][0], 0);
  EXPECT_EQ(A[0][1], 1);
  EXPECT_EQ(A[1][0], 2);
  EXPECT_EQ(A[1][1], 0);
}

TEST(BinaryOp, Add) {
  const ttl::Tensor<2, 2, int> A = {0,1,2,3}, B = {1,2,3,4};
  ttl::Tensor<2, 2, int> C;
  C(i,j) = A(i,j) + B(i,j);
  EXPECT_EQ(C[0][0], 1);
  EXPECT_EQ(C[0][1], 3);
  EXPECT_EQ(C[1][0], 5);
  EXPECT_EQ(C[1][1], 7);
}

TEST(BinaryOp, Subtract) {
  const ttl::Tensor<2, 2, int> A = {0,1,2,3}, B = {1,2,3,4};
  ttl::Tensor<2, 2, int> C;
  C(i,j) = A(i,j) - B(i,j);
  EXPECT_EQ(C[0][0], -1);
  EXPECT_EQ(C[0][1], -1);
  EXPECT_EQ(C[1][0], -1);
  EXPECT_EQ(C[1][1], -1);
}

TEST(TensorProduct, Multiply) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C;
  C(i,j) = A(i,k) * B(k,j);
  EXPECT_EQ(C[0][0], 10);
  EXPECT_EQ(C[0][1], 13);
  EXPECT_EQ(C[1][0], 22);
  EXPECT_EQ(C[1][1], 29);
}

TEST(TensorProduct, Inner) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<0,2,int> c;
  c() = A(i,j) * B(i,j);
  EXPECT_EQ(c(), 40);
}

TEST(TensorProduct, Outer) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<4,2,int> C;
  C(i,j,k,l) = A(i,j) * B(k,l);
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
}

TEST(Zero, Construct) {
  ttl::Tensor<2,3,double> Z = ttl::zero(i,j);
  EXPECT_DOUBLE_EQ(Z(0,0), 0.);
  EXPECT_DOUBLE_EQ(Z(0,1), 0.);
  EXPECT_DOUBLE_EQ(Z(1,0), 0.);
  EXPECT_DOUBLE_EQ(Z(1,1), 0.);
}

TEST(Zero, Assign) {
  ttl::Tensor<2,3,double> Z;
  Z = ttl::zero(i,j);
  EXPECT_DOUBLE_EQ(Z(0,0), 0.);
  EXPECT_DOUBLE_EQ(Z(0,1), 0.);
  EXPECT_DOUBLE_EQ(Z(1,0), 0.);
  EXPECT_DOUBLE_EQ(Z(1,1), 0.);
}

TEST(Zero, AssignExpression) {
  ttl::Tensor<2,3,double> Z;
  Z(i,j) = ttl::zero(i,j);
  EXPECT_DOUBLE_EQ(Z(0,0), 0.);
  EXPECT_DOUBLE_EQ(Z(0,1), 0.);
  EXPECT_DOUBLE_EQ(Z(1,0), 0.);
  EXPECT_DOUBLE_EQ(Z(1,1), 0.);
}

TEST(Zero, AssignProduct) {
  ttl::Tensor<2,3,double> Z, A = {1,2,3,4,5,6,7,8,9};
  Z(i,k) = ttl::zero(i,j)*A(j,k);
  EXPECT_DOUBLE_EQ(Z(0,0), 0.);
  EXPECT_DOUBLE_EQ(Z(0,1), 0.);
  EXPECT_DOUBLE_EQ(Z(1,0), 0.);
  EXPECT_DOUBLE_EQ(Z(1,1), 0.);
}
