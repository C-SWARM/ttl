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

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

static ttl::Tensor<2,2,int> B = {0, 1, 2, 3};

TEST(IndexMap, Identity) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j).to(i,j);
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[0][1]);
  EXPECT_EQ(B[1][0], A[1][0]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(IndexMap, Transpose) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(j,i).to(i,j);
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[1][0]);
  EXPECT_EQ(B[1][0], A[0][1]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

TEST(IndexMap, RValue) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = ttl::Tensor<2,2,int>{0, 1, 2, 3}(j,i).to(i,j);
  EXPECT_EQ(B[0][0], A[0][0]);
  EXPECT_EQ(B[0][1], A[1][0]);
  EXPECT_EQ(B[1][0], A[0][1]);
  EXPECT_EQ(B[1][1], A[1][1]);
}

constexpr int index(int D, int i, int j, int k) {
  return i * D * D + j * D + k;
}

TEST(IndexMap, Rotation) {
  ttl::Tensor<3,2,const int> B = { 0, 1,
                                   2, 3,
                                   4, 5,
                                   6, 7};
  ttl::Tensor<3,2,int> A;
  A(i,j,k) = B(j,k,i).to(i,j,k);
  EXPECT_EQ(A[0][0][0], B[0][0][0]);
  EXPECT_EQ(A[0][0][1], B[0][1][0]);
  EXPECT_EQ(A[0][1][0], B[1][0][0]);
  EXPECT_EQ(A[1][0][0], B[0][0][1]);
  EXPECT_EQ(A[0][1][1], B[1][1][0]);
  EXPECT_EQ(A[1][1][0], B[1][0][1]);
  EXPECT_EQ(A[1][0][1], B[0][1][1]);
  EXPECT_EQ(A[1][1][1], B[1][1][1]);
}
