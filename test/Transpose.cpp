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

const ttl::Index<'i'> i;
const ttl::Index<'j'> j;
const ttl::Index<'k'> k;

TEST(Transpose, Basic_2_2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4}, B = transpose(A(i,j));
  for (int n = 0; n < 2; ++n)
    for (int m = 0; m < 2; ++m)
      EXPECT_EQ(B(n,m), A(m,n));
}

TEST(Transpose, Basic_2_3) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8}, B = transpose(A(i,j));
  for (int n = 0; n < 3; ++n)
    for (int m = 0; m < 3; ++m)
      EXPECT_EQ(B(n,m), A(m,n));
}

TEST(Transpose, Basic_3_3) {
  ttl::Tensor<3,3,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,
                            13,14,15,16,17,18,19,20,21},
                       B = transpose(A(i,j,k));
  for (int n = 0; n < 3; ++n)
    for (int m = 0; m < 3; ++m)
      for (int o = 0; o < 3; ++o)
        EXPECT_EQ(B(n,m,o), A(o,m,n));
}

