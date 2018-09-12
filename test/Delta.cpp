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
static constexpr ttl::Index<'l'> l;

TEST(Delta, 1_1) {
  ttl::Tensor<1,1,int> d = ttl::delta(i);
  EXPECT_EQ(d(0), 1);
}

TEST(Delta, 2_2) {
  ttl::Tensor<2,2,int> d = ttl::delta(i,j);
  EXPECT_EQ(d(0,0), 1);
  EXPECT_EQ(d(1,0), 0);
  EXPECT_EQ(d(0,1), 0);
  EXPECT_EQ(d(1,1), 1);
}

TEST(Delta, 2_3) {
  ttl::Tensor<2,3,int> d = ttl::delta(i,j);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(d(i,j), (i == j) ? 1 : 0);
    }
  }
}

TEST(Delta, 3_2) {
  ttl::Tensor<3,2,int> d = ttl::delta(i,j,k);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(d(i,j,k), (i == j && j == k) ? 1 : 0);
      }
    }
  }
}

TEST(Delta, 4_3) {
  ttl::Tensor<4,3,int> d = ttl::delta(i,j,k,l);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          EXPECT_EQ(d(i,j,k,l), (i == j && j == k && k == l) ? 1 : 0);
        }
      }
    }
  }
}

TEST(Delta, Widen) {
  ttl::Tensor<2,2,int> d = 3u*ttl::delta(i,j);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(d(i,j), (i == j) ? 3 : 0);
    }
  }
}

TEST(Delta, Expression) {
  double d = 3.14;

  ttl::Tensor<2,3,double> D = d * ttl::delta(i,j);
  for (int n = 0; n < 3; ++n)
    for (int m = 0; m < 3; ++m)
      EXPECT_EQ(D(m,n), (m==n) ? d : 0.0);
}

TEST(Delta, Expression2) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8},
                       B = A(i,j) * ttl::delta(j,k),
                       C = A(i,j) * ttl::delta(k,j).to(j,k),
                       D = ((ttl::delta(i,k)*ttl::delta(j,l)).to(i,j,k,l))*A(k,l);

  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(B(m,n), A(m,n));
      EXPECT_EQ(C(m,n), A(m,n));
      EXPECT_EQ(D(m,n), A(m,n));
    }
  }

  ttl::Tensor<2,3,double> E = ttl::delta<3>(i,j)*ttl::delta<3>(j,k);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      if (n == m) EXPECT_EQ(E(n,m), 1);
      else EXPECT_EQ(E(n,m), 0);
    }
  }
}
