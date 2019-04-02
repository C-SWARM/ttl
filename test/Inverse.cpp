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

using namespace ttl;

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'k'> k;
static constexpr Index<'l'> l;
static constexpr Index<'m'> m;
static constexpr Index<'n'> n;

TEST(Inverse, Basic_2_2) {
  ttl::Tensor<2,2,double> A = {1, 2, 3, 5},
                          B = ttl::inverse(A),
                    INVERSE = {-5, 2, 3, -1};

  EXPECT_DOUBLE_EQ(B(0,0), INVERSE(0,0));
  EXPECT_DOUBLE_EQ(B(0,1), INVERSE(0,1));
  EXPECT_DOUBLE_EQ(B(1,0), INVERSE(1,0));
  EXPECT_DOUBLE_EQ(B(1,1), INVERSE(1,1));

  int e = ttl::inverse(A,B);
  EXPECT_EQ(e, 0);

  EXPECT_DOUBLE_EQ(B(0,0), INVERSE(0,0));
  EXPECT_DOUBLE_EQ(B(0,1), INVERSE(0,1));
  EXPECT_DOUBLE_EQ(B(1,0), INVERSE(1,0));
  EXPECT_DOUBLE_EQ(B(1,1), INVERSE(1,1));

  B = zero(i,j);
  e = ttl::inverse<2, 2, double, false>(A,B);
  EXPECT_EQ(e, 0);

  EXPECT_DOUBLE_EQ(B(0,0), INVERSE(0,0));
  EXPECT_DOUBLE_EQ(B(0,1), INVERSE(0,1));
  EXPECT_DOUBLE_EQ(B(1,0), INVERSE(1,0));
  EXPECT_DOUBLE_EQ(B(1,1), INVERSE(1,1));

}

TEST(Inverse, Singular_2_2) {
  ttl::Tensor<2,2,double> A = {1, 1, 3, 3},
                              B;
  int singular = 0;
  try {
    B = ttl::inverse(A);
  } catch (int) {
    singular = 1;
  }
  EXPECT_NE(singular, 0);

  singular = ttl::inverse(A, B);
  EXPECT_NE(singular, 0);
}

TEST(Inverse, Basic_2_3) {
  const Tensor<2,3,double> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 10},
                           B = inverse(A),
                           C = B(i,j)*A(j,k),
                          ID = identity(i,j);

  EXPECT_NEAR(C(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(C(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(C(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(C(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(C(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(C(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(C(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(C(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(C(2,2), ID(2,2), 1e-14);

  Tensor<2,3,double> D;
  int singular = inverse(A(i,j), D);
  EXPECT_EQ(singular, 0);

  Tensor<2,3,double> E = D(i,j)*A(j,k);

  EXPECT_NEAR(E(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(E(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(E(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(E(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(E(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(E(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(E(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(E(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(E(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, Singular_2_3) {
  ttl::Tensor<2,3,double> A = {1,2,3,4,5,6,7,8,9};
  int singular = 0;
  try {
    ttl::inverse(A);
  } catch (int) {
    singular = 1;
  }
  EXPECT_NE(singular, 0);

  decltype(A) B = zero(i,j);
  singular = inverse(A,B);
  EXPECT_NE(singular, 0);
}

TEST(Inverse, Extern_2_3) {
  double a[9] = {1, 2, 3,
                 4, 5, 6,
                 7, 8, 10};
  const Tensor<2,3,double*> A(a);
  auto B = inverse(A);

  Tensor<2,3,double> C = B(i,j)*A(j,k);
  Tensor<2,3,double> ID = identity(i,j);

  EXPECT_NEAR(C(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(C(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(C(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(C(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(C(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(C(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(C(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(C(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(C(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, RValue_2_3) {
  const Tensor<2,3,double> A = Tensor<2,3,double>{1, 2, 3, 4, 5, 6, 7, 8, 10};
  const Tensor<2,3,double> B = inverse(Tensor<2,3,double>{A});
  const Tensor<2,3,double> C = B(i,j)*A(j,k);
  const Tensor<2,3,double> ID = identity(i,j);

  EXPECT_NEAR(C(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(C(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(C(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(C(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(C(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(C(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(C(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(C(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(C(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, RValue_2_3_Infer) {
  const Tensor<2,3,double> A = {1, 2, 3, 4, 5, 6, 7, 8, 10};
  auto B = inverse(Tensor<2,3,double>{A});
  const Tensor<2,3,double> C = B(i,j)*A(j,k);
  const Tensor<2,3,double> ID = identity(i,j);

  EXPECT_NEAR(C(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(C(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(C(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(C(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(C(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(C(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(C(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(C(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(C(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, RValueExpression_2_3) {
  const Tensor<2,3,double> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 10},
                           B = {},
                           C = inverse(1*(A(i,j) + B(i,j))),
                           D = C(i,j)*A(j,k),
                          ID = identity(i,j);

  EXPECT_NEAR(D(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(D(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(D(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(D(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(D(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(D(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(D(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(D(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(D(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, Expression_2_3) {
  const Tensor<2,3,double> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 10},
                           B = {},
                          ID = identity(i,j);

  auto e = 1*(A(i,j) + B(i,j));
  auto C = inverse(e);
  auto D = expressions::force(C(i,j) * A(j,k));

  EXPECT_NEAR(D(0,0), ID(0,0), 1e-14);
  EXPECT_NEAR(D(0,1), ID(0,1), 1e-14);
  EXPECT_NEAR(D(0,2), ID(0,2), 1e-14);
  EXPECT_NEAR(D(1,0), ID(1,0), 1e-14);
  EXPECT_NEAR(D(1,1), ID(1,1), 1e-14);
  EXPECT_NEAR(D(1,2), ID(1,2), 1e-14);
  EXPECT_NEAR(D(2,0), ID(2,0), 1e-14);
  EXPECT_NEAR(D(2,1), ID(2,1), 1e-14);
  EXPECT_NEAR(D(2,2), ID(2,2), 1e-14);
}

TEST(Inverse, Basic_4_2) {
  ttl::Tensor<4,2,double> A = {1,2,3,4,
                               9,8,7,6,
                               11,13,12,14,
                               -16,17,-18,19},
                          B = inverse(A),
                          C = B(i,j,k,l)*A(k,l,m,n),
                         ID = identity(i,j,k,l);

  for (int q = 0; q < 2; ++q) {
    for (int r = 0; r < 2; ++r) {
      for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
          EXPECT_NEAR(C(q,r,s,t), ID(q,r,s,t), 1e-13);
        }
      }
    }
  }

  int singular = inverse(A, B);
  EXPECT_EQ(singular, 0);

  C = B(i,j,k,l)*A(k,l,m,n);
  for (int q = 0; q < 2; ++q) {
    for (int r = 0; r < 2; ++r) {
      for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
          EXPECT_NEAR(C(q,r,s,t), ID(q,r,s,t), 1e-13);
        }
      }
    }
  }
}

TEST(Inverse, Singular_4_2) {
  ttl::Tensor<4,2,double> A = {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4};
  int singular = 0;
  try {
    ttl::inverse(A);
  } catch (int i) {
    singular = i;
  }
  EXPECT_NE(singular, 0);

  decltype(A) B;
  singular = inverse(A,B);
  EXPECT_NE(singular, 0);
}

TEST(Inverse, Basic_4_3) {
  ttl::Tensor<4,3,double> A, B, ID = identity(i,j,k,l);
  for (int q = 0; q < 3; ++q) {
    for (int r = 0; r < 3; ++r) {
      for (int s = 0; s < 3; ++s) {
        for (int t = 0; t < 3; ++t) {
          A(q,r,s,t) = 27.*q + 9.*r + 3.*s + 1.*t + 1.;
        }
      }
      A(q,r,q,r) = 1.;
    }
  }

  try {
    B = inverse(A);
  }
  catch (int e) {
    std::cerr << "Matrix A found to be singular in column " << e << "\n";
    FAIL();
  }

  std::cerr << B(i,j,k,l) << "\n";

  auto C = force(B(i,j,k,l) * A(k,l,m,n));

  for (int q = 0; q < 2; ++q) {
    for (int r = 0; r < 2; ++r) {
      for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
          EXPECT_NEAR(C(q,r,s,t), ID(q,r,s,t), 1e-13);
        }
      }
    }
  }
}

TEST(Solve, Basic_2_2) {
  Tensor<2,2,double> A = {1,2,3,4};
  Tensor<1,2,double> b = {1,2},
                     x = solve(A,b),
                     y = A(i,j)*x(j);
  EXPECT_DOUBLE_EQ(y(0), b(0));
  EXPECT_DOUBLE_EQ(y(1), b(1));

  x = zero(i);
  solve(A,b,x);
  y = A(i,j)*x(j);
  EXPECT_DOUBLE_EQ(y(0), b(0));
  EXPECT_DOUBLE_EQ(y(1), b(1));
}

TEST(Solve, Expression_2_2) {
  Tensor<2,2,double> A = {1,2,3,4};
  Tensor<1,2,double> b = {1,2},
                     x = solve(A(i,j),b),
                     y = A(i,j)*x(j);

  EXPECT_DOUBLE_EQ(y(0), b(0));
  EXPECT_DOUBLE_EQ(y(1), b(1));

  x = zero(i);
  solve(A(i,j),b(j),x);
  y = A(i,j)*x(j);
  EXPECT_DOUBLE_EQ(y(0), b(0));
  EXPECT_DOUBLE_EQ(y(1), b(1));
}

TEST(Solve, Basic_2_3) {
  const Tensor<2,3,double> A = {1,2,3,4,5,6,7,8,10};
  const Tensor<1,3,double> b = {1,2,3},
                           x = solve(A,b),
                           y = A(i,j)*x(j);

  EXPECT_DOUBLE_EQ(y(0), b(0));
  EXPECT_DOUBLE_EQ(y(1), b(1));
  EXPECT_DOUBLE_EQ(y(2), b(2));
}

TEST(Solve, Singular) {
  const Tensor<2,3,double> A = {1,2,3,1,2,4,1,2,5};
  const Tensor<1,3,double> b = {1,2,3};

  int singular = 0;
  try {
    solve(A,b);
  } catch (int i) {
    singular = i;
  }
  EXPECT_NE(singular, 0);

  Tensor<1,3,double> x;
  singular = solve(A,b,x);
  EXPECT_NE(singular, 0);
}

TEST(LU, Basic_2_3) {
  Tensor<2,3,double> B = {
    7, -2, 1,
    7, -2, 1,
    7, -2, 1
  };

  int perm[3];

  ttl::lib::detail::lu_ikj_pp<3>(
      [&B](int i, int j) -> double& {
        return B[i][j];
      },
      [&perm](int i) -> int& {
        return perm[i];
      });

  std::cout.precision(17);
  std::cout << ttl::lib::bind(B) << "\n";
}
