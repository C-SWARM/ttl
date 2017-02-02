#include <ttl/ttl.h>
#include <gtest/gtest.h>

#include <ttl/Library/solve.h>

using namespace ttl;

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'k'> k;

TEST(Inverse, Basic_2_2) {
  ttl::Tensor<2,2,double> A = {1, 2, 3, 5};
  auto B = ttl::inverse(A);
  EXPECT_EQ(B[0][0], -5);
  EXPECT_EQ(B[0][1], 2);
  EXPECT_EQ(B[1][0], 3);
  EXPECT_EQ(B[1][1], -1);
}

TEST(Inverse, Basic_2_3) {
  const Tensor<2,3,const double> A = {1, 2, 3,
                                      4, 5, 6,
                                      7, 8, 10};
  Tensor<2,3,const double> B = inverse(A);
  std::cout << B(i,j) << "\n";
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, Extern_2_3) {
  const double a[9] = {1, 2, 3,
                       4, 5, 6,
                       7, 8, 10};
  const Tensor<2,3,const double*> A(a);
  auto B = inverse(A);
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, RValue_2_3) {
  const Tensor<2,3,double> B = inverse(Tensor<2,3,double>{1, 2, 3, 4, 5, 6, 7, 8, 10});
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, RValue_2_3_Infer) {
  auto B = inverse(Tensor<2,3,double>{1, 2, 3, 4, 5, 6, 7, 8, 10});
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, RValueExpression_2_3) {
  const Tensor<2,3,double> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 10},
                           C = {},
                           B = inverse(1*(A(i,j) + C(i,j)));
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, Expression_2_3) {
  const Tensor<2,3,double> A = {1, 2, 3,
                                4, 5, 6,
                                7, 8, 10},
                           C = {};
  auto e = 1*(A(i,j) + C(i,j));
  const Tensor<2,3,double> B = inverse(e);
  EXPECT_EQ(B(0,0), -2/3.0);
  EXPECT_EQ(B(0,1), -(1.0 + 1/3.0));
  EXPECT_EQ(B(0,2), 1);
  EXPECT_EQ(B(1,0), -2/3.0);
  EXPECT_EQ(B(1,1), 3.0 + 2/3.0);
  EXPECT_EQ(B(1,2), -2);
  EXPECT_EQ(B(2,0), 1);
  EXPECT_EQ(B(2,1), -2);
  EXPECT_EQ(B(2,2), 1);
}

TEST(Inverse, Basic_4_2) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  ttl::Index<'k'> k;
  ttl::Index<'l'> l;
  ttl::Index<'m'> m;
  ttl::Index<'n'> n;

  ttl::Tensor<4,2,double> A = {1,2,3,4,
                               9,8,7,6,
                               11,13,12,14,
                               -16,17,-18,19},
                          B = ttl::inverse(A),
                          C = B(i,j,k,l)*A(k,l,m,n), I;
  I(i,j,k,l) = ttl::identity(i,j,k,l);

  for (int q = 0; q < 2; ++q) {
    for (int r = 0; r < 2; ++r) {
      for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
          EXPECT_NEAR(C(q,r,s,t), I(q,r,s,t), 1e-13);
        }
      }
    }
  }
}

TEST(Inverse, Basic_4_3) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  ttl::Index<'k'> k;
  ttl::Index<'l'> l;
  ttl::Index<'m'> m;
  ttl::Index<'n'> n;

  ttl::Tensor<4,3,double> A, I;
  I(i,j,k,l) = ttl::identity(i,j,k,l);
  for (int q = 0; q < 3; ++q) {
    for (int r = 0; r < 3; ++r) {
      for (int s = 0; s < 3; ++s) {
        for (int t = 0; t < 3; ++t) {
          A(q,r,s,t) = (double)(q*27+r*9+s*3+t+1);
        }
      }
      A(q,r,q,r) = 1.;
    }
  }

  auto B = ttl::inverse(A);
  auto C = B(i,j,k,l)*A(k,l,m,n);

  for (int q = 0; q < 2; ++q) {
    for (int r = 0; r < 2; ++r) {
      for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
          EXPECT_NEAR(C(q,r,s,t), I(q,r,s,t), 1e-13);
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

  x = decltype(x){};
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

  x = decltype(x){};
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

TEST(Solve, SingularException) {
  const Tensor<2,3,double> A = {1,2,3,1,2,4,1,2,5};
  const Tensor<1,3,double> b = {1,2,3};

  int singular = 0;
  try {
    const Tensor<1,3,double> x = solve(A,b);
  } catch (int i) {
    std::cout << "Saw solve error: " << i << "\n";
    singular = 1;
  }

  EXPECT_EQ(singular, 1);
}

TEST(Solve, SingularOut) {
  const Tensor<2,3,double> A = {1,2,3,1,2,4,1,2,5};
  const Tensor<1,3,double> b = {1,2,3};
  Tensor<1,3,double> x;

  int singular = 0;
  if (int i = solve(A,b,x)) {
    std::cout << "Saw solve error: " << i << "\n";
    singular = 1;
  }

  EXPECT_EQ(singular, 1);
}
