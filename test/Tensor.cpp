#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;

constexpr int index(int D, int i, int j, int k, int l) {
  return i * D * D * D + j * D * D + k * D + l;
}

constexpr int index(int D, int i, int j, int k) {
  return i * D * D + j * D + k;
}

constexpr int index(int D, int i, int j) {
  return i * D + j;
}

TEST(TensorTest, ArrayIndexing) {
  ttl::Tensor<3, double, 3> A;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        A[index(3,i,j,k)] = index(3,i,j,k);
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(A[index(3,i,j,k)], index(3,i,j,k));
      }
    }
  }
}

TEST(TensorTest, Delta) {
  // D0 and D1 are compile errors.
  // auto D0 = ttl::Delta<0, double, 1>();
  // auto D1 = ttl::Delta<1, double, 6>();
  auto D2 = ttl::Delta<2, double, 6>();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_EQ(D2[index(6,i,j)], (i == j) ? 1.0 : 0.0);
    }
  }

  auto D3 = ttl::Delta<3, double, 4>(3.14);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(D3[index(4,i,j,k)], (i == j && j == k) ? 3.14 : 0.0);
      }
    }
  }

  auto D4 = ttl::Delta<4, int, 3>(42);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          EXPECT_EQ(D4[index(3,i,j,k,l)],
                    (i == j && j == k && k == l) ? 42 : 0.0);
        }
      }
    }
  }
}

TEST(TensorTest, TensorExprAssignment) {
  ttl::Tensor<1, double, 1> a, b;
  a[0] = 10;
  b(i) = a(i);
  EXPECT_EQ(a[0], b[0]);
  // assign(b(i), a(j));

  ttl::Tensor<2, double, 3> T, U(3.14);
  T(i, j) = U(i, j);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(T[index(3,n,m)], U[index(3,n,m)]);
    }
  }

  U[index(3,1,0)] = 42;
  T(i,j) = U(j,i);
  EXPECT_EQ(T[index(3,0,1)], 42);

  ttl::Tensor<3, double, 3> A, B;
  B(i,j,k) = A(k,j,i);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      for (int o = 0; o < 3; ++o) {
        EXPECT_EQ(B[index(3,n,m,o)], A[index(3,o,m,n)]);
      }
    }
  }
}

TEST(TensorTest, UnaryOp) {
  ttl::Tensor<2, int, 2> A(1), B;
  B(i, j) = -A(i, j);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(B[index(2,i,j)], -1);
    }
  }
}

TEST(TensorTest, BinaryOp) {
  ttl::Tensor<1, double, 4> x(1), y(2), z;

  z(i) = x(i) + y(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 3);
  }

  z(i) = y(i) - x(i) - x(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 0);
  }

  z(i) = y(i) - x(i) - x(i) + y(i) - x(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 1);
  }
  // z(i) = y(i) - x(j);
}

TEST(TensorTest, ScalarOp) {
  ttl::Tensor<0, double, 1> x(1), y;
  y() = 3.1 * x();
  EXPECT_EQ(y[0], 3.1);

  x() = y() * 3.0;
  EXPECT_EQ(x[0], 9.3);

  ttl::Tensor<2, int, 2> s = ttl::Delta<2,int,2>(2);
  ttl::Tensor<2, int, 2> t;
  ttl::Tensor<2, double, 2> u;

  t(i,j) = (ttl::Tensor<2, int, 2>(2)(i,j) - s(j,i))/2;
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 1);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 0);

  s(i,j) = 1.2 * t(i,j);
  EXPECT_EQ(s[0], 0);
  EXPECT_EQ(s[1], 1);
  EXPECT_EQ(s[2], 1);
  EXPECT_EQ(s[3], 0);

  u(i,j) = 1.2 * t(i,j);
  EXPECT_EQ(u[0], 0);
  EXPECT_EQ(u[1], 1.2);
  EXPECT_EQ(u[2], 1.2);
  EXPECT_EQ(u[3], 0);

  u(i,j) = s(i,j) % 1;
  EXPECT_EQ(u[0], 0.0);
  EXPECT_EQ(u[1], 0.0);
  EXPECT_EQ(u[2], 0.0);
  EXPECT_EQ(u[3], 0.0);
}

TEST(TensorTest, TensorProduct) {
  ttl::Tensor<2,double,2> A(3), B(2), C;

  ttl::Tensor<0,double,2> s;
  s() = B(k,l) * B(k,l);
  EXPECT_EQ(s[0], 16);

  C(i,k) = A(i,j) * B(j,k);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 12);
    }
  }

  C(j,k) = A(i,j) * B(k,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 12);
    }
  }

  C(i,j) = (A(i,j) * B(k,l)) * B(k,l);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  C(i,j) = A(i,j) * s();
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  C(i,j) = A(i,j) * (B(k,l) * B(k,l));
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  ttl::Tensor<4,double,2> D;

  D(i,j,k,l) = A(i,j) * B(k,l);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      for (int o = 0; o < 2; ++o) {
        for (int p = 0; p < 2; ++p) {
          EXPECT_EQ(D[index(2,n,m,o,p)], 6);
        }
      }
    }
  }

  ttl::Tensor<1,double,2> x, y;
  x[0] = 2;
  x[1] = 3;

  auto E = ttl::Delta<2,double,2>();
  E[1] = 1;
  y(i) = (B(i,j) + E(i,j)) * x(j);
  EXPECT_EQ(y[0], 15);
  EXPECT_EQ(y[1], 13);

  y(j) = (B(i,j) + E(i,j)) * x(i);
  EXPECT_EQ(y[0], 12);
  EXPECT_EQ(y[1], 15);

  y(l) = D(i,j,k,l) * 0.5 * ((B(i,j) + E(i,j)) + (B(j,i) + E(i,j))) * x(k);
  EXPECT_EQ(y[0], 330);
  EXPECT_EQ(y[1], 330);
}
