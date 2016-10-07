#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

int index(int i, int j, int k) {
  return i * 3 * 3 + j * 3 + k;
}

[[gnu::noinline]]
void
init(ttl::Tensor<3, double, 3>& T) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        T[index(i,j,k)] = index(i,j,k);
      }
    }
  }
}

[[gnu::noinline]]
void
check(ttl::Tensor<3, double, 3>& T) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(T[index(i,j,k)], index(i,j,k));
      }
    }
  }
}

template <class A, class B>
[[gnu::noinline]]
void
assign(A&& a, B&& b) {
  a = b;
}

TEST(TensorTest, ArrayIndexing) {
  ttl::Tensor<3, double, 3> A;
  init(A);
  check(A);
}


TEST(TensorTest, Delta) {
  // D0 and D1 are compile errors.
  // auto D0 = ttl::Delta<0, double, 1>();
  // auto D1 = ttl::Delta<1, double, 6>();
  auto D2 = ttl::Delta<2, double, 6>();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_EQ(D2[i*6 + j], (i == j) ? 1.0 : 0.0);
    }
  }

  auto D3 = ttl::Delta<3, double, 4>(3.14);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        int index = i*4*4 + j*4 + k;
        EXPECT_EQ(D3[index], (i == j && j == k) ? 3.14 : 0.0);
      }
    }
  }

  auto D4 = ttl::Delta<4, int, 3>(42);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int index = i*3*3*3 + j*3*3 + k*3 + l;
          EXPECT_EQ(D4[index], (i == j && j == k && k == l) ? 42 : 0.0);
        }
      }
    }
  }
}

TEST(TensorTest, TensorExprAssignment) {
  ttl::Tensor<1, double, 1> a, b;
  a[0] = 10;
  assign(b(i), a(i));
  EXPECT_EQ(a[0], b[0]);
  // assign(b(i), a(j));

  ttl::Tensor<2, double, 3> T, U;
  assign(T(i, j), U(j, i));
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(T[n*3 + m], U[m*3 + n]);
    }
  }

  // assign(T(i, j), U(j, k));

  ttl::Tensor<3, double, 3> A, B, C;
  init(A);
  assign(B(i,j,k), A(i,j,k));
  check(B);

  assign(B(j,i,k), A(i,j,k));
  assign(C(i,j,k), B(j,i,k));
  check(C);
}

TEST(TensorTest, UnaryOp) {
  ttl::Tensor<2, int, 2> A(1), B;
  B(i, j) = -A(i, j);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(B[i*2 + j], -1);
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

