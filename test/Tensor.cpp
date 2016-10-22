#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr double E = 2.72;
static constexpr double PI = 3.14;

TEST(TensorTest, ArrayIndexing) {
  ttl::Tensor<3, 3, double> A;
  for (int i = 0; i < 27; ++i) {
    A[i] = i;
  }

  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(A[i], i);
  }
}

TEST(TensorTest, Tensor) {
  ttl::Tensor<1, 1, double> A{};
  EXPECT_EQ(A[0], 0.0);

  ttl::Tensor<1, 1, double> B{E};
  EXPECT_EQ(B[0], E);

  ttl::Tensor<1, 3, double> C{E, PI};
  EXPECT_EQ(C[0], E);
  EXPECT_EQ(C[1], PI);
  EXPECT_EQ(C[2], 0.0);

  ttl::Tensor<2, 3, int> I = {1, 0, 0,
                              0, 1, 0,
                              0, 0, 1};
  EXPECT_EQ(I[0], 1);
  EXPECT_EQ(I[1], 0);
  EXPECT_EQ(I[2], 0);
  EXPECT_EQ(I[3], 0);
  EXPECT_EQ(I[4], 1);
  EXPECT_EQ(I[5], 0);
  EXPECT_EQ(I[6], 0);
  EXPECT_EQ(I[7], 0);
  EXPECT_EQ(I[8], 1);
}

TEST(TensorTest, TensorInitializer) {
  ttl::Tensor<1, 3, double> A = {E, PI};
  EXPECT_EQ(A[0], E);
  EXPECT_EQ(A[1], PI);
  EXPECT_EQ(A[2], 0.0);
}

TEST(TensorTest, TensorWiden) {
  ttl::Tensor<1, 3, double> A{0, 1, 2};
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
  EXPECT_EQ(A[2], 2.0);
}

TEST(TensorTest, TensorWidenInitializer) {
  ttl::Tensor<1, 3, double> A = {0, 1, 2};
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
  EXPECT_EQ(A[2], 2.0);
}

TEST(TensorTest, ConstTensor) {
  const ttl::Tensor<1, 1, double> A{};
  EXPECT_EQ(A[0], 0.0);

  const ttl::Tensor<1, 1, double> B{E};
  EXPECT_EQ(B[0], E);

  const ttl::Tensor<1, 3, double> C{E, PI};
  EXPECT_EQ(C[0], E);
  EXPECT_EQ(C[1], PI);
  EXPECT_EQ(C[2], 0.0);
}

TEST(TensorTest, TensorConst) {
  ttl::Tensor<1, 1, const double> A{};
  EXPECT_EQ(A[0], 0.0);

  ttl::Tensor<1, 1, const double> B{E};
  EXPECT_EQ(B[0], E);

  ttl::Tensor<1, 3, const double> C{E, PI};
  EXPECT_EQ(C[0], E);
  EXPECT_EQ(C[1], PI);
  EXPECT_EQ(C[2], 0.0);
}

TEST(TensorTest, TensorPointer) {
  double a[] = {E};
  double b[] = {E, PI};

  ttl::Tensor<1, 1, double*> A{a};
  EXPECT_EQ(A[0], a[0]);

  ttl::Tensor<1, 2, double*> B{b};
  EXPECT_EQ(B[0], b[0]);
  EXPECT_EQ(B[1], b[1]);
}

TEST(TensorTest, TensorConstPointer) {
  const double a[] = {E};
  const double b[] = {E, PI};

  ttl::Tensor<1, 1, const double*> A{a};
  EXPECT_EQ(A[0], a[0]);

  ttl::Tensor<1, 2, const double*> B{b};
  EXPECT_EQ(B[0], b[0]);
  EXPECT_EQ(B[1], b[1]);
}

TEST(TensorTest, Assign) {
  ttl::Tensor<2, 2, double> I{1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], I[0]);
  EXPECT_EQ(A[1], I[1]);
  EXPECT_EQ(A[2], I[2]);
  EXPECT_EQ(A[3], I[3]);
}

TEST(TensorTest, AssignWiden) {
  ttl::Tensor<2, 2, int> I{1, 0, 0, 1};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], double(I[0]));
  EXPECT_EQ(A[1], double(I[1]));
  EXPECT_EQ(A[2], double(I[2]));
  EXPECT_EQ(A[3], double(I[3]));
}

TEST(TensorTest, AssignToConst) {
  ttl::Tensor<2, 2, int> I{1, 0, 0, 1};
  ttl::Tensor<2, 2, const int> A = I;
  EXPECT_EQ(A[0], double(I[0]));
  EXPECT_EQ(A[1], double(I[1]));
  EXPECT_EQ(A[2], double(I[2]));
  EXPECT_EQ(A[3], double(I[3]));
}

TEST(TensorTest, AssignInitializer) {
  ttl::Tensor<2, 2, double> A;
  A = {1.0, 0.0, 0.0, 1.0};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 0.0);
  EXPECT_EQ(A[2], 0.0);
  EXPECT_EQ(A[3], 1.0);
}

TEST(TensorTest, AssignInitializerWiden) {
  ttl::Tensor<2, 2, double> A;
  A = {1, 0, 0, 1};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 0.0);
  EXPECT_EQ(A[2], 0.0);
  EXPECT_EQ(A[3], 1.0);
}

TEST(TensorTest, AssignConst) {
  const ttl::Tensor<2, 2, double> I{1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], I[0]);
  EXPECT_EQ(A[1], I[1]);
  EXPECT_EQ(A[2], I[2]);
  EXPECT_EQ(A[3], I[3]);
}

TEST(TensorTest, AssignExternal) {
  double i[] = {1.0, 0.0, 0.0, 1.0};
  double a[4];
  ttl::Tensor<2, 2, double*> I{i}, A{a};
  A = I;
  EXPECT_EQ(a[0], i[0]);
  EXPECT_EQ(a[1], i[1]);
  EXPECT_EQ(a[2], i[2]);
  EXPECT_EQ(a[3], i[3]);
}

TEST(TensorTest, AssignExternInitializer) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  A = {1.0, 0.0, 0.0, 1.0};
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 0.0);
  EXPECT_EQ(a[2], 0.0);
  EXPECT_EQ(a[3], 1.0);
}

TEST(TensorTest, AssignExternInitializerWiden) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  A = {1, 0, 0, 1};
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 0.0);
  EXPECT_EQ(a[2], 0.0);
  EXPECT_EQ(a[3], 1.0);
}

TEST(TensorTest, AssignConstExternal) {
  const double i[] = {1.0, 0.0, 0.0, 1.0};
  double a[4];
  const ttl::Tensor<2, 2, const double*> I{i};
  ttl::Tensor<2, 2, double*> A{a};
  A = I;
  EXPECT_EQ(a[0], i[0]);
  EXPECT_EQ(a[1], i[1]);
  EXPECT_EQ(a[2], i[2]);
  EXPECT_EQ(a[3], i[3]);
}

TEST(TensorTest, AssignFromExternal) {
  double i[] = {1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double*> I{i};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], i[0]);
  EXPECT_EQ(A[1], i[1]);
  EXPECT_EQ(A[2], i[2]);
  EXPECT_EQ(A[3], i[3]);
}

TEST(TensorTest, AssignFromConstExternal) {
  const double i[] = {1.0, 0.0, 0.0, 1.0};
  const ttl::Tensor<2, 2, const double*> I{i};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], i[0]);
  EXPECT_EQ(A[1], i[1]);
  EXPECT_EQ(A[2], i[2]);
  EXPECT_EQ(A[3], i[3]);
}

TEST(TensorTest, AssignToExternal) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  ttl::Tensor<2, 2, double> I = {1.0, 0.0,
                                 0.0, 1.0};
  A = I;
  EXPECT_EQ(a[0], I[0]);
  EXPECT_EQ(a[1], I[1]);
  EXPECT_EQ(a[2], I[2]);
  EXPECT_EQ(a[3], I[3]);
}

TEST(TensorTest, AssignConstToExternal) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  const ttl::Tensor<2, 2, const double> I = {1.0, 0.0,
                                             0.0, 1.0};
  A = I;
  EXPECT_EQ(a[0], I[0]);
  EXPECT_EQ(a[1], I[1]);
  EXPECT_EQ(a[2], I[2]);
  EXPECT_EQ(a[3], I[3]);
}

TEST(TensorTest, AssignFromRValue) {
  ttl::Tensor<3, 3, double> T;
  T = ttl::Tensor<3, 3, double>{};
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(T[i], 0.0);
  }
}

TEST(TensorTest, AssignExternalFromRValue) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A = ttl::Tensor<3, 3, double>{};
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], 0.0);
  }
}

TEST(TensorTest, Fill) {
  ttl::Tensor<3, 3, double> A;
  A.fill(E);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(A[i], E);
  }
}

TEST(TensorTest, FillWiden) {
  ttl::Tensor<3, 3, double> A;
  A.fill(1u);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(A[i], 1.0);
  }
}

TEST(TensorTest, FillExternal) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A.fill(E);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], E);
  }
}

TEST(TensorTest, FillExternalWiden) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A.fill(1u);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], 1.0);
  }
}

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

TEST(TensorTest, TensorExprAssignment) {
  ttl::Tensor<1, 1, double> a, b;
  a[0] = 10;
  b(i) = a(i);
  EXPECT_EQ(a[0], b[0]);
  // assign(b(i), a(j));

  ttl::Tensor<2, 3, double> T, U;
  U.fill(3.14);
  T(i, j) = U(i, j);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(T[index(3,n,m)], U[index(3,n,m)]);
    }
  }

  U[index(3,1,0)] = 42;
  T(i,j) = U(j,i);
  EXPECT_EQ(T[index(3,0,1)], 42);

  ttl::Tensor<3, 3, double> A, B;
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
  ttl::Tensor<2, 2, int> A, B;
  A.fill(1);
  B(i, j) = -A(i, j);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(B[index(2,i,j)], -1);
    }
  }
}

TEST(TensorTest, BinaryOp) {
  ttl::Tensor<1, 4, double> x, y, z;
  x.fill(1);
  y.fill(2);

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
  ttl::Tensor<0, 1, double> x, y;
  x.fill(1);

  y() = 3.1 * x();
  EXPECT_EQ(y[0], 3.1);

  x() = y() * 3.0;
  EXPECT_EQ(x[0], 9.3);

  ttl::Tensor<2, 2, int> s = ttl::Delta<2,2,int>(2);
  ttl::Tensor<2, 2, int> t;
  ttl::Tensor<2, 2, double> u;

  t(i,j) = (ttl::Tensor<2,2,int>{}.fill(2)(i,j) - s(j,i))/2;
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
  ttl::Tensor<2,2,double> A, B, C;
  A.fill(3);
  B.fill(2);

  ttl::Tensor<0,2,double> s;
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

  ttl::Tensor<4,2,double> D;

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

  ttl::Tensor<1,2,double> x, y;
  x[0] = 2;
  x[1] = 3;

  auto E = ttl::Delta<2,2,double>();
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

TEST(TensorTest, ExternalStorage) {
  double storage[12];
  ttl::Tensor<2,2,double*> A{&storage[0]}, B{&storage[4]}, C{&storage[8]};
  B.fill(1);
  C.fill(2);

  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(B[index(2,n,m)], 1);
      EXPECT_EQ(C[index(2,n,m)], 2);
    }
  }

  A(i,j) = B(i,j) + C(j,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], 3);
    }
  }

  ttl::Tensor<1,2,double*> x(&storage[0]), y(&storage[4]);
  x(i) = C(i,j) * y(j);
  EXPECT_EQ(A[0], 4);
  EXPECT_EQ(A[1], 4);

  A(i,j) = B(i,j);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], B[index(2,n,m)]);
    }
  }

  ttl::Tensor<2,2,double> D;
  D.fill(3.14);
  A(i,j) = D(i,j);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], D[index(2,n,m)]);
    }
  }

  D(i,j) = B(j,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(D[index(2,n,m)], B[index(2,n,m)]);
    }
  }

  A(i,k) = C(i,j) * D(j,k);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], 4);
    }
  }
}

#if 0

#endif
