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

TEST(TensorTest, TensorExprAssignment) {
  ttl::Tensor<1, double, 1> a, b;
  a[0] = 10;
  assign(b(i), a(i));
  EXPECT_EQ(a[0], b[0]);

  ttl::Tensor<2, double, 3> T, U;
  assign(T(i, j), U(j, i));


  ttl::Tensor<3, double, 3> A, B, C;
  init(A);
  assign(B(i,j,k), A(i,j,k));
  check(B);

  assign(B(j,i,k), A(i,j,k));
  assign(C(i,j,k), B(j,i,k));
  check(C);
}

TEST(TensorTest, AddExpression) {
  ttl::Tensor<1, double, 4> x(1), y(2), z;

  // z(i) = x(i) + y(i);
}
