#include <ttl/ttl.h>
#include <gtest/gtest.h>

using namespace ttl;

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'k'> k;
static constexpr Index<'l'> l;
static constexpr Index<'m'> m;
static constexpr Index<'n'> n;

TEST(Identity, Scalar) {
  double scalar = 3.14;
  auto d = identity();
  EXPECT_EQ(d*scalar, scalar);
}

TEST(Identity, Vector) {
  Tensor<1,2,double> v = {1,2},
                     u = identity(i,j)*v(j);
  EXPECT_EQ(u(0), v(0));
  EXPECT_EQ(u(1), v(1));
}

TEST(Identity, Matrix) {
  Tensor<2,3,double> A = {1,2,3,4,5,6,7,8,9},
                     B = identity(i,j,k,l)*A(k,l);
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      EXPECT_EQ(B(m,n), A(m,n));
    }
  }
}
