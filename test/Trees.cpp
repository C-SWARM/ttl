#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;
static constexpr ttl::Index<'m'> m;

TEST(Trees, LazyTensorProduct) {
  ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C;
  auto t0 = A(i,k);
  auto t1 = B(k,j);
  C(i,j) = t0 * t1;
  EXPECT_EQ(C.get(0), 10);
  EXPECT_EQ(C.get(1), 13);
  EXPECT_EQ(C.get(2), 22);
  EXPECT_EQ(C.get(3), 29);
}

TEST(Trees, LazyTensorProductComplex) {
  ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C = {3, 4, 5, 6};
  ttl::Tensor<2,2,int> D = {4, 5, 6, 7};
  ttl::Tensor<2,2,int> E;
  auto t0 = A(i,j) * B(j,k);
  auto t1 = C(k,l) * D(l,m);
  E(i,m) = t0 * t1;
  EXPECT_EQ(E.get(0), 1088);
  EXPECT_EQ(E.get(1), 1301);
  EXPECT_EQ(E.get(2), 2416);
  EXPECT_EQ(E.get(3), 2889);
}
