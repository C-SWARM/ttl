#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;
static constexpr ttl::Index<'m'> m;

TEST(TensorProduct, Simple) {
  ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C;
  C(i,j) = A(i,k) * B(k,j);
  EXPECT_EQ(C[0], 10);
  EXPECT_EQ(C[0], 13);
  EXPECT_EQ(C[0], 22);
  EXPECT_EQ(C[0], 29);
}

TEST(TensorProduct, Lazy) {
  ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C;
  auto t0 = A(i,k);
  auto t1 = B(k,j);
  C(i,j) = t0 * t1;
  EXPECT_EQ(C[0], 10);
  EXPECT_EQ(C[0], 13);
  EXPECT_EQ(C[0], 22);
  EXPECT_EQ(C[0], 29);
}

TEST(TensorProduct, ComplexLazy) {
  ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C = {3, 4, 5, 6};
  ttl::Tensor<2,2,int> D = {4, 5, 6, 7};
  ttl::Tensor<2,2,int> E;
  auto t0 = A(i,j) * B(j,k);
  auto t1 = C(k,l) * D(l,m);
  E(i,m) = t0 * t1;
}
