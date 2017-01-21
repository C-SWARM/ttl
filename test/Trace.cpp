#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

TEST(Trace, Simple) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  auto t = A(i,i);
  EXPECT_EQ(t, 5);
}

TEST(Trace, Moderate) {
  ttl::Tensor<3,2,int> A = {1,2,3,4,5,6,7,8};
  auto t = A(i,i,i);
  EXPECT_EQ(t, A[0] + A[7]);
}

TEST(Trace, Complex) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int t = A(i,i,j,j);
  EXPECT_EQ(t, A[0] + A[3] + A[12] + A[15]);

  ttl::Tensor<2,2,int> B = A(i,i,j,k);
  int u = B(j,j);
  EXPECT_EQ(u, B[0] + B[3]);
}
