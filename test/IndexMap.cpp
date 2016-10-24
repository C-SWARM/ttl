#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

static ttl::Tensor<2,2,int> B = {0, 1, 2, 3};

constexpr int index(int D, int i, int j) {
  return i * D + j;
}

TEST(IndexMap, Identity) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j).to(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

TEST(IndexMap, Transpose) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(j,i).to(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

TEST(IndexMap, RValue) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = ttl::Tensor<2,2,int>{0, 1, 2, 3}(j,i).to(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

constexpr int index(int D, int i, int j, int k) {
  return i * D * D + j * D + k;
}

TEST(IndexMap, Rotation) {
  ttl::Tensor<3,2,const int> B = { 0, 1,
                                   2, 3,
                                   4, 5,
                                   6, 7};
  ttl::Tensor<3,2,int> A;
  A(i,j,k) = B(j,k,i).to(i,j,k);
  EXPECT_EQ(A[index(2,0,0,0)], B[index(2,0,0,0)]);
  EXPECT_EQ(A[index(2,0,0,1)], B[index(2,0,1,0)]);
  EXPECT_EQ(A[index(2,0,1,0)], B[index(2,1,0,0)]);
  EXPECT_EQ(A[index(2,1,0,0)], B[index(2,0,0,1)]);
  EXPECT_EQ(A[index(2,0,1,1)], B[index(2,1,1,0)]);
  EXPECT_EQ(A[index(2,1,1,0)], B[index(2,1,0,1)]);
  EXPECT_EQ(A[index(2,1,0,1)], B[index(2,0,1,1)]);
  EXPECT_EQ(A[index(2,1,1,1)], B[index(2,1,1,1)]);
}
