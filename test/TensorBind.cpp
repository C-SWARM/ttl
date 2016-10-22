#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;

static constexpr const ttl::Tensor<2,2,int> B = {0, 1, 2, 3};
static constexpr const ttl::Tensor<2,2,const int> C = {0, 1, 2, 3};

const int e[4] = {0,1,2,3};
static ttl::Tensor<2,2,const int*> E{e};

constexpr int index(int D, int i, int j) {
  return i * D + j;
}

TEST(TensorBind, Assign) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

TEST(TensorBind, AssignFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

TEST(TensorBind, AssignFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(i,j);
  EXPECT_EQ(B[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], A[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,0)], A[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,1)], A[index(2,1,1)]);
}

TEST(TensorBind, AssignFromExternal) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = E(i,j);
  EXPECT_EQ(e[index(2,0,0)], A[index(2,0,0)]);
  EXPECT_EQ(e[index(2,0,1)], A[index(2,0,1)]);
  EXPECT_EQ(e[index(2,1,0)], A[index(2,1,0)]);
  EXPECT_EQ(e[index(2,1,1)], A[index(2,1,1)]);
}

TEST(TensorBind, AssignToExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A{a};
  A(i,j) = B(i,j);
  EXPECT_EQ(B[index(2,0,0)], a[index(2,0,0)]);
  EXPECT_EQ(B[index(2,0,1)], a[index(2,0,1)]);
  EXPECT_EQ(B[index(2,1,0)], a[index(2,1,0)]);
  EXPECT_EQ(B[index(2,1,1)], a[index(2,1,1)]);
}

TEST(TensorBind, AssignExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A{a};
  A(i,j) = E(i,j);
  EXPECT_EQ(e[index(2,0,0)], a[index(2,0,0)]);
  EXPECT_EQ(e[index(2,0,1)], a[index(2,0,1)]);
  EXPECT_EQ(e[index(2,1,0)], a[index(2,1,0)]);
  EXPECT_EQ(e[index(2,1,1)], a[index(2,1,1)]);
}

