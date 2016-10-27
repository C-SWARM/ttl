#include <ttl/ttl.h>
#include <gtest/gtest.h>

const int e[] = {0,1,2,3,4,5,6,7};
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static const ttl::Tensor<2,2,int> B = {0, 1, 2, 3};
static const ttl::Tensor<2,2,const int> C = {0, 1, 2, 3};
static const ttl::Tensor<2,2,const int*> E(e);

TEST(TensorBind, InitializeRValue) {
  ttl::Tensor<2,2,int> A = 2 * B(i,j);
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(TensorBind, InitializeLValue) {
  auto e = 2 * B(i,j);
  ttl::Tensor<2,2,int> A = e;
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(TensorBind, Assign) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j);
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(TensorBind, AssignRValueExpression) {
  ttl::Tensor<2,2,int> A;
  A = 2 * B(i,j);
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(TensorBind, AssignLValueExpression) {
  auto b = 2 * B(i,j);
  ttl::Tensor<2,2,int> A;
  A = b;
  EXPECT_EQ(2 * B[0], A[0]);
  EXPECT_EQ(2 * B[1], A[1]);
  EXPECT_EQ(2 * B[2], A[2]);
  EXPECT_EQ(2 * B[3], A[3]);
}

TEST(TensorBind, Accumulate) {
  ttl::Tensor<2,2,int> A = {};
  A(i,j) += B(i,j);
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(TensorBind, AssignFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(i,j);
  EXPECT_EQ(C[0], A[0]);
  EXPECT_EQ(C[1], A[1]);
  EXPECT_EQ(C[2], A[2]);
  EXPECT_EQ(C[3], A[3]);
}

TEST(TensorBind, AssignFromExternal) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = E(i,j);
  EXPECT_EQ(e[0], A[0]);
  EXPECT_EQ(e[1], A[1]);
  EXPECT_EQ(e[2], A[2]);
  EXPECT_EQ(e[3], A[3]);
}

TEST(TensorBind, AssignToExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = B(i,j);
  EXPECT_EQ(a[0], B[0]);
  EXPECT_EQ(a[1], B[1]);
  EXPECT_EQ(a[2], B[2]);
  EXPECT_EQ(a[3], B[3]);
}

TEST(TensorBind, AssignExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = E(i,j);
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[1]);
  EXPECT_EQ(a[2], e[2]);
  EXPECT_EQ(a[3], e[3]);
}

TEST(TensorBind, AssignPermute) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(j,i);
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[2]);
  EXPECT_EQ(B[2], A[1]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(TensorBind, AssignPermuteFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(j,i);
  EXPECT_EQ(C[0], A[0]);
  EXPECT_EQ(C[1], A[2]);
  EXPECT_EQ(C[2], A[1]);
  EXPECT_EQ(C[3], A[3]);
}

TEST(TensorBind, AssignPermuteFromExternal) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = E(j,i);
  EXPECT_EQ(e[0], A[0]);
  EXPECT_EQ(e[1], A[2]);
  EXPECT_EQ(e[2], A[1]);
  EXPECT_EQ(e[3], A[3]);
}

TEST(TensorBind, AssignPermuteToExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = B(j,i);
  EXPECT_EQ(a[0], B[0]);
  EXPECT_EQ(a[1], B[2]);
  EXPECT_EQ(a[2], B[1]);
  EXPECT_EQ(a[3], B[3]);
}

TEST(TensorBind, AssignPermuteExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = E(j,i);
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[2]);
  EXPECT_EQ(a[2], e[1]);
  EXPECT_EQ(a[3], e[3]);
}


TEST(TensorBind, ExternalInitializeRValue) {
  int a[4];
  const ttl::Tensor<2,2,int*> A = {a, 2 * B(i,j)};
  EXPECT_EQ(2 * B[0], a[0]);
  EXPECT_EQ(2 * B[1], a[1]);
  EXPECT_EQ(2 * B[2], a[2]);
  EXPECT_EQ(2 * B[3], a[3]);
}

TEST(TensorBind, ExternalInitializeLValue) {
  auto e = 2 * B(i,j);
  int a[4];
  const ttl::Tensor<2,2,int*> A = {a, e};
  EXPECT_EQ(2 * B[0], a[0]);
  EXPECT_EQ(2 * B[1], a[1]);
  EXPECT_EQ(2 * B[2], a[2]);
  EXPECT_EQ(2 * B[3], a[3]);
}

TEST(TensorBind, ExternalAssignRValueExpression) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A = 2 * B(i,j);
  EXPECT_EQ(2 * B[0], a[0]);
  EXPECT_EQ(2 * B[1], a[1]);
  EXPECT_EQ(2 * B[2], a[2]);
  EXPECT_EQ(2 * B[3], a[3]);
}

TEST(TensorBind, ExternalAssignLValueExpression) {
  auto e = 2 * B(i,j);
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A = e;
  EXPECT_EQ(2 * B[0], a[0]);
  EXPECT_EQ(2 * B[1], a[1]);
  EXPECT_EQ(2 * B[2], a[2]);
  EXPECT_EQ(2 * B[3], a[3]);
}
