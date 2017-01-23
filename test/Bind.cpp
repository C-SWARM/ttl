#include <ttl/ttl.h>
#include <gtest/gtest.h>

const int e[] = {0,1,2,3,4,5,6,7};
static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

static const ttl::Tensor<2,2,int> B = {0,1,2,3};
static const ttl::Tensor<2,2,const int> C = {0,1,2,3};
static const ttl::Tensor<2,2,const int*> E(e);

TEST(Bind, InitializeRValue) {
  ttl::Tensor<2,2,int> A = 2 * B(i,j);
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));
}

TEST(Bind, InitializeLValue) {
  auto e = 2 * B(i,j);
  ttl::Tensor<2,2,int> A = e;
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));
}

TEST(Bind, Assign) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(i,j);
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(1));
  EXPECT_EQ(B.get(2), A.get(2));
  EXPECT_EQ(B.get(3), A.get(3));
}

TEST(Bind, AssignRValueExpression) {
  ttl::Tensor<2,2,int> A;
  A = 2 * B(i,j);
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));
}

TEST(Bind, AssignLValueExpression) {
  auto b = 2 * B(i,j);
  ttl::Tensor<2,2,int> A;
  A = b;
  EXPECT_EQ(2 * B.get(0), A.get(0));
  EXPECT_EQ(2 * B.get(1), A.get(1));
  EXPECT_EQ(2 * B.get(2), A.get(2));
  EXPECT_EQ(2 * B.get(3), A.get(3));
}

TEST(Bind, Accumulate) {
  ttl::Tensor<2,2,int> A = {};
  A(i,j) += B(i,j);
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(1));
  EXPECT_EQ(B.get(2), A.get(2));
  EXPECT_EQ(B.get(3), A.get(3));
}

TEST(Bind, AssignFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(i,j);
  EXPECT_EQ(C.get(0), A.get(0));
  EXPECT_EQ(C.get(1), A.get(1));
  EXPECT_EQ(C.get(2), A.get(2));
  EXPECT_EQ(C.get(3), A.get(3));
}

TEST(Bind, AssignFromExternal) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = E(i,j);
  EXPECT_EQ(e[0], A.get(0));
  EXPECT_EQ(e[1], A.get(1));
  EXPECT_EQ(e[2], A.get(2));
  EXPECT_EQ(e[3], A.get(3));
}

TEST(Bind, AssignToExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = B(i,j);
  EXPECT_EQ(a[0], B.get(0));
  EXPECT_EQ(a[1], B.get(1));
  EXPECT_EQ(a[2], B.get(2));
  EXPECT_EQ(a[3], B.get(3));
}

TEST(Bind, AssignExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = E(i,j);
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[1]);
  EXPECT_EQ(a[2], e[2]);
  EXPECT_EQ(a[3], e[3]);
}

TEST(Bind, AssignPermute) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = B(j,i);
  EXPECT_EQ(B.get(0), A.get(0));
  EXPECT_EQ(B.get(1), A.get(2));
  EXPECT_EQ(B.get(2), A.get(1));
  EXPECT_EQ(B.get(3), A.get(3));
}

TEST(Bind, AssignPermuteFromConst) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = C(j,i);
  EXPECT_EQ(C.get(0), A.get(0));
  EXPECT_EQ(C.get(1), A.get(2));
  EXPECT_EQ(C.get(2), A.get(1));
  EXPECT_EQ(C.get(3), A.get(3));
}

TEST(Bind, AssignPermuteFromExternal) {
  ttl::Tensor<2,2,int> A;
  A(i,j) = E(j,i);
  EXPECT_EQ(e[0], A.get(0));
  EXPECT_EQ(e[1], A.get(2));
  EXPECT_EQ(e[2], A.get(1));
  EXPECT_EQ(e[3], A.get(3));
}

TEST(Bind, AssignPermuteToExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = B(j,i);
  EXPECT_EQ(a[0], B.get(0));
  EXPECT_EQ(a[1], B.get(2));
  EXPECT_EQ(a[2], B.get(1));
  EXPECT_EQ(a[3], B.get(3));
}

TEST(Bind, AssignPermuteExternal) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A(i,j) = E(j,i);
  EXPECT_EQ(a[0], e[0]);
  EXPECT_EQ(a[1], e[2]);
  EXPECT_EQ(a[2], e[1]);
  EXPECT_EQ(a[3], e[3]);
}

TEST(Bind, ExternalInitializeRValue) {
  int a[4];
  const ttl::Tensor<2,2,int*> A = {a, 2 * B(i,j)};
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);
}

TEST(Bind, ExternalInitializeLValue) {
  auto e = 2 * B(i,j);
  int a[4];
  const ttl::Tensor<2,2,int*> A = {a, e};
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);
}

TEST(Bind, ExternalAssignRValueExpression) {
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A = 2 * B(i,j);
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);
}

TEST(Bind, ExternalAssignLValueExpression) {
  auto e = 2 * B(i,j);
  int a[4];
  ttl::Tensor<2,2,int*> A(a);
  A = e;
  EXPECT_EQ(2 * B.get(0), a[0]);
  EXPECT_EQ(2 * B.get(1), a[1]);
  EXPECT_EQ(2 * B.get(2), a[2]);
  EXPECT_EQ(2 * B.get(3), a[3]);
}

TEST(Bind, Trace2x2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  auto t = A(i,i);
  EXPECT_EQ(t, 5);
}

TEST(Bind, Trace2x3) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8,9};
  auto t = A(i,i);
  EXPECT_EQ(t, 15);
}

TEST(Bind, Trace3x2) {
  ttl::Tensor<3,2,int> A = {1,2,3,4,5,6,7,8};
  auto t = A(i,i,i);
  EXPECT_EQ(t, A.get(0) + A.get(7));
}

TEST(Bind, ParallelContract) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  auto t = A(i,i,j,j);
  EXPECT_EQ(t, A.get(0) + A.get(3) + A.get(12) + A.get(15));
}

TEST(Bind, SequentialContract) {
  ttl::Tensor<4,2,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  ttl::Tensor<2,2,int> B = A(i,i,j,k);
  auto t = B(j,j);
  EXPECT_EQ(t, B.get(0) + B.get(3));
}
