#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr double E = 2.72;
static constexpr double PI = 3.14;

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;

TEST(UnaryOp, Negate) {
  ttl::Tensor<2, 2, int> A, B = {0,1,2,3};
  A(i,j) = -B(i,j);
  EXPECT_EQ(A.get(0), -B.get(0));
  EXPECT_EQ(A.get(1), -B.get(1));
  EXPECT_EQ(A.get(2), -B.get(2));
  EXPECT_EQ(A.get(3), -B.get(3));
}

TEST(ScalarOp, MultiplyRHS) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = PI * B(i,j);
  EXPECT_EQ(A.get(0), PI * E);
  EXPECT_EQ(A.get(1), PI * E);
  EXPECT_EQ(A.get(2), PI * E);
  EXPECT_EQ(A.get(3), PI * E);
}

TEST(ScalarOp, MultiplyLHS) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = B(i,j) * PI;
  EXPECT_EQ(A.get(0), PI * E);
  EXPECT_EQ(A.get(1), PI * E);
  EXPECT_EQ(A.get(2), PI * E);
  EXPECT_EQ(A.get(3), PI * E);
}

TEST(ScalarOp, Divide) {
  ttl::Tensor<2, 2, double> A, B;
  B.fill(E);
  A(i,j) = B(i,j) / PI;
  EXPECT_EQ(A.get(0), E / PI);
  EXPECT_EQ(A.get(1), E / PI);
  EXPECT_EQ(A.get(2), E / PI);
  EXPECT_EQ(A.get(3), E / PI);
}

TEST(ScalarOp, Modulo) {
  ttl::Tensor<2, 2, int> A, B = {0,1,2,3};
  A(i,j) = B(i,j) % 3;
  EXPECT_EQ(A.get(0), 0);
  EXPECT_EQ(A.get(1), 1);
  EXPECT_EQ(A.get(2), 2);
  EXPECT_EQ(A.get(3), 0);
}

TEST(BinaryOp, Add) {
  const ttl::Tensor<2, 2, int> A = {0,1,2,3}, B = {1,2,3,4};
  ttl::Tensor<2, 2, int> C;
  C(i,j) = A(i,j) + B(i,j);
  EXPECT_EQ(C.get(0), 1);
  EXPECT_EQ(C.get(1), 3);
  EXPECT_EQ(C.get(2), 5);
  EXPECT_EQ(C.get(3), 7);
}

TEST(BinaryOp, Subtract) {
  const ttl::Tensor<2, 2, int> A = {0,1,2,3}, B = {1,2,3,4};
  ttl::Tensor<2, 2, int> C;
  C(i,j) = A(i,j) - B(i,j);
  EXPECT_EQ(C.get(0), -1);
  EXPECT_EQ(C.get(1), -1);
  EXPECT_EQ(C.get(2), -1);
  EXPECT_EQ(C.get(3), -1);
}

TEST(TensorProduct, Multiply) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,int> C;
  C(i,j) = A(i,k) * B(k,j);
  EXPECT_EQ(C.get(0), 10);
  EXPECT_EQ(C.get(1), 13);
  EXPECT_EQ(C.get(2), 22);
  EXPECT_EQ(C.get(3), 29);
}

TEST(TensorProduct, Inner) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<0,2,int> c;
  c() = A(i,j) * B(i,j);
  EXPECT_EQ(c.get(0), 40);
}

TEST(TensorProduct, Outer) {
  const ttl::Tensor<2,2,int> A = {1, 2, 3, 4};
  const ttl::Tensor<2,2,int> B = {2, 3, 4, 5};
  ttl::Tensor<4,2,int> C;
  C(i,j,k,l) = A(i,j) * B(k,l);
  EXPECT_EQ(C.get(0), 2);
  EXPECT_EQ(C.get(1), 3);
  EXPECT_EQ(C.get(2), 4);
  EXPECT_EQ(C.get(3), 5);
  EXPECT_EQ(C.get(4), 4);
  EXPECT_EQ(C.get(5), 6);
  EXPECT_EQ(C.get(6), 8);
  EXPECT_EQ(C.get(7), 10);
  EXPECT_EQ(C.get(8), 6);
  EXPECT_EQ(C.get(9), 9);
  EXPECT_EQ(C.get(10), 12);
  EXPECT_EQ(C.get(11), 15);
  EXPECT_EQ(C.get(12), 8);
  EXPECT_EQ(C.get(13), 12);
  EXPECT_EQ(C.get(14), 16);
  EXPECT_EQ(C.get(15), 20);
}
