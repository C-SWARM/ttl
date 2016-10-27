#include <ttl/ttl.h>
#include <gtest/gtest.h>

const ttl::Index<'i'> i;
const ttl::Index<'j'> j;

TEST(Determinant, Basic_2_2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  int d = ttl::det(A);
  EXPECT_EQ(d, -2);
}

TEST(Determinant, External_2_2) {
  int a[4];
  const ttl::Tensor<2,2,int*> A = {a, {1,2,3,4}};
  int d = ttl::det(A);
  EXPECT_EQ(d, -2);
}

TEST(Determinant, RValue_2_2) {
  int d = ttl::det(ttl::Tensor<2,2,int>{1,2,3,4});
  EXPECT_EQ(d, -2);
}

TEST(Determinant, ExpressionRValue_2_2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  int d = ttl::det(A(i,j));
  EXPECT_EQ(d, -2);
}

TEST(Determinant, Expression_2_2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4};
  auto e = A(i,j);
  int d = ttl::det(e);
  EXPECT_EQ(d, -2);
}

TEST(Determinant, 3_3) {
  ttl::Tensor<2,3,int> A = {1,2,3,
                            4,5,6,
                            7,8,10};
  int d = ttl::det(A);
  EXPECT_EQ(d, -3);
}
