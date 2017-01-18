#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Tensor, ScalarTensor) {
  ttl::Tensor<0, 2, double> A = {1.2};
  EXPECT_EQ(A[0], 1.2);
}

TEST(Tensor, ScalarExpression) {
  ttl::Tensor<0, 2, double> A = {1.2};
  auto d = A();
  EXPECT_EQ(d, 1.2);
}

TEST(Tensor, ScalarContraction) {
  static constexpr ttl::Index<'i'> i;
  ttl::Tensor<1, 2, int> A = {1, 2}, B = {5, 6};
  auto d = A(i)*B(i);
  EXPECT_EQ(d, 17);
}
