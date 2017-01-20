#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Trace, Simple) {
  static constexpr ttl::Index<'i'> i;

  ttl::Tensor<4,2,int> A;
  auto t = A(i,i,i,i);
  // EXPECT_EQ(t, 2);
}
