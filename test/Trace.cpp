#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Trace, Simple) {
  static constexpr ttl::Index<'i'> i;

  ttl::Tensor<2,2,int> A = {1, 0,
                            0, 1};
  auto t = A(i,i);
  EXPECT_EQ(t, 2);
}
