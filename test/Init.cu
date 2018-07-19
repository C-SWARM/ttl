#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Tensor, Init) {
  ttl::Tensor<2,2,double> A;
  ttl::Tensor<2,2,double> B = { 1.0, 2.0, 3.0, 4.0 };

  A = B;
}
