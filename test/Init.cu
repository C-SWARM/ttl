#include <ttl/ttl.h>
#include <gtest/gtest.h>

__global__
void test_init() {
  ttl::Tensor<2,2,double> A;
  ttl::Tensor<2,2,double> B = { 1.0, 2.0, 3.0, 4.0 };

  A=B;
}

TEST(Tensor, Init) {
  test_init<<<1, 1>>>();
}
