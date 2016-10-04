#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(TensorTest, ArrayIndexing) {
  ttl::Tensor<3, double, 3> A;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; j < 3; ++j) {
        A[{i,j,k}] = i * 3 * 3 + j * 3 + k;
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; j < 3; ++j) {
        EXPECT_EQ(A[i * 3 * 3 + j * 3 + k], i * 3 * 3 + j * 3 + k);
      }
    }
  }
}

