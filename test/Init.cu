#include <ttl/ttl.h>
#include <gtest/gtest.h>

__global__
void test_init() {
  ttl::Tensor<2,2,double> A;
  ttl::Tensor<2,2,double> B = { 1.0, 2.0, 3.0, 4.0 };
  A=B;
}


__global__
void test_2(double *a, double *b) {
  ttl::Tensor<2,2,double*> A = { a };
  ttl::Tensor<2,2,double*> B = { b, { 1.0, 2.0, 3.0, 4.0 }};
  A=B;
}

TEST(Tensor, Init) {
  test_init<<<1, 1>>>();

  double *a, *b;
  cudaMalloc(&a, 4*sizeof(double));
  cudaMalloc(&b, 4*sizeof(double));
  test_2<<<1, 1>>>(a, b);
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);

  EXPECT_EQ(b[0], 1.0);
  EXPECT_EQ(b[1], 2.0);
  EXPECT_EQ(b[2], 3.0);
  EXPECT_EQ(b[3], 4.0);
  cudaFree(a);
  cudaFree(b);
}
