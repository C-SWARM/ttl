#include <ttl/ttl.h>
#include <gtest/gtest.h>

__global__
void test_init(double *a) {
  ttl::Tensor<2,2,double*> A = { a, { 1.0, 2.0, 3.0, 4.0 }};
}

TEST(Tensor, Init) {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

  double *a;                                    // allocate device space
  cudaMallocManaged(&a, 4*sizeof(double));
  assert(a);
  a[0] = 0.0;
  a[1] = 0.0;
  a[2] = 0.0;
  a[3] = 0.0;
  test_init<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  cudaFree(a);
}
