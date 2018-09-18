#include <gtest/gtest.h>
#include <ttl/ttl.h>

#include <ttl/ttl.h>

#define cudaCheckErrors(msg)                                \
  do {                                                      \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",    \
              msg, cudaGetErrorString(__err),               \
              __FILE__, __LINE__);                          \
      fprintf(stderr, "*** FAILED - ABORTING\n");           \
      exit(1);                                              \
    }                                                       \
  } while (0)

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

__global__
static void k_init(double *a) {
  ttl::Tensor<2,2,double*> A = { a, { 1.0, 2.0, 3.0, 4.0 }};
}

__global__
static void k_init_stack(double *a) {
  ttl::Tensor<2,2,double> A = { 1.0, 2.0, 3.0, 4.0 };
  a[0] = A.data[0];
  a[1] = A.data[1];
  a[2] = A.data[2];
  a[3] = A.data[3];
}

TEST(Tensor, Init) {
  double *a;                                    // allocate device space
  cudaMallocManaged(&a, 4*sizeof(double));
  assert(a);

  k_init<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  cudaCheckErrors("failed");
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  k_init_stack<<<1, 1>>>(a);
  cudaDeviceSynchronize();
      cudaCheckErrors("failed");
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  cudaFree(a);
}

__global__
static void k_copy(double *out) {
  ttl::Tensor<2,2,double> A = { 1.0, 2.0, 3.0, 4.0 };
  ttl::Tensor<2,2,double*> B = { out };
  B = A;
}

__global__
static void k_move(double *out) {
  ttl::Tensor<2,2,double*> A = { out };
  A = ttl::Tensor<2,2,double>{ 1.0, 2.0, 3.0, 4.0 };
}

TEST(Tensor, Copy) {
  double *a;                                    // allocate device space
  cudaMallocManaged(&a, 4*sizeof(double));
  assert(a);

  k_copy<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  cudaCheckErrors("failed");
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  k_move<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  cudaCheckErrors("failed");
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  cudaFree(a);
}

__global__
static void k_index(double *a) {
  ttl::Tensor<2,2,double*> A = { a };
  A[0][0] = 1.0;
  A[0][1] = 2.0;
  A[1][0] = 3.0;
  A[1][1] = 4.0;
}

__global__
static void k_index_2(double *a) {
  ttl::Tensor<2,2,double*> A = { a };
  A(0,0) = 1.0;
  A(0,1) = 2.0;
  A(1,0) = 3.0;
  A(1,1) = 4.0;
}

__global__
static void k_slice(double *a) {
  ttl::Tensor<2,2,double> A = { 1.0, 2.0, 3.0, 4.0 };
  a[0] = A(i,0)(1);
  a[1] = A(i,0)(0);
  a[2] = A(1,i)(1);
  a[3] = A(1,i)(0);
}

TEST(Tensor, Index) {
  double *a;                                    // allocate device space
  cudaMallocManaged(&a, 4*sizeof(double));
  assert(a);

  k_index<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  k_index_2<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 2.0);
  EXPECT_EQ(a[2], 3.0);
  EXPECT_EQ(a[3], 4.0);
  std::fill_n(a, 4, 0.0);

  k_slice<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  EXPECT_EQ(a[0], 3.0);
  EXPECT_EQ(a[1], 1.0);
  EXPECT_EQ(a[2], 4.0);
  EXPECT_EQ(a[3], 3.0);
  std::fill_n(a, 4, 0.0);

  cudaFree(a);
}

__global__
static void k_tensor_product(double *c) {
  ttl::Tensor<2,2,double> A = {1, 2, 3, 4};
  ttl::Tensor<2,2,double> B = {2, 3, 4, 5};
  ttl::Tensor<2,2,double*> C = {c, A(i,k) * B(k,j) };
}

TEST(Trees, TensorProduct) {
  double *out;
  cudaMallocManaged(&out, 4*sizeof(double));
  assert(out);
  k_tensor_product<<<1, 1>>>(out);
  cudaDeviceSynchronize();
  EXPECT_EQ(out[0], 10);
  EXPECT_EQ(out[1], 13);
  EXPECT_EQ(out[2], 22);
  EXPECT_EQ(out[3], 29);
  cudaFree(out);
}
