#include <ttl/ttl.h>
#include <gtest/gtest.h>

using namespace ttl;

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

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'k'> k;
static constexpr Index<'l'> l;

__global__
void test_dim_inf(double *m1, double *m2, double *m3, double *m4, double *m5, double *m6, int *n){
  Tensor<2,2,double*> M1 = {m1},
                      M2 = {m2},
                      M3 = {m3},
                      M4 = {m4},
                      M5 = {m5},
                      M6 = {m6};

  M1 = identity(i,j);
  
  M2 = identity(i,j)/3.;

  M3 = 3.*identity(i,j);

  M4 = identity(i,j)*identity<2>(j,k);           // contraction needs dimension

  M5 = identity(i,j) + identity(i,j);

  M6 = identity(i,j).to(j,i);

  Tensor<4,2,int*> N {n};
  N = identity(i,j)*identity(k,l);
  
}

TEST(Identity, DimensionalityInference){
// Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

// instantiate data pointers
  double *m1,*m2,*m3,*m4,*m5,*m6;
  int *n;

// initialize memory on device and host
  cudaMallocManaged(&m1, 4*sizeof(double));
  cudaMallocManaged(&m2, 4*sizeof(double));
  cudaMallocManaged(&m3, 4*sizeof(double));
  cudaMallocManaged(&m4, 4*sizeof(double));
  cudaMallocManaged(&m5, 4*sizeof(double));
  cudaMallocManaged(&m6, 4*sizeof(double));
  cudaMallocManaged(&n, 16*sizeof(int));

// instantiate cpu side tensor
  Tensor<2,2,double*> M1 = {m1},
                      M2 = {m2},
                      M3 = {m3},
                      M4 = {m4},
                      M5 = {m5},
                      M6 = {m6};

  Tensor<4,2,int*> N {n};

// launch kernel
  test_dim_inf<<<1,1>>>(m1,m2,m3,m4,m5,m6,n);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify
  std::cout << "M1\n" << M1(i,j);
  std::cout << "M2\n" << M2(i,j);
  std::cout << "M3\n" << M3(i,j);
  std::cout << "M4\n" << M4(i,j);
  std::cout << "M5\n" << M5(i,j);
  std::cout << "M6\n" << M6(i,j);

  std::cout << "N\n" << N(i,j,k,l);

// garbage
  cudaFree(m1);
  cudaFree(m2);
  cudaFree(m3);
  cudaFree(m4);
  cudaFree(m5);
  cudaFree(m6);
  cudaFree(n);
}

__global__
void test_scalar(double *a, double *b) {
    double scalar = 3.14;
    *a = scalar;

    auto d = identity();
    *b = d;
}

TEST(Identity, Scalar) {
// Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

// instantiate data pointers
  double *scalar;
  double *d;

// initialize memory on device and host
  cudaMallocManaged(&scalar, 1*sizeof(double));
  cudaMallocManaged(&d, 1*sizeof(double));

// launch kernel
  test_scalar<<<1,1>>>(scalar,d);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_EQ(*d * *scalar, *scalar);
  EXPECT_EQ(*d * identity(), *d);

// garbage  
  cudaFree(scalar);
  cudaFree(d);
}

__global__
void test_vector(double *v, double *u) {
  Tensor<1,2,double*> V {v,{1,2}};
  Tensor<1,2,double*> U {u};
  U = identity(i,j)*V(j);
}

TEST(Identity, Vector) {
// Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

// instantiate data pointers
  double *v;
  double *u;

// initialize memory on device and host
  cudaMallocManaged(&v, 1*sizeof(double));
  cudaMallocManaged(&u, 1*sizeof(double));

// instantiate cpu side tensor
  Tensor<1, 2, double*> V {v};
  Tensor<1, 2, double*> U {u};

// launch kernel
  test_vector<<<1,1>>>(v,u);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  EXPECT_EQ(U(0), V(0));
  EXPECT_EQ(U(1), V(1));

// garbage
  cudaFree(u);
  cudaFree(v);
}

__global__
void test_matrix(int *a_h,int *b_h,int *c_h,int *d_h,int *e_h,int *f_h,int *g_h,int *j_h,int *id_h){
  Tensor<4,3,int> ID = identity(i,j,k,l), J;
  J(i,j,k,l) = identity(i,j,k,l);
  
  //  host side verifiers
  Tensor<4,3,int*> ID_h {id_h}, J_h {j_h};
  ID_h = ID;
  J_h = J;

  // device side
  Tensor<2,3,int> A = {1,2,3,4,5,6,7,8,9},
                  B = ID(i,j,k,l)*A(k,l),
                  C = J(i,j,k,l)*A(k,l),
                  D = identity(i,j,k,l)*A(k,l),
                      E, F, G;

  E(i,j) = ID(i,j,k,l)*A(k,l);
  F(i,j) = J(i,j,k,l)*A(k,l);
  G(i,j) = identity(i,j,k,l)*A(k,l);

  //  host side verifiers 
  Tensor<2,3,int*> A_h = {a_h}, 
                   B_h = {b_h},
                   C_h = {c_h},
                   D_h = {d_h},
                   E_h = {e_h},
                   F_h = {f_h},
                   G_h = {g_h};

  A_h = A;
  B_h = B;
  C_h = C;
  D_h = D;
  E_h = E;
  F_h = F;
  G_h = G;
}
  

TEST(Identity, Matrix) {
// Check for available device
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(nDevices > 0);

// instantiate data pointers
  int *a;
  int *b;
  int *c;
  int *d;
  int *e;
  int *f;
  int *g;
  int *j;
  int *id;

// initialize memory on device and host
  cudaMallocManaged(&a, 9*sizeof(int));
  cudaMallocManaged(&b, 9*sizeof(int));
  cudaMallocManaged(&c, 9*sizeof(int));
  cudaMallocManaged(&d, 9*sizeof(int));
  cudaMallocManaged(&e, 9*sizeof(int));
  cudaMallocManaged(&f, 9*sizeof(int));
  cudaMallocManaged(&g, 9*sizeof(int));
  cudaMallocManaged(&j, 81*sizeof(int));
  cudaMallocManaged(&id, 81*sizeof(int));

// instantiate cpu side tensor
  Tensor<2, 3, int*> A {a};
  Tensor<2, 3, int*> B {b};
  Tensor<2, 3, int*> C {c};
  Tensor<2, 3, int*> D {d};
  Tensor<2, 3, int*> E {e};
  Tensor<2, 3, int*> F {f};
  Tensor<2, 3, int*> G {g};

  Tensor<4, 3, int*> J {j};
  Tensor<4, 3, int*> ID {id};
// launch kernel
  test_matrix<<<1,1>>>(a,b,c,d,e,f,g,j,id);

// control for race conditions
  cudaDeviceSynchronize();

// check errors
  cudaCheckErrors("failed");

// verify results
  for (int m = 0; m < 3; ++m) {
    for (int n = 0; n < 3; ++n) {
      for (int o = 0; o < 3; ++o) {
        for (int p = 0; p < 3; ++p) {
          EXPECT_EQ(ID(m,n,o,p), J(m,n,o,p));
        }
      }
      EXPECT_EQ(ID(m,n,m,n), 1);
      EXPECT_EQ(B(m,n), A(m,n));
      EXPECT_EQ(C(m,n), A(m,n));
      EXPECT_EQ(D(m,n), A(m,n));
      EXPECT_EQ(E(m,n), A(m,n));
      EXPECT_EQ(F(m,n), A(m,n));
      EXPECT_EQ(G(m,n), A(m,n));
    }
  }

// garbage
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(d);
  cudaFree(e);
  cudaFree(f);
  cudaFree(g);
  cudaFree(j);
  cudaFree(id);
}
