#include <ttl/ttl.h>
#include <gtest/gtest.h>

using namespace ttl;

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'k'> k;
static constexpr Index<'l'> l;

TEST(Identity, DimensionalityInference) {
  Tensor<2,2,double> M;
  M = identity(i,j);
  std::cout << M(i,j);
  M = identity(i,j)/3.;
  std::cout << M(i,j);
  M = 3.*identity(i,j);
  std::cout << M(i,j);
  M = identity(i,j)*identity<2>(j,k);           // contraction needs dimension
  std::cout << M(i,j);
  M = identity(i,j) + identity(i,j);
  std::cout << M(i,j);
  M = identity(i,j).to(j,i);
  std::cout << M(i,j);
  Tensor<4,2,int> N = identity(i,j)*identity(k,l);
  std::cout << N(i,j,k,l);
}

TEST(Identity, Scalar) {
  double scalar = 3.14;
  auto d = identity();
  EXPECT_EQ(d*scalar, scalar);
  EXPECT_EQ(scalar*d, scalar);
  EXPECT_EQ(d*identity(), d);
  EXPECT_EQ(identity()*d, d);
}

TEST(Identity, Vector) {
  Tensor<1,2,double> v = {1,2},
                     u = identity(i,j)*v(j);
  EXPECT_EQ(u(0), v(0));
  EXPECT_EQ(u(1), v(1));
}

TEST(Identity, Matrix) {
  Tensor<4,3,int> ID = identity(i,j,k,l), J;
  J(i,j,k,l) = identity(i,j,k,l);

  std::cout << "I\n" << ID(i,j,k,l) << "\n";


  Tensor<2,3,int> A = {1,2,3,4,5,6,7,8,9},
                  B = ID(i,j,k,l)*A(k,l),
                  C = J(i,j,k,l)*A(k,l),
                  D = identity(i,j,k,l)*A(k,l),
                      E, F, G;

  E(i,j) = ID(i,j,k,l)*A(k,l);
  F(i,j) = J(i,j,k,l)*A(k,l);
  G(i,j) = identity(i,j,k,l)*A(k,l);

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
}
