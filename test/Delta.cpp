#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Delta, 2_6) {
  auto D2 = ttl::Delta<2, 6, double>();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_EQ(D2[i][j], (i == j) ? 1.0 : 0.0);
    }
  }
}

TEST(Delta, 3_4) {
  auto D3 = ttl::Delta<3, 4, double>(3.14);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(D3[i][j][k], (i == j && j == k) ? 3.14 : 0.0);
      }
    }
  }
}

TEST(Delta, 4_3) {
  auto D4 = ttl::Delta<4, 3, int>(42);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          EXPECT_EQ(D4[i][j][k][l], (i == j && j == k && k == l) ? 42 : 0.0);
        }
      }
    }
  }
}

TEST(Delta, Widen) {
  auto D2 = ttl::Delta<2, 2, double>(3u);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(D2[i][j], (i == j) ? 3.0 : 0.0);
    }
  }
}


template <int D, class S>
auto Identity() {
  static constexpr ttl::Index<'i'> i;
  static constexpr ttl::Index<'j'> j;
  static constexpr ttl::Index<'k'> k;
  static constexpr ttl::Index<'l'> l;
  return ttl::expressions::force((ttl::Delta<2,D,S>()(i,k)*ttl::Delta<2,D,S>()(j,l)).to(i,j,k,l));
}

TEST(Delta, Identity) {
  static constexpr ttl::Index<'i'> i;
  static constexpr ttl::Index<'j'> j;
  static constexpr ttl::Index<'k'> k;
  static constexpr ttl::Index<'l'> l;
  auto I = Identity<3,int>();
  std::cout << "I\n" << I(i,j,k,l) << "\n";

  ttl::Tensor<2,3,int> A = {1, 3, 5,
                            7, 9, 11,
                            13, 15, 17}, B{};

  std::cout << "A\n" << A(i,j) << "\n";
  B(i,j) = I(i,j,k,l) * A(k,l);
  std::cout << "B\n" << B(i,j) << "\n";

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(A[i][j], B[i][j]);
    }
  }
}
