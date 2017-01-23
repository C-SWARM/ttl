#include <ttl/ttl.h>
#include <gtest/gtest.h>

constexpr int index(int D, int i, int j, int k, int l) {
  return i * D * D * D + j * D * D + k * D + l;
}

constexpr int index(int D, int i, int j, int k) {
  return i * D * D + j * D + k;
}

constexpr int index(int D, int i, int j) {
  return i * D + j;
}

TEST(Delta, 2_6) {
  auto D2 = ttl::Delta<2, 6, double>();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_EQ(D2.get(index(6,i,j)), (i == j) ? 1.0 : 0.0);
    }
  }
}

TEST(Delta, 3_4) {
  auto D3 = ttl::Delta<3, 4, double>(3.14);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(D3.get(index(4,i,j,k)), (i == j && j == k) ? 3.14 : 0.0);
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
          EXPECT_EQ(D4.get(index(3,i,j,k,l)),
                    (i == j && j == k && k == l) ? 42 : 0.0);
        }
      }
    }
  }
}

TEST(Delta, Widen) {
  auto D2 = ttl::Delta<2, 2, double>(3u);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(D2.get(index(2,i,j)), (i == j) ? 3.0 : 0.0);
    }
  }
}
