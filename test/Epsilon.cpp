#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Epsilon, 2_2) {
  auto e = ttl::epsilon<2,int>();

  EXPECT_EQ(e[0],  0); EXPECT_EQ(e[1], 1);
  EXPECT_EQ(e[2], -1); EXPECT_EQ(e[3], 0);
}

TEST(Epsilon, 3_3) {
  auto e = ttl::epsilon<3,int>();

  EXPECT_EQ(e[0],   0); EXPECT_EQ(e[1],  0); EXPECT_EQ(e[2],   0);
  EXPECT_EQ(e[3],   0); EXPECT_EQ(e[4],  0); EXPECT_EQ(e[5],   1);
  EXPECT_EQ(e[6],   0); EXPECT_EQ(e[7], -1); EXPECT_EQ(e[8],   0);

  EXPECT_EQ(e[9],   0); EXPECT_EQ(e[10], 0); EXPECT_EQ(e[11], -1);
  EXPECT_EQ(e[12],  0); EXPECT_EQ(e[13], 0); EXPECT_EQ(e[14],  0);
  EXPECT_EQ(e[15],  1); EXPECT_EQ(e[16], 0); EXPECT_EQ(e[17],  0);

  EXPECT_EQ(e[18],  0); EXPECT_EQ(e[19], 1); EXPECT_EQ(e[20],  0);
  EXPECT_EQ(e[21], -1); EXPECT_EQ(e[22], 0); EXPECT_EQ(e[23],  0);
  EXPECT_EQ(e[24],  0); EXPECT_EQ(e[25], 0); EXPECT_EQ(e[26],  0);
}
