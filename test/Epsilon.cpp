#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Epsilon, 2_2) {
  auto e = ttl::epsilon<2,int>();

  EXPECT_EQ(e.get(0),  0); EXPECT_EQ(e.get(1), 1);
  EXPECT_EQ(e.get(2), -1); EXPECT_EQ(e.get(3), 0);
}

TEST(Epsilon, 3_3) {
  auto e = ttl::epsilon<3,int>();

  EXPECT_EQ(e.get(0),   0); EXPECT_EQ(e.get(1),  0); EXPECT_EQ(e.get(2),   0);
  EXPECT_EQ(e.get(3),   0); EXPECT_EQ(e.get(4),  0); EXPECT_EQ(e.get(5),   1);
  EXPECT_EQ(e.get(6),   0); EXPECT_EQ(e.get(7), -1); EXPECT_EQ(e.get(8),   0);

  EXPECT_EQ(e.get(9),   0); EXPECT_EQ(e.get(10), 0); EXPECT_EQ(e.get(11), -1);
  EXPECT_EQ(e.get(12),  0); EXPECT_EQ(e.get(13), 0); EXPECT_EQ(e.get(14),  0);
  EXPECT_EQ(e.get(15),  1); EXPECT_EQ(e.get(16), 0); EXPECT_EQ(e.get(17),  0);

  EXPECT_EQ(e.get(18),  0); EXPECT_EQ(e.get(19), 1); EXPECT_EQ(e.get(20),  0);
  EXPECT_EQ(e.get(21), -1); EXPECT_EQ(e.get(22), 0); EXPECT_EQ(e.get(23),  0);
  EXPECT_EQ(e.get(24),  0); EXPECT_EQ(e.get(25), 0); EXPECT_EQ(e.get(26),  0);
}
