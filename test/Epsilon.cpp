#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;

TEST(Epsilon, 2_2) {
  auto e = ttl::epsilon<2>(i,j);

  EXPECT_EQ(e(0,0),  0); EXPECT_EQ(e(0,1), 1);
  EXPECT_EQ(e(1,0), -1); EXPECT_EQ(e(1,1), 0);
}

TEST(Epsilon, 3_3) {
  auto e = ttl::epsilon<3>(i,j,k);

  EXPECT_EQ(e(0,0,0), 0); EXPECT_EQ(e(0,0,1),  0); EXPECT_EQ(e(0,0,2), 0);
  EXPECT_EQ(e(0,1,0), 0); EXPECT_EQ(e(0,1,1),  0); EXPECT_EQ(e(0,1,2), 1);
  EXPECT_EQ(e(0,2,0), 0); EXPECT_EQ(e(0,2,1), -1); EXPECT_EQ(e(0,2,2), 0);

  EXPECT_EQ(e(1,0,0), 0); EXPECT_EQ(e(1,0,1), 0); EXPECT_EQ(e(1,0,2), -1);
  EXPECT_EQ(e(1,1,0), 0); EXPECT_EQ(e(1,1,1), 0); EXPECT_EQ(e(1,1,2),  0);
  EXPECT_EQ(e(1,2,0), 1); EXPECT_EQ(e(1,2,1), 0); EXPECT_EQ(e(1,2,2),  0);

  EXPECT_EQ(e(2,0,0),  0); EXPECT_EQ(e(2,0,1), 1); EXPECT_EQ(e(2,0,2), 0);
  EXPECT_EQ(e(2,1,0), -1); EXPECT_EQ(e(2,1,1), 0); EXPECT_EQ(e(2,1,2), 0);
  EXPECT_EQ(e(2,2,0),  0); EXPECT_EQ(e(2,2,1), 0); EXPECT_EQ(e(2,2,2), 0);
}
