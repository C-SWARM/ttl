#include <ttl/ttl.h>
#include <gtest/gtest.h>

using namespace ttl;

static constexpr Index<'i'> i;
static constexpr Index<'j'> j;
static constexpr Index<'i'> k;
static constexpr Index<'j'> l;
static constexpr Index<'i'> m;
static constexpr Index<'j'> n;

TEST(Identity, Scalar) {
  auto d = identity();
  EXPECT_EQ(d, 1);
}
