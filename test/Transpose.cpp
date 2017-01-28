#include <ttl/ttl.h>
#include <gtest/gtest.h>

const ttl::Index<'i'> i;
const ttl::Index<'j'> j;
const ttl::Index<'k'> k;

TEST(Transpose, Basic_2_2) {
  ttl::Tensor<2,2,int> A = {1,2,3,4}, B = transpose(A(i,j));
  for (int n = 0; n < 2; ++n)
    for (int m = 0; m < 2; ++m)
      EXPECT_EQ(B(n,m), A(m,n));
}

TEST(Transpose, Basic_2_3) {
  ttl::Tensor<2,3,int> A = {1,2,3,4,5,6,7,8}, B = transpose(A(i,j));
  for (int n = 0; n < 3; ++n)
    for (int m = 0; m < 3; ++m)
      EXPECT_EQ(B(n,m), A(m,n));
}

TEST(Transpose, Basic_3_3) {
  ttl::Tensor<3,3,int> A = {1,2,3,4,5,6,7,8,9,10,11,12,
                            13,14,15,16,17,18,19,20,21},
                       B = transpose(A(i,j,k));
  for (int n = 0; n < 3; ++n)
    for (int m = 0; m < 3; ++m)
      for (int o = 0; o < 3; ++o)
        EXPECT_EQ(B(n,m,o), A(o,m,n));
}

