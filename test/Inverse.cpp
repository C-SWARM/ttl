#include <ttl/ttl.h>
#include <gtest/gtest.h>

TEST(Inverse, Basic_2_2) {
  ttl::Tensor<2,2,double> A = {1, 2, 3, 5};
  auto B = ttl::inverse(A);
  EXPECT_EQ(B[0][0], -5);
  EXPECT_EQ(B[0][1], 2);
  EXPECT_EQ(B[1][0], 3);
  EXPECT_EQ(B[1][1], -1);
}

TEST(Inverse, Basic_2_3) {
  const ttl::Tensor<2,3,const double> A = {1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 10};
  auto B = ttl::inverse(A);
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, Extern_2_3) {
  const double a[9] = {1, 2, 3,
                       4, 5, 6,
                       7, 8, 10};
  const ttl::Tensor<2,3,const double*> A(a);
  auto B = ttl::inverse(A);
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, RValue_2_3) {
  const ttl::Tensor<2,3,double> B = ttl::inverse(ttl::Tensor<2,3,double>{1, 2, 3, 4, 5, 6, 7, 8, 10});
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, RValue_2_3_Infer) {
  auto B = ttl::inverse(ttl::Tensor<2,3,double>{1, 2, 3, 4, 5, 6, 7, 8, 10});
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, RValueExpression_2_3) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  const ttl::Tensor<2,3,double> A = {1, 2, 3,
                                     4, 5, 6,
                                     7, 8, 10}, C = {};
  const ttl::Tensor<2,3,double> B = ttl::inverse(1*(A(i,j) + C(i,j)));
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, Expression_2_3) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  const ttl::Tensor<2,3,double> A = {1, 2, 3,
                                     4, 5, 6,
                                     7, 8, 10}, C = {};
  auto e = 1*(A(i,j) + C(i,j));
  const ttl::Tensor<2,3,double> B = ttl::inverse(e);
  EXPECT_EQ(B[0][0], -2/3.0);
  EXPECT_EQ(B[0][1], -(1.0 + 1/3.0));
  EXPECT_EQ(B[0][2], 1);
  EXPECT_EQ(B[1][0], -2/3.0);
  EXPECT_EQ(B[1][1], 3.0 + 2/3.0);
  EXPECT_EQ(B[1][2], -2);
  EXPECT_EQ(B[2][0], 1);
  EXPECT_EQ(B[2][1], -2);
  EXPECT_EQ(B[2][2], 1);
}

TEST(Inverse, Basic_4xo_3) {
  const ttl::Tensor<4,3,double> A = {};
  auto B = ttl::inverse(A);
}

TEST(Inverse, Basic_2_4) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  ttl::Index<'k'> k;

  ttl::Tensor<2,4,double> A = {};

  for (int i = 0; i < 4; i++){   //init A
    for (int j = 0; j < 4; j++){   //init A
      A[i][j] = (rand()%10);
    }
  }

  ttl::Tensor<2,4,double> B = ttl::inverse(A);


  ttl::Tensor<2,4,double> AxB = {};
  AxB(i,j) = A(i,k) * B(k,j);

}


TEST(Inverse, Basic_2_9) {
  ttl::Index<'i'> i;
  ttl::Index<'j'> j;
  ttl::Index<'k'> k;

  ttl::Tensor<2,9,double> A = {};

  for (int i = 0; i < 9; i++){   //init A
    for (int j = 0; j < 9; j++){   //init A
      A[i][j] = (rand()%10);
    }
  }

  ttl::Tensor<2,9,double> B = ttl::inverse(A);

  ttl::Tensor<2,9,double> AxB = {};
  AxB(i,j) = A(i,k) * B(k,j);

}
