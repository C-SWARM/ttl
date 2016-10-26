#include <ttl/ttl.h>
#include <gtest/gtest.h>

static constexpr double E = 2.72;
static constexpr double PI = 3.14;

TEST(Tensor, CtorDefault) {
  ttl::Tensor<2, 2, double> A;
}

TEST(Tensor, ArrayIndexing) {
  ttl::Tensor<2, 2, double> A;
  A[0] = 0.0;
  A[1] = 1.0;
  A[2] = E;
  A[3] = PI;
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
  EXPECT_EQ(A[2], E);
  EXPECT_EQ(A[3], PI);
}

TEST(Tensor, Ctor) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, CtorZero) {
  ttl::Tensor<2, 2, int> A = {};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 0);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, CtorZeroSuffix) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, CtorIgnoreOverflow) {
  ttl::Tensor<2, 2, int> B = {};
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
  EXPECT_EQ(B[0], 0);
  EXPECT_EQ(B[1], 0);
  EXPECT_EQ(B[2], 0);
  EXPECT_EQ(B[3], 0);
}

TEST(Tensor, CtorWiden) {
  ttl::Tensor<1, 3, double> A = {int(1), float(E), PI};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], float(E));
  EXPECT_EQ(A[2], PI);
}

TEST(Tensor, ConstCtor) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, ZeroConst) {
  ttl::Tensor<2, 2, const int> A = {};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 0);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, ConstCtorZero) {
  const ttl::Tensor<2, 2, int> A = {};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 0);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, CtorConst) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, ConstCtorConst) {
  const ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, ConstCtorZeroConst) {
  const ttl::Tensor<2, 2, const int> A = {};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 0);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, CopyCtor) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorWiden) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, float> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorFromConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorToConst) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  const ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorFromConstToConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  const ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorFromConstData) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyCtorToConstData) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, const int> B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, MoveCtor) {
  ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, MoveCtorWiden) {
  ttl::Tensor<2, 2, double> A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
  EXPECT_EQ(A[2], 2.0);
  EXPECT_EQ(A[3], 3.0);
}

TEST(Tensor, MoveCtorToConst) {
  const ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, MoveCtorFromConstData) {
  ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, const int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, MoveCtorToConstData) {
  ttl::Tensor<2, 2, const int> A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, Assign) {
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2, 3};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, AssignZero) {
  ttl::Tensor<2, 2, int> A;
  A = {};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 0);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, AssignZeroSuffix) {
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 0);
}

TEST(Tensor, AssignIgnoreOverflow) {
  ttl::Tensor<2, 2, int> B = {};
  ttl::Tensor<2, 2, int> A;
  A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
  EXPECT_EQ(B[0], 0);
  EXPECT_EQ(B[1], 0);
  EXPECT_EQ(B[2], 0);
  EXPECT_EQ(B[3], 0);
}

TEST(Tensor, AssignWiden) {
  ttl::Tensor<1, 3, double> A;
  A = {int(1), float(E), PI};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], float(E));
  EXPECT_EQ(A[2], PI);
}

TEST(Tensor, Copy) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyWiden) {
  ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, float> B;
  B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyFromConst) {
  const ttl::Tensor<2, 2, int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, CopyFromConstData) {
  ttl::Tensor<2, 2, const int> A = {0, 1, 2, 3};
  ttl::Tensor<2, 2, int> B;
  B = A;
  EXPECT_EQ(B[0], A[0]);
  EXPECT_EQ(B[1], A[1]);
  EXPECT_EQ(B[2], A[2]);
  EXPECT_EQ(B[3], A[3]);
}

TEST(Tensor, Move) {
  ttl::Tensor<2, 2, int> A;
  A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, MoveWiden) {
  ttl::Tensor<2, 2, double> A;
  A = std::move(ttl::Tensor<2, 2, int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0.0);
  EXPECT_EQ(A[1], 1.0);
  EXPECT_EQ(A[2], 2.0);
  EXPECT_EQ(A[3], 3.0);
}

TEST(Tensor, MoveFromConstData) {
  ttl::Tensor<2, 2, int> A;
  A = std::move(ttl::Tensor<2, 2, const int>{0, 1, 2, 3});
  EXPECT_EQ(A[0], 0);
  EXPECT_EQ(A[1], 1);
  EXPECT_EQ(A[2], 2);
  EXPECT_EQ(A[3], 3);
}

TEST(Tensor, Fill) {
  ttl::Tensor<2, 2, double> A;
  A.fill(E);
  EXPECT_EQ(A[0], E);
  EXPECT_EQ(A[1], E);
  EXPECT_EQ(A[2], E);
  EXPECT_EQ(A[3], E);
}

TEST(Tensor, FillWiden) {
  ttl::Tensor<2, 2, double> A;
  A.fill(2);
  EXPECT_EQ(A[0], 2.0);
  EXPECT_EQ(A[1], 2.0);
  EXPECT_EQ(A[2], 2.0);
  EXPECT_EQ(A[3], 2.0);
}

TEST(ExternalTensor, Ctor) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int*> A(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  ttl::Tensor<2, 2, const int*> B(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  const ttl::Tensor<2, 2, int*> C(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);

  const ttl::Tensor<2, 2, const int*> D(a);
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, CtorPointer) {
  int a[8] = {0,1,2,3,4,5,6,7};
  ttl::Tensor<2, 2, int*> A(&a[2]);
  EXPECT_EQ(A[0], 2);
  EXPECT_EQ(A[1], 3);
  EXPECT_EQ(A[2], 4);
  EXPECT_EQ(A[3], 5);
}

TEST(ExternalTensor, CtorConst) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2,2,const int*> A(a);
  const ttl::Tensor<2,2,const int*> B(a);
}

TEST(ExternalTensor, ArrayIndexing) {
  double a[4];
  ttl::Tensor<2, 2, double*> A(a);
  A[0] = 0.0;
  A[1] = 1.0;
  A[2] = E;
  A[3] = PI;
  EXPECT_EQ(a[0], 0.0);
  EXPECT_EQ(a[1], 1.0);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], PI);
}

TEST(ExternalTensor, CopyCtor) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int*> A(a);
  ttl::Tensor<2, 2, int*> B = A;
  EXPECT_EQ(B[0], a[0]);
  EXPECT_EQ(B[1], a[1]);
  EXPECT_EQ(B[2], a[2]);
  EXPECT_EQ(B[3], a[3]);
}

TEST(ExternalTensor, CopyCtorConst) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, const int*> B = A;
  EXPECT_EQ(B[0], a[0]);
  EXPECT_EQ(B[1], a[1]);
  EXPECT_EQ(B[2], a[2]);
  EXPECT_EQ(B[3], a[3]);

  const ttl::Tensor<2, 2, const int*> C(a);
  const ttl::Tensor<2, 2, const int*> D = C;
  EXPECT_EQ(D[0], a[0]);
  EXPECT_EQ(D[1], a[1]);
  EXPECT_EQ(D[2], a[2]);
  EXPECT_EQ(D[3], a[3]);
}

TEST(ExternalTensor, MoveCtor) {
  int a[4];
  ttl::Tensor<2, 2, int*> A = std::move(ttl::Tensor<2, 2, int*>(a));
  A[0] = 0;
  A[1] = 1;
  A[2] = 2;
  A[3] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, CopyCtorExternal) {
  const int a[4] = {0,1,2,3};
  const ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, int> B = A;
  B[3] = 0;
  EXPECT_EQ(B[0], 0);
  EXPECT_EQ(B[1], 1);
  EXPECT_EQ(B[2], 2);
  EXPECT_EQ(B[3], 0);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, MoveCtorExternal) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int> A = std::move(ttl::Tensor<2, 2, int*>(a));
  A[0] = 0;
  A[1] = 1;
  A[2] = 2;
  A[3] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, AssignExternal) {
  const int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, const int*> A(a);
  ttl::Tensor<2, 2, int> B;
  B = A;
  B[3] = 0;
  EXPECT_EQ(B[0], 0);
  EXPECT_EQ(B[1], 1);
  EXPECT_EQ(B[2], 2);
  EXPECT_EQ(B[3], 0);
  EXPECT_EQ(a[3], 3);
}

TEST(Tensor, MoveExternal) {
  int a[4] = {0,1,2,3};
  ttl::Tensor<2, 2, int> A;
  A = std::move(ttl::Tensor<2, 2, int*>(a));
  A[0] = 0;
  A[1] = 1;
  A[2] = 2;
  A[3] = 3;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, Assign) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = {0,1,2,3};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, AssignZeroSuffix) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = {0, 1, 2};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 0);
}

TEST(ExternalTensor, AssignIgnoreOverflow) {
  int a[8] = {};
  ttl::Tensor<2, 2, int*> A(a);
  A = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
  EXPECT_EQ(a[4], 0);
  EXPECT_EQ(a[5], 0);
  EXPECT_EQ(a[6], 0);
  EXPECT_EQ(a[7], 0);
}

TEST(ExternalTensor, AssignInternal) {
  ttl::Tensor<2, 2, int> B = {0,1,2,3};
  int a[4] ;
  ttl::Tensor<2, 2, int*> A(a);
  A = B;
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, MoveInternal) {
  int a[4];
  ttl::Tensor<2, 2, int*> A(a);
  A = std::move(ttl::Tensor<2, 2, int>{0,1,2,3});
  EXPECT_EQ(a[0], 0);
  EXPECT_EQ(a[1], 1);
  EXPECT_EQ(a[2], 2);
  EXPECT_EQ(a[3], 3);
}

TEST(ExternalTensor, Fill) {
  double a[4];
  ttl::Tensor<2, 2, double*> A(a);
  A.fill(E);
  EXPECT_EQ(a[0], E);
  EXPECT_EQ(a[1], E);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], E);
}

TEST(ExternalTensor, FillRvalue) {
  double a[4];
  ttl::Tensor<2, 2, double*>(a).fill(E);
  EXPECT_EQ(a[0], E);
  EXPECT_EQ(a[1], E);
  EXPECT_EQ(a[2], E);
  EXPECT_EQ(a[3], E);
}
