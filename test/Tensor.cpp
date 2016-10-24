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

#if 0
TEST(Tensor, TensorPointer) {
  double a[] = {E};
  double b[] = {E, PI};

  ttl::Tensor<1, 1, double*> A{a};
  EXPECT_EQ(A[0], a[0]);

  ttl::Tensor<1, 2, double*> B{b};
  EXPECT_EQ(B[0], b[0]);
  EXPECT_EQ(B[1], b[1]);
}

TEST(Tensor, TensorConstPointer) {
  const double a[] = {E};
  const double b[] = {E, PI};

  ttl::Tensor<1, 1, const double*> A{a};
  EXPECT_EQ(A[0], a[0]);

  ttl::Tensor<1, 2, const double*> B{b};
  EXPECT_EQ(B[0], b[0]);
  EXPECT_EQ(B[1], b[1]);
}

TEST(Tensor, Assign) {
  ttl::Tensor<2, 2, double> I{1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], I[0]);
  EXPECT_EQ(A[1], I[1]);
  EXPECT_EQ(A[2], I[2]);
  EXPECT_EQ(A[3], I[3]);
}

TEST(Tensor, AssignWiden) {
  ttl::Tensor<2, 2, int> I{1, 0, 0, 1};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], double(I[0]));
  EXPECT_EQ(A[1], double(I[1]));
  EXPECT_EQ(A[2], double(I[2]));
  EXPECT_EQ(A[3], double(I[3]));
}

TEST(Tensor, AssignToConst) {
  ttl::Tensor<2, 2, int> I{1, 0, 0, 1};
  ttl::Tensor<2, 2, const int> A = I;
  EXPECT_EQ(A[0], double(I[0]));
  EXPECT_EQ(A[1], double(I[1]));
  EXPECT_EQ(A[2], double(I[2]));
  EXPECT_EQ(A[3], double(I[3]));
}

TEST(Tensor, AssignInitializer) {
  ttl::Tensor<2, 2, double> A;
  A = {1.0, 0.0, 0.0, 1.0};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 0.0);
  EXPECT_EQ(A[2], 0.0);
  EXPECT_EQ(A[3], 1.0);
}

TEST(Tensor, AssignInitializerWiden) {
  ttl::Tensor<2, 2, double> A;
  A = {1, 0, 0, 1};
  EXPECT_EQ(A[0], 1.0);
  EXPECT_EQ(A[1], 0.0);
  EXPECT_EQ(A[2], 0.0);
  EXPECT_EQ(A[3], 1.0);
}

TEST(Tensor, AssignConst) {
  const ttl::Tensor<2, 2, double> I{1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], I[0]);
  EXPECT_EQ(A[1], I[1]);
  EXPECT_EQ(A[2], I[2]);
  EXPECT_EQ(A[3], I[3]);
}

TEST(Tensor, AssignExternal) {
  double i[] = {1.0, 0.0, 0.0, 1.0};
  double a[4];
  ttl::Tensor<2, 2, double*> I{i}, A{a};
  A = I;
  EXPECT_EQ(a[0], i[0]);
  EXPECT_EQ(a[1], i[1]);
  EXPECT_EQ(a[2], i[2]);
  EXPECT_EQ(a[3], i[3]);
}

TEST(Tensor, AssignExternInitializer) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  A = {1.0, 0.0, 0.0, 1.0};
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 0.0);
  EXPECT_EQ(a[2], 0.0);
  EXPECT_EQ(a[3], 1.0);
}

TEST(Tensor, AssignExternInitializerWiden) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  A = {1, 0, 0, 1};
  EXPECT_EQ(a[0], 1.0);
  EXPECT_EQ(a[1], 0.0);
  EXPECT_EQ(a[2], 0.0);
  EXPECT_EQ(a[3], 1.0);
}

TEST(Tensor, AssignConstExternal) {
  const double i[] = {1.0, 0.0, 0.0, 1.0};
  double a[4];
  const ttl::Tensor<2, 2, const double*> I{i};
  ttl::Tensor<2, 2, double*> A{a};
  A = I;
  EXPECT_EQ(a[0], i[0]);
  EXPECT_EQ(a[1], i[1]);
  EXPECT_EQ(a[2], i[2]);
  EXPECT_EQ(a[3], i[3]);
}

TEST(Tensor, AssignFromExternal) {
  double i[] = {1.0, 0.0, 0.0, 1.0};
  ttl::Tensor<2, 2, double*> I{i};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], i[0]);
  EXPECT_EQ(A[1], i[1]);
  EXPECT_EQ(A[2], i[2]);
  EXPECT_EQ(A[3], i[3]);
}

TEST(Tensor, AssignFromConstExternal) {
  const double i[] = {1.0, 0.0, 0.0, 1.0};
  const ttl::Tensor<2, 2, const double*> I{i};
  ttl::Tensor<2, 2, double> A;
  A = I;
  EXPECT_EQ(A[0], i[0]);
  EXPECT_EQ(A[1], i[1]);
  EXPECT_EQ(A[2], i[2]);
  EXPECT_EQ(A[3], i[3]);
}

TEST(Tensor, AssignToExternal) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  ttl::Tensor<2, 2, double> I = {1.0, 0.0,
                                 0.0, 1.0};
  A = I;
  EXPECT_EQ(a[0], I[0]);
  EXPECT_EQ(a[1], I[1]);
  EXPECT_EQ(a[2], I[2]);
  EXPECT_EQ(a[3], I[3]);
}

TEST(Tensor, AssignConstToExternal) {
  double a[4];
  ttl::Tensor<2, 2, double*> A{a};
  const ttl::Tensor<2, 2, const double> I = {1.0, 0.0,
                                             0.0, 1.0};
  A = I;
  EXPECT_EQ(a[0], I[0]);
  EXPECT_EQ(a[1], I[1]);
  EXPECT_EQ(a[2], I[2]);
  EXPECT_EQ(a[3], I[3]);
}

TEST(Tensor, AssignFromRValue) {
  ttl::Tensor<3, 3, double> T;
  T = ttl::Tensor<3, 3, double>{};
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(T[i], 0.0);
  }
}

TEST(Tensor, AssignExternalFromRValue) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A = ttl::Tensor<3, 3, double>{};
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], 0.0);
  }
}

TEST(Tensor, Fill) {
  ttl::Tensor<3, 3, double> A;
  A.fill(E);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(A[i], E);
  }
}

TEST(Tensor, FillWiden) {
  ttl::Tensor<3, 3, double> A;
  A.fill(1u);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(A[i], 1.0);
  }
}

TEST(Tensor, FillExternal) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A.fill(E);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], E);
  }
}

TEST(Tensor, FillExternalWiden) {
  double a[27];
  ttl::Tensor<3, 3, double*> A{a};
  A.fill(1u);
  for (int i = 0; i < 27; ++i) {
    EXPECT_EQ(a[i], 1.0);
  }
}

static constexpr ttl::Index<'i'> i;
static constexpr ttl::Index<'j'> j;
static constexpr ttl::Index<'k'> k;
static constexpr ttl::Index<'l'> l;

constexpr int index(int D, int i, int j, int k, int l) {
  return i * D * D * D + j * D * D + k * D + l;
}

constexpr int index(int D, int i, int j, int k) {
  return i * D * D + j * D + k;
}

constexpr int index(int D, int i, int j) {
  return i * D + j;
}

TEST(Tensor, TensorExprAssignment) {
  ttl::Tensor<1, 1, double> a, b;
  a[0] = 10;
  b(i) = a(i);
  EXPECT_EQ(a[0], b[0]);
  // assign(b(i), a(j));

  ttl::Tensor<2, 3, double> T, U;
  U.fill(3.14);
  T(i, j) = U(i, j);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      EXPECT_EQ(T[index(3,n,m)], U[index(3,n,m)]);
    }
  }

  U[index(3,1,0)] = 42;
  T(i,j) = U(j,i);
  EXPECT_EQ(T[index(3,0,1)], 42);

  ttl::Tensor<3, 3, double> A, B;
  B(i,j,k) = A(k,j,i);
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      for (int o = 0; o < 3; ++o) {
        EXPECT_EQ(B[index(3,n,m,o)], A[index(3,o,m,n)]);
      }
    }
  }
}

TEST(Tensor, UnaryOp) {
  ttl::Tensor<2, 2, int> A, B;
  A.fill(1);
  B(i, j) = -A(i, j);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(B[index(2,i,j)], -1);
    }
  }
}

TEST(Tensor, BinaryOp) {
  ttl::Tensor<1, 4, double> x, y, z;
  x.fill(1);
  y.fill(2);

  z(i) = x(i) + y(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 3);
  }

  z(i) = y(i) - x(i) - x(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 0);
  }

  z(i) = y(i) - x(i) - x(i) + y(i) - x(i);
  for (int n = 0; n < 4; ++n) {
    EXPECT_EQ(z[n], 1);
  }
  // z(i) = y(i) - x(j);
}

TEST(Tensor, ScalarOp) {
  ttl::Tensor<0, 1, double> x, y;
  x.fill(1);

  y() = 3.1 * x();
  EXPECT_EQ(y[0], 3.1);

  x() = y() * 3.0;
  EXPECT_EQ(x[0], 9.3);

  ttl::Tensor<2, 2, int> s = ttl::Delta<2,2,int>(2);
  ttl::Tensor<2, 2, int> t;
  ttl::Tensor<2, 2, double> u;

  t(i,j) = (ttl::Tensor<2,2,int>{}.fill(2)(i,j) - s(j,i))/2;
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 1);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 0);

  s(i,j) = 1.2 * t(i,j);
  EXPECT_EQ(s[0], 0);
  EXPECT_EQ(s[1], 1);
  EXPECT_EQ(s[2], 1);
  EXPECT_EQ(s[3], 0);

  u(i,j) = 1.2 * t(i,j);
  EXPECT_EQ(u[0], 0);
  EXPECT_EQ(u[1], 1.2);
  EXPECT_EQ(u[2], 1.2);
  EXPECT_EQ(u[3], 0);

  u(i,j) = s(i,j) % 1;
  EXPECT_EQ(u[0], 0.0);
  EXPECT_EQ(u[1], 0.0);
  EXPECT_EQ(u[2], 0.0);
  EXPECT_EQ(u[3], 0.0);
}

TEST(Tensor, TensorProduct) {
  ttl::Tensor<2,2,double> A, B, C;
  A.fill(3);
  B.fill(2);

  ttl::Tensor<0,2,double> s;
  s() = B(k,l) * B(k,l);
  EXPECT_EQ(s[0], 16);

  C(i,k) = A(i,j) * B(j,k);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 12);
    }
  }

  C(j,k) = A(i,j) * B(k,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 12);
    }
  }

  C(i,j) = (A(i,j) * B(k,l)) * B(k,l);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  C(i,j) = A(i,j) * s();
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  C(i,j) = A(i,j) * (B(k,l) * B(k,l));
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(C[index(2,n,m)], 48);
    }
  }

  ttl::Tensor<4,2,double> D;

  D(i,j,k,l) = A(i,j) * B(k,l);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      for (int o = 0; o < 2; ++o) {
        for (int p = 0; p < 2; ++p) {
          EXPECT_EQ(D[index(2,n,m,o,p)], 6);
        }
      }
    }
  }

  ttl::Tensor<1,2,double> x, y;
  x[0] = 2;
  x[1] = 3;

  auto E = ttl::Delta<2,2,double>();
  E[1] = 1;
  y(i) = (B(i,j) + E(i,j)) * x(j);
  EXPECT_EQ(y[0], 15);
  EXPECT_EQ(y[1], 13);

  y(j) = (B(i,j) + E(i,j)) * x(i);
  EXPECT_EQ(y[0], 12);
  EXPECT_EQ(y[1], 15);

  y(l) = D(i,j,k,l) * 0.5 * ((B(i,j) + E(i,j)) + (B(j,i) + E(i,j))) * x(k);
  EXPECT_EQ(y[0], 330);
  EXPECT_EQ(y[1], 330);
}

TEST(Tensor, ExternalStorage) {
  double storage[12];
  ttl::Tensor<2,2,double*> A{&storage[0]}, B{&storage[4]}, C{&storage[8]};
  B.fill(1);
  C.fill(2);

  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(B[index(2,n,m)], 1);
      EXPECT_EQ(C[index(2,n,m)], 2);
    }
  }

  A(i,j) = B(i,j) + C(j,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], 3);
    }
  }

  ttl::Tensor<1,2,double*> x(&storage[0]), y(&storage[4]);
  x(i) = C(i,j) * y(j);
  EXPECT_EQ(A[0], 4);
  EXPECT_EQ(A[1], 4);

  A(i,j) = B(i,j);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], B[index(2,n,m)]);
    }
  }

  ttl::Tensor<2,2,double> D;
  D.fill(3.14);
  A(i,j) = D(i,j);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], D[index(2,n,m)]);
    }
  }

  D(i,j) = B(j,i);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(D[index(2,n,m)], B[index(2,n,m)]);
    }
  }

  A(i,k) = C(i,j) * D(j,k);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      EXPECT_EQ(A[index(2,n,m)], 4);
    }
  }
}
#endif
