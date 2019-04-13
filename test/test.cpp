#include <ttl2/ttl.hpp>

int main() {
  constexpr ttl::Index<'i'> i;
  ttl::Tensor<2, 2, double> A;

  A(0,0) = 1;
  A(0,1) = 2;
  A(1,0) = 3;
  A(1,1) = 4;

  ttl::Tensor<2, 2, double> B = { 1, 2, 3, 4 };
  B = A;
  B = A(i,i);

  return 0;
}
