// -*- C++ -*-
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

#include <array>
#include <iostream>

namespace ttl {
template <int Rank, typename Scalar, int Dimension>
class Tensor {
 public:
  using Index = std::array<int, Rank>;

  Tensor() {
  }

  static constexpr int size() {
    return pow(Dimension, Rank);
  }

  /// Multidimensional addressing based on an array of integers.
  constexpr Scalar operator[](Index&& i) const {
    return value_[index_of(0, std::forward<Index>(i))];
  }

  /// Multidimensional addressing based on an array of integers.
  Scalar& operator[](Index&& i) {
    return value_[index_of(0, std::forward<Index>(i))];
  }

  /// Simple linear addressing.
  constexpr Scalar operator[](int i) const {
    return value_[i];
  }

  // template <typename ... Indices,
  //           typename = typename std::enable_if<sizeof...(Indices) == R>::type>
  // auto operator()(Indices ... indices) {
  //   return expressions::Bind<Tensor<R, S, D>, IndexSet<Indices...>>(*this);
  // }

 private:
  static constexpr int offset_of(int n, Index&& i) {
    return i[n] * pow(Dimension, Rank - n - 1);
  }

  /// Recursively compute the index for an array.
  static constexpr int index_of(int n, Index&& i) {
    return (n < Rank) ? offset_of(n, std::forward<Index>(i)) + index_of(n + 1, std::forward<Index>(i)) : 0;
  }

  /// Recursively compute k^n at compile time for integers.
  ///
  /// This is used by the class to allocate the appropriate number of Scalar
  /// values in the @p values_ member.
  static constexpr int pow(int k, int n) {
    return (n) ? k * pow(k, n - 1) : 1;
  }

  Scalar value_[size()];
};
}

#endif // TTL_TENSOR_H
