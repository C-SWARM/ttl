// -*- C++ -*-
#ifndef TTL_LIBRARY_EPSILON_H
#define TTL_LIBRARY_EPSILON_H

#include <array>

namespace ttl {
namespace detail {
template <std::size_t N, class Op>
void permute(std::array<int,N> index, Op&& op, int i = 0, int even = 1) {
  if (i == N) {
    op(index, even);
  }
  else {
    for (int j = i; j < N; ++j) {
      std::swap(index[i], index[j]);
      permute(index, std::forward<Op>(op), i + 1, (i == j) ? even : 1 - even);
      std::swap(index[i], index[j]);
    }
  }
}

/// @todo Hard coded initializer for now... can use metaprogramming to generate
///       these.
template <int>
struct init;

template <>
struct init<1> {
  static constexpr std::array<int,1> value() {
    return {0};
  }
};

template <>
struct init<2> {
  static constexpr std::array<int,2> value() {
    return {0,1};
  }
};

template <>
struct init<3> {
  static constexpr std::array<int,3> value() {
    return {0,1,2};
  }
};

template <>
struct init<4> {
  static constexpr std::array<int,4> value() {
    return {0,1,2,3};
  }
};
}

template <int N, class S, class T = S>
Tensor<N,N,S> epsilon(T t = T{1}) {
  Tensor<N,N,S> A = {};
  auto index = detail::init<N>::value();
  detail::permute(index, [&](std::array<int,N> i, int even) {
      A.eval(i) = (even) ? t : -t;
    });
  return A;
}
}

#endif // #ifndef TTL_LIBRARY_EPSILON_H
