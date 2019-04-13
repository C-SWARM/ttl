#pragma once

#include <tuple>

namespace ttl {
template <int D, int n, int N>
struct forall {
  template <class Index, class Op>
  static constexpr void op(Index index, Op&& op) noexcept {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index) = i;
      forall<D, n + 1, N>::op(index, std::forward<Op>(op));
    }
  }
};

template <int D, int N>
struct forall<D, N, N> {
  template <class Index, class Op>
  static constexpr void op(Index index, Op&& op) noexcept {
    op(index);
  }
};

template <int D, int n, int N>
struct contract {
  template <class Index, class Op>
  static constexpr auto op(Index index, Op&& op) noexcept {
    decltype(op(index)) s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index) = i;
      s += contract<D, n + 1, N>::op(index, std::forward<Op>(op));
    }
    return s;
  }
};

template <int D, int N>
struct contract<D, N, N> {
  template <class Index, class Op>
  static constexpr auto op(Index index, Op&& op) noexcept {
    return op(index);
  }
};
}
