#pragma once

#include <tuple>

namespace ttl {
template <int D, int n, int N>
struct forall_impl {
  template <class Index, class Op>
  static constexpr void op(Index index, Op&& op) noexcept {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index) = i;
      forall_impl<D, n + 1, N>::op(index, std::forward<Op>(op));
    }
  }
};

template <int D, int N>
struct forall_impl<D, N, N> {
  template <class Index, class Op>
  static constexpr void op(Index index, Op&& op) noexcept {
    op(index);
  }
};

template <int D, class Index, class Op>
constexpr void forall(Op op) noexcept {
  constexpr int N = std::tuple_size<Index>::value;
  forall_impl<D, 0, N>::op(Index{}, std::move(op));
}

template <int D, int n, int N>
struct contract_impl {
  template <class Index, class Op>
  static constexpr auto op(Index index, Op&& op) noexcept {
    decltype(op(index)) s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index) = i;
      s += contract_impl<D, n + 1, N>::op(index, std::forward<Op>(op));
    }
    return s;
  }
};

template <int D, int N>
struct contract_impl<D, N, N> {
  template <class Index, class Op>
  static constexpr auto op(Index index, Op&& op) noexcept {
    return op(index);
  }
};

template <int D, class Inner, class Outer, class Op>
constexpr auto contract(Outer index, Op op) noexcept {
  auto i = std::tuple_cat(index, Inner{});
  using Union = decltype(i);
  constexpr int n = std::tuple_size<Outer>::value;
  constexpr int N = std::tuple_size<Union>::value;
  return contract_impl<D, n, N>::op(i, std::move(op));
}
} // namespace ttl
