#pragma once

#include "ttl/Expressions/transform.h"
#include "ttl/mp/subset_sequence.hpp"

namespace ttl {
template <class T, class Op, size_t... Is>
constexpr auto apply(T tuple, Op&& op, std::index_sequence<Is...>) {
  return op(std::get<Is>(tuple)...);
}

template <class... T, class Op>
constexpr auto apply(std::tuple<T...> tuple, Op&& op) {
  return apply(tuple, std::forward<Op>(op), std::make_index_sequence<sizeof...(T)>());
}

template <class T, class U, size_t... Is>
constexpr T subset(U u, std::index_sequence<Is...>) noexcept {
  return std::make_tuple(std::get<Is>(u)...);
}

template <class T, class U>
constexpr T subset(U u) noexcept {
  return subset<T>(std::move(u), mp::subset_sequence_t<T, U>{});
}

template <class T, class U>
constexpr T select(U from) noexcept {
  return expressions::transform<T>(from);
}

template <class T, class U>
constexpr T select(T index, U from) noexcept {
  return expressions::transform(index, from);
}
}
