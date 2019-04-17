#pragma once

#include "ttl/Expressions/transform.h"
#include "ttl/mp/subset_sequence.hpp"

namespace ttl {
template <class T, class Op, size_t... Is>
constexpr auto apply(T tuple, Op&& op, std::index_sequence<Is...>) noexcept {
  return op(std::get<Is>(tuple)...);
}

template <class... T, class Op>
constexpr auto apply(std::tuple<T...> tuple, Op&& op) noexcept {
  return apply(tuple, std::forward<Op>(op), std::make_index_sequence<sizeof...(T)>());
}

/// Use an index sequence to select a subset of a tuple.
template <class T, class U, size_t... Is>
constexpr T subset(U u, std::index_sequence<Is...>) noexcept {
  return std::make_tuple(std::get<Is>(u)...);
}

/// Select a subset of a tuple based on their types.
///
/// This is implemented by creating an index sequence that acts as a map between
/// the two tuple types. For each type in the output, the index sequence
/// provides the index of the type in the input. We then use a basic fold
/// operation to create the new tuple.
///
/// @tparam           T A tuple type that contains the types we want to select.
/// @tparam           U A tuple type that contains the superset to select from.
///
/// @param            u The tuple with the values we're selecting from.
///
/// @returns            A tuple that contains the type-based subset of U.
template <class T, class U>
constexpr T subset(U u) noexcept {
  return subset<T>(std::move(u), mp::subset_sequence_t<T, U>{});
}

template <class T, class U,
          std::enable_if_t<std::is_same<T, U>::value, void**> = nullptr>
constexpr T select(U from) noexcept {
  return from;
}

template <class T, class U,
          std::enable_if_t<std::is_same<T, U>::value, void**> = nullptr>
constexpr T select(T, U from) noexcept {
  return from;
}

template <class T, class U,
          std::enable_if_t<!std::is_same<T, U>::value, void**> = nullptr>
constexpr T select(U from) noexcept {
  return expressions::transform<T>(from);
}

template <class T, class U,
          std::enable_if_t<!std::is_same<T, U>::value, void**> = nullptr>
constexpr T select(T index, U from) noexcept {
  return expressions::transform(index, from);
}

constexpr bool is_diagonal() noexcept {
  return true;
}

template <class T>
constexpr bool is_diagonal(T car) noexcept {
  return true;
}

template <class T0, class T1, class... U>
constexpr bool is_diagonal(T0 p, T1 car, U... cdr) noexcept {
  return (p == car) and is_diagonal(car, cdr...);
}
}
