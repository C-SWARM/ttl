// -*- C++ -*-
#ifndef TTL_DETAIL_INDEX_OF_H
#define TTL_DETAIL_INDEX_OF_H

namespace ttl {
namespace detail {

/// Get the index of a type in a pack.
///
/// @tparam           N The current index.
/// @tparam           T The type that we're searching for.
/// @tparam...        U The pack that we're searching.
template <int N, class T, class... U>
struct indexof_impl;

/// Peel off the first right-hand-side type if it doesn't match T.
///
/// @tparam           N The current index.
/// @tparam           T The type that we're searching for.
/// @tparam          U0 The next type in the pack (doesn't match T)
/// @tparam...        U The pack that we're searching.
template <int N, class T, class U0, class... U>
struct indexof_impl<N, T, U0, U...> {
  static constexpr int value = indexof_impl<N + 1, T, U...>::value;
};

/// If T is the first type in the right-hand-side, we found it.
///
/// @tparam           N The current index.
/// @tparam           T The type that we're searching for, and the first type in
///                     the pack.
/// @tparam...        U The pack that we're searching.
template <int N, class T, class... U>
struct indexof_impl<N, T, T, U...> {
  static constexpr int value = N;
};

/// If we exhaust the right-hand-side types we haven't found it.
///
/// @tparam           N The current index.
/// @tparam           T The type that we're searching for.
template <int N, class T>
struct indexof_impl<N, T> {
  static constexpr int value = -1;
};

/// Get the index of a type in a parameter pack.
///
/// @tparam           T The type that we're searching for.
/// @tparam...        U The pack that we're searching.
template <class T, class... U>
using indexof = indexof_impl<0, T, U...>;

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_INDEX_OF_H
