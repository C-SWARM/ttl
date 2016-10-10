// -*- C++ -*-
#ifndef TTL_UTIL_INDEX_OF_H
#define TTL_UTIL_INDEX_OF_H

namespace ttl {
namespace util {
namespace detail {
/// Get the index of a type in a pack.
///
/// @tparam           n The current index.
/// @tparam           T The type that we're searching for.
/// @tparam...        U The pack that we're searching.
template <int n, class T, class... U>
struct index_of_impl;

/// Peel off the first right-hand-side type if it doesn't match T.
///
/// @tparam           n The current index.
/// @tparam           T The type that we're searching for.
/// @tparam          U0 The next type in the pack (doesn't match T)
/// @tparam...        U The pack that we're searching.
template <int n, class T, class U0, class... U>
struct index_of_impl<n, T, U0, U...> {
  using type = typename index_of_impl<n + 1, T, U...>::type;
};

/// If T is the first type in the right-hand-side, we found it.
///
/// @tparam           N The current index.
/// @tparam           T The type that we're searching for, and the first type in
///                     the pack.
/// @tparam...        U The pack that we're searching.
template <int N, class T, class... U>
struct index_of_impl<N, T, T, U...> {
  using type = std::integral_constant<int, N>;
};
} // namespace detail

/// Get the index of a type in a parameter pack.
///
/// @tparam           T The type that we're searching for.
/// @tparam...        U The pack that we're searching.
template <class T, class... U>
using index_of = typename detail::index_of_impl<0, T, U...>::type;

} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_INDEX_OF_H
