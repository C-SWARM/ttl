// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_IS_SUBSET_H
#define TTL_EXPRESSIONS_IS_SUBSET_H

#include <ttl/util/contains.h>
#include <ttl/util/iif.h>
#include <type_traits>

/// Metafunction to see if one parameter pack is a subset of another parameter
/// pack.
///
/// @code
///   T... lhs
///   U... rhs
///   if (is_subset<std::tuple<T...>, std::tuple<U...>>:value) ...
/// @code

namespace ttl {
namespace util {
namespace detail {
template <class T, class U>
struct is_subset_impl;

template <template <class...> class pack, class V>
struct is_subset_impl<pack<>, V>
{
  using type = std::true_type;
};

template <template <class...> class pack, class U0, class... U, class... V>
struct is_subset_impl<pack<U0, U...>, pack<V...>>
{
  using found = typename contains<U0, V...>::type;
  using next = is_subset_impl<pack<U...>, pack<V...>>;
  using type = typename iif<found, next, std::false_type>::type;
};
} // namespace detail

template <class T, class U>
using is_subset = typename detail::is_subset_impl<T,U>::type;

} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_IS_SUBSET_H
