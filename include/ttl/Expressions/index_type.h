// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_INDEX_TYPE_H
#define TTL_EXPRESSIONS_INDEX_TYPE_H

#include <ttl/util/contains.h>
#include <ttl/util/iif.h>

namespace ttl {
namespace expressions {
namespace detail {

//------------------------------------------------------------------------------
// Implement the metaprogramming operation that will intersect two packs.
//------------------------------------------------------------------------------
/// @{
template <class T, class U, class... V>
struct intersection_impl;

/// Base case when we processed all of the types in the left-hand-side.
template <template <class...> class pack, class... U, class... V>
struct intersection_impl<pack<>, pack<U...>, V...>
{
  using type = pack<V...>;
};

/// Process the next left-hand-side type.
///
/// 1) If T0 is in the pack U..., then add T0 to the intersection V....
/// 2) Continue processing T....
template <template <class...> class pack, class T0, class... T, class... U,
          class... V>
struct intersection_impl<pack<T0, T...>, pack<U...>, V...>
{
  using found = typename util::contains<T0, U...>::type;
  using retain = intersection_impl<pack<T...>, pack<U...>, V..., T0>;
  using drop = intersection_impl<pack<T...>, pack<U...>, V...>;
  using type = typename util::iif<found, retain, drop>::type;
};
/// @}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Implement the metaprogramming operation that will compute the set V such
// that each v is in T or U.
//------------------------------------------------------------------------------
/// @{
template <class T, class U, class... V>
struct union_impl;

/// Base case when we've exhausted all of the packed types.
template <template <class...> class pack, class... V>
struct union_impl<pack<>, pack<>, V...>
{
  using type = pack<V...>;
};

/// Reverse the left and right hand side when there are no left hand type but at
/// least one right hand side.
template <template <class...> class pack, class U0, class... U, class... V>
struct union_impl<pack<>, pack<U0, U...>, V...>
{
  using type = typename union_impl<pack<U0, U...>, pack<>, V...>::type;
};

/// Process the next left-hand-side type.
///
/// 1) If T0 is not in the union V..., then add it to V....
/// 2) Continue processing T....
template <template <class...> class pack, class T0, class... T, class... U,
          class... V>
struct union_impl<pack<T0, T...>, pack<U...>, V...>
{
  using found = typename util::contains<T0, V...>::type;
  using retain = union_impl<pack<T...>, pack<U...>, V..., T0>;
  using drop = union_impl<pack<T...>, pack<U...>, V...>;
  using type = typename util::iif<found, drop, retain>::type;
};
/// @}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Implement the metaprogramming operation that will compute difference the set
// V such that each v is in T but not in U.
//------------------------------------------------------------------------------
/// @{
//------------------------------------------------------------------------------
template <class T, class U, class... V>
struct difference_impl;

/// Base case is when we've processed the left hand side.
template <template <class...> class pack, class... U, class... V>
struct difference_impl<pack<>, pack<U...>, V...>
{
  using type = pack<V...>;
};

/// Process the next type from the union.
///
/// 1. If T0 is not in the intersection, V..., then add T0 to the intersection.
/// 2. Continue processing T....
template <template <class...> class pack, class T0, class... T, class... U,
          class... V>
struct difference_impl<pack<T0, T...>, pack<U...>, V...>
{
  using found = typename util::contains<T0, U...>::type;
  using retain = difference_impl<pack<T...>, pack<U...>, V..., T0>;
  using drop = difference_impl<pack<T...>, pack<U...>, V...>;
  using type = typename util::iif<found, drop, retain>::type;
};
/// @}
//------------------------------------------------------------------------------

} // namespace detail

template <class T, class U>
using intersection = typename detail::intersection_impl<
  typename std::remove_reference<T>::type,
  typename std::remove_reference<U>::type>::type;

template <class T, class U>
using union_ = typename detail::union_impl<
  typename std::remove_reference<T>::type,
  typename std::remove_reference<U>::type>::type;

template <class T, class U>
using difference = typename detail::difference_impl<
  typename std::remove_reference<T>::type,
  typename std::remove_reference<U>::type>::type;

// helper to make defining the outer type easier
template <class T, class U>
using outer_type = difference<union_<T, U>, intersection<T,U>>;

} // namespace epressions
} // namespace ttl

#endif // #ifndef TTL_UTIL_EXPRESSIONS_TYPE__H
