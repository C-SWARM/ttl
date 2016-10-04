// -*- C++ -*-
#ifndef TTL_DETAIL_XOR_TYPE_H
#define TTL_DETAIL_XOR_TYPE_H

#include <ttl/Pack.h>
#include <ttl/detail/contains.h>
#include <ttl/detail/iif.h>
#include <ttl/detail/and_type.h>
#include <ttl/detail/or_type.h>

namespace ttl {
namespace detail {

/// Implement the metaprogramming operation that will create the symmetric
/// difference in two lists of types (represented by some packed
/// object---std::tuple in ttl).
///
/// @{
template <class T, class U, class... V>
struct xor_type_impl;

/// Base case is when we've processed the union.
template <class... U, class... V>
struct xor_type_impl<Pack<>, Pack<U...>, V...> {
  using type = Pack<V...>;
};

/// Process the next type from the union.
///
/// 1. If T0 is not in the intersection, V..., then add T0 to the intersection.
/// 2. Continue processing T....
template <class T0, class... T, class... U, class... V>
struct xor_type_impl<Pack<T0, T...>, Pack<U...>, V...> {
 private:
  using lhs_ = Pack<T...>;
  using rhs_ = Pack<U...>;
  using found_ = typename contains<T0, U...>::type;
  using retain_ = xor_type_impl<lhs_, rhs_, V..., T0>;
  using drop_ = xor_type_impl<lhs_, rhs_, V...>;

 public:
  using type = typename iif<found_, drop_, retain_>::type;
};

template <class T, class U>
struct xor_type {
 private:
  using Intersection_ = typename and_type<T, U>::type;
  using Union_ = typename or_type<T, U>::type;

 public:
  /// The xor type is a pack of those types that are in the union of the packs T
  /// and U but not in their intersection.
  using type = typename xor_type_impl<Union_, Intersection_>::type;
};

/// @}

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_XOR_TYPE_H
