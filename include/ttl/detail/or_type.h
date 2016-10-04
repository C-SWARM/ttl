// -*- C++ -*-
#ifndef TTL_DETAIL_OR_TYPE_H
#define TTL_DETAIL_OR_TYPE_H

#include <ttl/Pack.h>
#include <ttl/detail/contains.h>
#include <ttl/detail/iif.h>

namespace ttl {
namespace detail {

/// Implement the metaprogramming operation that will compute the union of two
/// lists of types (represented by some packed object---std::tuple in ttl).
///
/// @{
template <class T, class U, class... V>
struct or_type_impl;

/// Base case when we've exhausted all of the packed types.
template <class... V>
struct or_type_impl<Pack<>, Pack<>, V...> {
  using type = Pack<V...>;
};

/// Process the next right-hand-side type
///
/// 1) If U0 is not in the union V..., then add it to V....
/// 2) Continue processing U....
template <class U0, class... U, class... V>
struct or_type_impl<Pack<>, Pack<U0, U...>, V...> {
 private:
  using lhs_ = Pack<>;
  using rhs_ = Pack<U...>;
  using found_ = typename contains<U0, V...>::type;
  using retain_ = or_type_impl<lhs_, rhs_, V..., U0>;
  using drop_ = or_type_impl<lhs_, rhs_, V...>;

 public:
  using type = typename iif<found_, drop_, retain_>::type;
};

/// Process the next left-hand-side type.
///
/// 1) If T0 is not in the union V..., then add it to V....
/// 2) Continue processing T....
template <class T0, class... T, class... U, class... V>
struct or_type_impl<Pack<T0, T...>, Pack<U...>, V...> {
 private:
  using lhs_ = Pack<T...>;
  using rhs_ = Pack<U...>;
  using found_ = typename contains<T0, V...>::type;
  using retain_ = or_type_impl<lhs_, rhs_, V..., T0>;
  using drop_ = or_type_impl<lhs_, rhs_, V...>;

 public:
  using type = typename iif<found_, drop_, retain_>::type;
};

template <class T, class U>
struct or_type {
 private:
  using L_ = typename std::remove_reference<T>::type;
  using R_ = typename std::remove_reference<U>::type;

 public:
  using type = typename or_type_impl<L_, R_>::type;
};

/// @}

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_OR_TYPE_H
