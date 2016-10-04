// -*- C++ -*-
#ifndef TTL_DETAIL_AND_TYPE_H
#define TTL_DETAIL_AND_TYPE_H

#include <ttl/detail/contains.h>
#include <ttl/detail/iif.h>

namespace ttl {
namespace detail {

/// Implement the metaprogramming operation that will intersect two lists of
/// types (represented by some packed object---std::tuple in ttl).
///
/// @{
template <class T, class U, class... V>
struct and_type_impl;

/// Base case when we processed all of the left-hand-side types.
template <template <class...> class Pack, class... U, class... V>
struct and_type_impl<Pack<>, Pack<U...>, V...> {
  using type = Pack<V...>;
};

/// Process the next left-hand-side type.
///
/// 1) If T0 is in the pack U..., then add T0 to the intersection V....
/// 2) Continue processing T....
template <template <class...> class Pack, class T0, class... T, class... U,
          class... V>
struct and_type_impl<Pack<T0, T...>, Pack<U...>, V...> {
 private:
  using lhs_ = Pack<T...>;
  using rhs_ = Pack<U...>;
  using found_ = typename contains<T0, U...>::type;
  using retain_ = and_type_impl<lhs_, rhs_, V..., T0>;
  using drop_ = and_type_impl<lhs_, rhs_, V...>;

 public:
  using type = typename iif<found_, retain_, drop_>::type;
};

template <class T, class U>
struct and_type {
 private:
  using L_ = typename std::remove_reference<T>::type;
  using R_ = typename std::remove_reference<U>::type;

 public:
  using type = typename and_type_impl<L_, R_>::type;
};

/// @}

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_AND_TYPE_H
