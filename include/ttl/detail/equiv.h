// -*- C++ -*-
#ifndef TTL_DETAIL_EQUIV_H
#define TTL_DETAIL_EQUIV_H

#include <ttl/detail/contains.h>
#include <ttl/detail/iif.h>

namespace ttl {
namespace detail {

/// Check to see if two packs have equivalent type.
///
/// @{
template <class T, class U, class V>
struct equiv_impl;

/// Base case when we processed all of the types.
template <template <class...> class Pack, class... U>
struct equiv_impl<Pack<>, Pack<>, Pack<U...>> {
  static constexpr bool value = true;
};

/// Process the next right-hand-side type.
template <template <class...> class Pack, class U0, class... U, class... V>
struct equiv_impl<Pack<>, Pack<U0, U...>, Pack<V...>> {
 private:
  using found_ = typename contains<U0, V...>::type;
  using next_ = equiv_impl<Pack<>, Pack<U...>, Pack<V...>>;

 public:
  static constexpr bool value = iif<found_, next_, std::false_type>::value;
};

/// Process the next left-hand-side type.
template <template <class...> class Pack, class T0, class... T, class... U,
          class... V>
struct equiv_impl<Pack<T0, T...>, Pack<U...>, Pack<V...>> {
 private:
  using found_ = typename contains<T0, U...>::type;
  using next_ = equiv_impl<Pack<T...>, Pack<U...>, Pack<V...>>;

 public:
  static constexpr bool value = iif<found_, next_, std::false_type>::value;
};

template <class T, class U>
struct equiv {
 private:
  using L_ = typename std::remove_reference<T>::type;
  using R_ = typename std::remove_reference<U>::type;

 public:
  static constexpr bool value = equiv_impl<L_, R_, L_>::value;
};

/// @}

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_EQUIV_H
