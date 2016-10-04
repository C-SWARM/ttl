// -*- C++ -*-
#ifndef TTL_PACK_H
#define TTL_PACK_H

/// Simple utility to deal with index packs.
#include <ttl/detail/contains.h>
#include <ttl/detail/iif.h>
#include <type_traits>

namespace ttl {
template <class... T>
struct Pack {
  using type = Pack<T...>;
};

namespace detail {
template <class T>
struct is_empty_impl;

template <class... T>
struct is_empty_impl<Pack<T...>> {
  static constexpr bool value = false;
};

template <>
struct is_empty_impl<Pack<>> {
  static constexpr bool value = true;
};
} // namespace detail

template <class T>
struct is_empty;

template <class... T>
struct is_empty<Pack<T...>> {
  static constexpr bool value = detail::is_empty_impl<Pack<T...>>::value;
};

template <template <class, class> class op, class... T, class... U>
struct is_empty<op<Pack<T...>, Pack<U...>>> {
  using type_ = typename op<Pack<T...>, Pack<U...>>::type;
  static constexpr bool value = detail::is_empty_impl<type_>::value;
};

namespace detail {

/// Check to see if two packs have equivalent type.
///
/// @{
template <class T, class U, class V>
struct equiv_impl;

/// Base case when we processed all of the types.
template <class... U>
struct equiv_impl<Pack<>, Pack<>, Pack<U...>> {
  static constexpr bool value = true;
};

/// Process the next right-hand-side type.
template <class U0, class... U, class... V>
struct equiv_impl<Pack<>, Pack<U0, U...>, Pack<V...>> {
 private:
  using found_ = typename contains<U0, V...>::type;
  using next_ = equiv_impl<Pack<>, Pack<U...>, Pack<V...>>;

 public:
  static constexpr bool value = iif<found_, next_, std::false_type>::value;
};

/// Process the next left-hand-side type.
template <class T0, class... T, class... U, class... V>
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
} // namespace detail

template <class T, class U>
struct is_equivalent {
  static constexpr bool value = detail::equiv<T, U>::value;
};

template <class T, class U>
struct is_equal {
  static constexpr bool value = std::is_same<T, U>::value;
};

} // namespace ttl

#endif // #ifndef TTL_PACK_H
