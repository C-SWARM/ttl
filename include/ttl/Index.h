// -*- C++ -*-
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

#include <ttl/detail/and_type.h>
#include <ttl/detail/or_type.h>
#include <ttl/detail/xor_type.h>
#include <ttl/detail/equiv.h>

namespace ttl {
template <char ID>
struct Index {
  static constexpr char id = ID;
};

template <class... T>
struct IndexPack {
};

template <class... T>
inline constexpr IndexPack<T...>
index_list_pack(T...) {
  return IndexPack<T...>();
}

template <class T, class U>
inline constexpr typename detail::and_type<T, U>::type
index_list_and(T&&, U&&) {
  using type = typename detail::and_type<T, U>::type;
  return type();
}

template <class T, class U>
inline constexpr typename detail::or_type<T, U>::type
index_list_or(T&&, U&&) {
  using type = typename detail::or_type<T, U>::type;
  return type();
}

template <class T, class U>
inline constexpr typename detail::xor_type<T, U>::type
index_list_xor(T&&, U&&) {
  using type = typename detail::xor_type<T, U>::type;
  return type();
}

template <class T, class U>
inline constexpr bool index_list_eq(T&&, U&&) {
  return std::is_same<T, U>::value;
}

template <class T, class U>
inline constexpr bool index_list_equiv(T&&, U&&) {
  return detail::equiv<T, U>::value;
}
} // namespace ttl

#endif // TTL_INDEX_H
