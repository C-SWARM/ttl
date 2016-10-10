// -*- C++ -*-
#ifndef TTL_UTIL_TYPE_AT_H
#define TTL_UTIL_TYPE_AT_H

#include <type_traits>

namespace ttl {
namespace util {
namespace detail {
template <int N, class T, class... U>
struct type_at_impl {
  using type = typename type_at<N - 1, U...>::type;
};

template <class T, class... U>
struct type_at_impl<0, T, U...> {
  using type = T;
};
} // namespace detail

template <class T, class... U>
using type_at = typename detail::type_at_impl<T, U...>::type;

} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_TYPE_AT_H
