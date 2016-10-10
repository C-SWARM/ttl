// -*- C++ -*-
#ifndef TTL_UTIL_IIF_H
#define TTL_UTIL_IIF_H

#include <type_traits>

namespace ttl {
namespace util {
namespace detail {
template <class B, class T, class U>
struct iif_impl;

template <class T, class U>
struct iif_impl<std::true_type, T, U>
{
  using type = T;
};

template <class T, class U>
struct iif_impl<std::false_type, T, U>
{
  using type = U;
};
} // namespace detail

template <class B, class T, class U>
using iif = typename detail::iif_impl<B, T, U>::type;

} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_IIF_H
