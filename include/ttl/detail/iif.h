// -*- C++ -*-
#ifndef TTL_DETAIL_IIF_H
#define TTL_DETAIL_IIF_H

#include <type_traits>

namespace ttl {
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

template <class B, class T, class U>
using iif = typename iif_impl<B, T, U>::type;

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_IIF_H
