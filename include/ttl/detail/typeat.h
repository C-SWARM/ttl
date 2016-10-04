// -*- C++ -*-
#ifndef TTL_DETAIL_TYPEAT_H
#define TTL_DETAIL_TYPEAT_H

#include <type_traits>

namespace ttl {
namespace detail {

template <int N, class T, class... U>
struct typeat {
  using type = typename typeat<N - 1, U...>::type;
};

template <class T, class... U>
struct typeat<0, T, U...> {
  using type = T;
};

} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_TYPEAT_H
