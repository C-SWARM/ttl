// -*- C++ -*-
#ifndef TTL_LIBRARY_BINDER_H
#define TTL_LIBRARY_BINDER_H

#include <ttl/Index.h>
#include <ttl/Expressions/Bind.h>
#include <tuple>

namespace ttl {
namespace lib {
template <char N>
constexpr auto binder() {
  return std::tuple_cat(std::tuple<Index<N>>(), binder<N-1>());
}

template <>
constexpr auto binder<1>() {
  return std::tuple<Index<1>>();
}

template <class E>
constexpr auto bind(E&& e) {
  using namespace ttl::expressions;
  return make_bind(std::forward<E>(e), binder<rank<E>::value>());
}
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_BINDER_H
