// -*- C++ -*-
#ifndef TTL_LIBRARY_BINDER_H
#define TTL_LIBRARY_BINDER_H

#include <ttl/Index.h>
#include <ttl/Expressions/Bind.h>
#include <ttl/Expressions/pack.h>
#include <tuple>

namespace ttl {
namespace lib {
template <char N>
struct bind_impl
{
  static_assert(N>0);
  using next = typename bind_impl<N-1>::type;
  using type = expressions::concat<std::tuple<Index<N>>, next>;
};

template <>
struct bind_impl<1>
{
  using type = std::tuple<Index<1>>;
};

template <class E>
constexpr auto bind(E&& e) {
  using namespace ttl::expressions;
  return Bind<E,typename bind_impl<rank<E>::value>::type>(std::forward<E>(e));
}
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_BINDER_H
