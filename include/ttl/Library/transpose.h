// -*- C++ -*-
#ifndef TTL_LIBRARY_TRANSPOSE_H
#define TTL_LIBRARY_TRANSPOSE_H

#include <ttl/config.h>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>

namespace ttl {
namespace lib {
template <class T, class Pack> struct
push_back_impl;

template <template <class...> class Pack, class T, class... U>
struct push_back_impl<T, Pack<U...>> {
  using type = Pack<U..., T>;
};

template <class T, class Pack>
using push_back = typename push_back_impl<T, Pack>::type;

template <class Pack>
struct reverse_impl;

template <template <class...> class Pack>
struct reverse_impl<Pack<>>
{
  using type = Pack<>;
};

template <template <class...> class Pack, class T0, class... T>
struct reverse_impl<Pack<T0, T...>>
{
  using type = push_back<T0, typename reverse_impl<Pack<T...>>::type>;
};

template <class Pack>
using reverse = typename reverse_impl<Pack>::type;
} // namespace lib

template <class E>
constexpr auto transpose(E t) {
  using Outer = expressions::outer_type<E>;
  using Type = lib::reverse<Outer>;
  return expressions::Bind<E,Type>(t);
}

template <int R, int D, class S>
constexpr auto transpose(const Tensor<R,D,S>& t) {
  return transpose(lib::bind(t));
}

}// namespace ttl

#endif // #define TTL_LIBRARY_TRANSPOSE_H

