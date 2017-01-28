// -*- C++ -*-
#ifndef TTL_LIBRARY_TRANSPOSE_H
#define TTL_LIBRARY_TRANSPOSE_H

#include <ttl/Expressions/traits.h>

namespace ttl {
namespace detail {
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

} // namespace detail

template <class E>
auto transpose(E t) {
  using Outer = expressions::outer_type<E>;
  using Type = detail::reverse<Outer>;
  return t.to(Type());
}
}// namespace ttl

#endif // #define TTL_LIBRARY_TRANSPOSE_H

