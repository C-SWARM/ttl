// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_PROMOTE_H
#define TTL_EXPRESSIONS_PROMOTE_H

#include <ttl/Expressions/Expression.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {

/// Template for promoting scalar types.
///
/// We use multiplication as the default promotion operator. This might not be
/// the best choice, but we're going with it for now.
template <class L, class R,
          bool = std::is_arithmetic<L>::value,
          bool = std::is_arithmetic<R>::value>
struct promote_impl;
}

template <class L, class R>
using promote = typename detail::promote_impl<L, R>::type;

namespace detail {
template <class L, class R>
struct promote_impl<L, R, true, true>
{
  using type = decltype(L() * R());             // both scalars
};

template <class L, class R>
struct promote_impl<L, R, true, false>
{
  using type = promote<L, scalar_type<R>>;
};

template <class L, class R>
struct promote_impl<L, R, false, true>
{
  using type = promote<scalar_type<L>, R>;
};

template <class L, class R>
struct promote_impl<L, R, false, false>
{
  using type = promote<scalar_type<L>, scalar_type<R>>;
};
} // namespace detail

} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_PROMOTE_H
