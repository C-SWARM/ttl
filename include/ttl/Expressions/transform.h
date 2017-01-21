// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TRANSFORM_H
#define TTL_EXPRESSIONS_TRANSFORM_H

#include <ttl/Expressions/pack.h>
#include <tuple>

namespace ttl {
namespace expressions {
namespace detail {

/// This core template transforms an index from one space to another space.
template <class To, class From>
struct transform_impl;

/// Base case when the space we're transforming to has no more indices to match.
template <template <class...> class Pack, class From>
struct transform_impl<Pack<>, From>
{
  static constexpr Pack<> op(From) {
    return Pack<>();
  }
};

/// Match the head index on the left hand side with an index on the right hand
/// side and recursively transform the tail of the left hand side.
template <class To, class From>
struct transform_impl
{
  static constexpr To op(From from) {
    return std::tuple_cat(head(from), tail::op(from));
  }

 private:
  static_assert(subset<To, From>::value, "Index space is incompatible");

  static constexpr car<To> head(From from) {
    return car<To>(std::get<index_of<car<To>, From>::value>(from));
  }

  using tail = transform_impl<cdr<To>, From>;
};
} // namespace detail

template <class To, class From>
constexpr To transform(From from) {
  return detail::transform_impl<To, From>::op(from);
}

} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_TRANSFORM_H
