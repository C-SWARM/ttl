// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TRANSFORM_H
#define TTL_EXPRESSIONS_TRANSFORM_H

#include <ttl/util/index_of.h>
#include <ttl/util/is_subset.h>
#include <tuple>

namespace ttl {
namespace expressions {
namespace detail {
/// This core template transforms an index from one space to another space.
template <class To, class From>
struct transform_impl;

/// Specialization for when the space we're transforming to has no more indices
/// to match.
template <class From>
struct transform_impl<std::tuple<>, From>
{
  static constexpr std::tuple<> op(From) {
    return std::tuple<>();
  }
};

/// Match the head index on the left hand side with an index on the right hand
/// side and recursively transform the tail of the left hand side.
template <class T, class... To, class... From>
struct transform_impl<std::tuple<T, To...>, std::tuple<From...>>
{
  static constexpr std::tuple<T, To...> op(std::tuple<From...> from) {
    return std::tuple_cat(head(from), tail::op(from));
  }

 private:
  static_assert(util::is_subset<std::tuple<To...>, std::tuple<From...>>::value,
                "Index space is incompatible");
  static constexpr std::tuple<T> head(std::tuple<From...> from) {
    return std::tuple<T>(std::get<util::index_of<T, From...>::value>(from));
  }

  using tail = transform_impl<std::tuple<To...>, std::tuple<From...>>;
};
} // namespace detail

template <class To, class From>
constexpr To transform(From from) {
  return detail::transform_impl<To, From>::op(from);
}

} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_TRANSFORM_H
