// -*- C++ -*-
#ifndef TTL_UTIL_MULTI_ARRAY_H
#define TTL_UTIL_MULTI_ARRAY_H

namespace ttl {
namespace util {
namespace detail {
template <int R, int D, class S>
struct multi_array_impl
{
  using type = typename multi_array_impl<R-1,D,S>::type[D];
};

template <int D, class S>
struct multi_array_impl<0,D,S>
{
  using type = S;
};
} // namespace detail

template <int R, int D, class S>
using multi_array = typename detail::multi_array_impl<R,D,S>::type;
} // namespace util
} // namespace ttl

#endif // #define TTL_UTIL_MULTI_ARRAY_H

