// -*- C++ -*-
#ifndef TTL_DETAIL_POW_H
#define TTL_DETAIL_POW_H

namespace ttl {
namespace detail {
template <typename T>
constexpr T pow(const T base, const T exp, const T accum = 1) {
  return (exp) ? pow(base, exp - 1, accum * base) : accum;
}
} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_POW_H
