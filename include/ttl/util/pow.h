// -*- C++ -*-
#ifndef TTL_UTIL_POW_H
#define TTL_UTIL_POW_H

namespace ttl {
namespace util {
template <class T>
constexpr T pow(T base, T exp) {
  return (exp > 0) ? base * pow(base, exp - 1) : 1;
}
} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_POW_H
