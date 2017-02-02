// -*- C++ -*-
#ifndef TTL_LIBRARY_FP_UTILS_H
#define TTL_LIBRARY_FP_UTILS_H

namespace ttl {
namespace lib {
constexpr bool FPNEZ(double d, double e = 1e-10) {
  return (d > e) || (-e > d);
}
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_FP_UTILS_H
