// -*- C++ -*-
#ifndef TTL_LIBRARY_MATRIX_H
#define TTL_LIBRARY_MATRIX_H

#include <ttl/Expressions/traits.h>
#include <ttl/util/pow.h>
#include <ttl/util/log2.h>

namespace ttl {
namespace lib {
template <class E>
constexpr int matrix_dimension() {
  using namespace ttl::expressions;
  using namespace ttl::util;
  return pow(dimension<E>::value, log2<rank<E>::value>::value);
};
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_MATRIX_H
