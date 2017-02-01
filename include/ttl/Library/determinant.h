// -*- C++ -*-
#ifndef TTL_LIBRARY_DETERMINANT_H
#define TTL_LIBRARY_DETERMINANT_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>

namespace ttl {
namespace lib {
template <class E,
          int R = expressions::rank<E>::value,
          int D = expressions::dimension<E>::value>
struct det_impl;

/// Analytical determinant for 2x2
template <class E>
struct det_impl<E, 2, 2>
{
  static constexpr auto op(E f) {
    return f(0,0)*f(1,1) - f(0,1)*f(1,0);
  }
};

template <class E>
struct det_impl<E, 2, 3>
{
  static auto op(E f) {
    auto t0 = f(0,0)*f(1,1)*f(2,2);
    auto t1 = f(1,0)*f(2,1)*f(0,2);
    auto t2 = f(2,0)*f(0,1)*f(1,2);
    auto s0 = f(0,0)*f(1,2)*f(2,1);
    auto s1 = f(1,1)*f(2,0)*f(0,2);
    auto s2 = f(2,2)*f(0,1)*f(1,0);
    return (t0 + t1 + t2) - (s0 + s1 + s2);
  }
};
} // namespace lib

template <class E>
constexpr expressions::scalar_type<E> det(E e) {
  return lib::det_impl<E>::op(e);
}

template <int D, class S>
constexpr auto det(const Tensor<2,D,S>& matrix) {
  return det(lib::bind(matrix));
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_DETERMINANT_H
