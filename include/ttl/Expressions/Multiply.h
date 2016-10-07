// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_MULTIPLY_H
#define TTL_EXPRESSIONS_MULTIPLY_H

#include <ttl/Expressions/ScalarOp.h>
#include <ttl/Expressions/TensorProduct.h>

/// We use the operator* operator for both scalar multiplication and tensor
/// product, depending on the types of the left and right hand side. This file
/// contains the multiply_impl<> metaprogramming to disambiguate this context
/// and map to the correct one.

namespace ttl {
namespace expressions {

/// scalar multiply
template <class L, class R,
          bool = std::is_arithmetic<L>::value or std::is_arithmetic<R>::value>
struct multiply_impl {
  using type = ScalarOp<std::multiplies<promote<L, R>>, L, R>;
  static constexpr type op(L lhs, R rhs) {
    return ScalarOp<std::multiplies<promote<L, R>>, L, R>(lhs, rhs);
  }
};

/// tensor product specialization
template <class L, class R>
struct multiply_impl<L, R, false> {
  using type = TensorProduct<L, R>;
  static constexpr type op(L lhs, R rhs) {
    return TensorProduct<L, R>(lhs, rhs);
  }
};

template <class L, class R>
constexpr auto operator*(L lhs, R rhs)
  -> typename multiply_impl<L, R>::type
{
  return multiply_impl<L, R>::op(lhs, rhs);
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_MULTIPLY_H
