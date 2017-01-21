// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_OPERATORS_H
#define TTL_EXPRESSIONS_OPERATORS_H

/// This file binds operators to expressions.
#include <ttl/Expressions/BinaryOp.h>
#include <ttl/Expressions/ScalarOp.h>
#include <ttl/Expressions/UnaryOp.h>

namespace ttl {
namespace expressions {

template <class L, class R>
constexpr const auto operator+(L lhs, R rhs) {
  return AddOp<L, R>(lhs, rhs);
}

template <class L, class R>
constexpr const auto operator-(L lhs, R rhs) {
  return SubtractOp<L, R>(lhs, rhs);
}

template <class L, class R>
constexpr const auto operator/(L lhs, R rhs) {
  return DivideOp<L, R>(lhs, rhs);
}

template <class L, class R>
constexpr const auto operator%(L lhs, R rhs) {
  return ModulusOp<L, R>(lhs, rhs);
}

template <class R>
constexpr const auto operator-(R rhs) {
  return NegateOp<R>(rhs);
}

/// Product needs to select between the scalar multiply and the tensor product,
/// based on the left and right types.
template <class L, class R,
          bool = std::is_arithmetic<L>::value or std::is_arithmetic<R>::value>
struct ProductOp {
  using type = MultiplyOp<L, R>;
};

template <class L, class R>
struct ProductOp<L, R, false> {
  using type = Product<L, R>;
};

template <class L, class R>
constexpr const auto operator*(L lhs, R rhs) {
  return typename ProductOp<L, R>::type(lhs, rhs);
}

} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_OPERATORS_H
