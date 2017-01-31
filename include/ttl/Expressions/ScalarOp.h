// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_SCALAR_OP_H
#define TTL_EXPRESSIONS_SCALAR_OP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/promote.h>
#include <functional>
#include <type_traits>

namespace ttl {
namespace expressions {
/// ScalarOp represents a multiplication between a scalar and an expression.
///
/// Either the left-hand-side or the right-hand-side might be a scalar. The
/// scalar operation must preserve this ordering.
///
/// Ops supported: *, /, %
template <class Op, class L, class R, bool = std::is_arithmetic<L>::value>
class ScalarOp;

/// The expression Traits for ScalarOp expressions.
///
/// The scalar op just exports its Expression's Traits, with the scalar type
/// adjusted as necessary.
template <class Op, class L, class R>
struct traits<ScalarOp<Op, L, R, true>> : public traits<R>
{
  using scalar_type = promote<L, R>;
};

template <class Op, class L, class R>
struct traits<ScalarOp<Op, L, R, false>> : public traits<L>
{
  using scalar_type = promote<L, R>;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the left-hand-side
/// operation.
template <class Op, class L, class R>
class ScalarOp<Op, L, R, true> : public  Expression<ScalarOp<Op, L, R, true>>
{
  static_assert(is_expression_t<R>::value, "Operand is not Expression");
 public:
  constexpr ScalarOp(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(lhs_, rhs_.eval(index));
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the right-hand-side
/// operation.
template <class Op, class L, class R>
class ScalarOp<Op, L, R, false> : public Expression<ScalarOp<Op, L, R, false>>
{
  static_assert(is_expression_t<L>::value, "Operand is not Expression");

 public:
  constexpr ScalarOp(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(lhs_.eval(index), rhs_);
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

template <class L, class R>
using DivideOp = ScalarOp<std::divides<promote<L, R>>, L, R>;

template <class L, class R>
using ModulusOp = ScalarOp<std::modulus<promote<L, R>>, L, R>;

template <class L, class R>
using MultiplyOp = ScalarOp<std::multiplies<promote<L, R>>, L, R>;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_SCALAR_OP_H
