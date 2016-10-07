// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_SCALAR_OP_H
#define TTL_EXPRESSIONS_SCALAR_OP_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>
#include <ttl/detail/iif.h>
#include <functional>
#include <type_traits>

namespace ttl {
namespace expressions {
/// ScalarOp represents a multiplication between a scalar and an expression.
///
/// Either the left-hand-side or the right-hand-side might be a scalar. The
/// scalar operation must preserve this ordering.
///
/// Ops supported: *, /
template <class Op, class L, class R, bool = std::is_arithmetic<L>::value>
class ScalarOp;

/// The expression Traits for ScalarOp expressions.
///
/// The scalar op just exports its Expression's Traits, with the scalar type
/// adjusted as necessary.
template <class Op, class L, class R>
struct Traits<ScalarOp<Op, L, R, true>> : public Traits<R> {
  using ScalarType = promote<L, R>;
};

template <class Op, class L, class R>
struct Traits<ScalarOp<Op, L, R, false>> : public Traits<L> {
  using ScalarType = promote<L, R>;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the left-hand-side
/// operation.
template <template <class> class Op, class L, class R>
class ScalarOp<Op<L>, L, R, true> : Expression<ScalarOp<Op<L>, L, R, true>>
{
 public:
  ScalarOp(L lhs, R rhs) : lhs_(lhs), rhs_(rhs), op_() {
  }

  constexpr auto operator[](IndexSet<Traits<ScalarOp>::Rank> i) const
    -> typename Traits<ScalarOp>::ScalarType
  {
    return op_(lhs_, rhs_[i]);
  }

 private:
  L lhs_;
  R rhs_;
  Op<typename Traits<ScalarOp>::ScalarType> op_;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the right-hand-side
/// operation.
template <template <class> class Op, class L, class R>
class ScalarOp<Op<R>, L, R, false> : Expression<ScalarOp<Op<R>, L, R, false>>
{
 public:
  ScalarOp(L lhs, R rhs) : lhs_(lhs), rhs_(rhs), op_() {
  }

  constexpr auto operator[](IndexSet<Traits<ScalarOp>::Rank> i) const
    -> typename Traits<ScalarOp>::ScalarType
  {
    return op_(lhs_[i], rhs_);
  }

 private:
  L lhs_;
  R rhs_;
  Op<typename Traits<ScalarOp>::ScalarType> op_;
};

template <class L, class R>
constexpr auto operator/(L lhs, R rhs)
  -> ScalarOp<std::divides<promote<L, R>>, L, R>
{
  return ScalarOp<std::divides<promote<L, R>>, L, R>(lhs, rhs);
}

template <class L, class R>
constexpr auto operator%(L lhs, R rhs)
  -> ScalarOp<std::modulus<promote<L, R>>, L, R>
{
  return ScalarOp<std::modulus<promote<L, R>>, L, R>(lhs, rhs);
}
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_SCALAR_OP_H
