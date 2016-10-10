// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_UNARY_OP_H
#define TTL_EXPRESSIONS_UNARY_OP_H

#include <ttl/Expressions/Expression.h>
#include <functional>

namespace ttl {
namespace expressions {
/// UnaryOp represents a unary operation on an expression.
///
/// @tparam           E The Expression.
template <class Op, class E>
class UnaryOp;

/// The expression Traits for UnaryOp expressions.
///
/// This just exports the traits of the underlying expression.
template <class Op, class E>
struct expression_traits<UnaryOp<Op, E>> : public expression_traits<E>
{
};

template <class Op, class E>
class UnaryOp : Expression<UnaryOp<Op, E>>
{
 public:
  UnaryOp(E e) : e_(e), op_() {
  }

  template <class I>
  constexpr scalar_type<UnaryOp> operator[](I i) const {
    return op_(e_[i]);
  }

 private:
  const E e_;
  const Op op_;
};

template <class E>
using NegateOp = UnaryOp<std::negate<scalar_type<E>>, E>;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_UNARY_OP_H
