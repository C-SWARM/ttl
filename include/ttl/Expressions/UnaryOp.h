// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_UNARY_OP_H
#define TTL_EXPRESSIONS_UNARY_OP_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>

namespace ttl {
namespace expressions {
/// UnaryOp represents a unary operation on an expression.
///
/// @tparam           E The Expression.
template <class E, class Op>
class UnaryOp;

/// The expression Traits for UnaryOp expressions.
///
/// This just exports the traits of the underlying expression.
template <class E, class Op>
struct Traits<UnaryOp<E, Op>> : public Traits<E> {
};

template <class E, class Op>
class UnaryOp : Expression<UnaryOp<E, Op>>
{
 public:
  UnaryOp(E e) : e_(e), op_() {
  }

  constexpr auto operator[](IndexSet<Traits<UnaryOp>::Rank> i) const
    -> typename Traits<UnaryOp>::ScalarType
  {
    return op_(e_[i]);
  }

 private:
  const E e_;
  const Op op_;
};

/// Convenience metafunction to create a unary op for a template.
///
/// The standard library provides some standard template function objects for
/// the unary operators. This metafunction binds any of these function object
/// types with the scalar value of the expression.
///
/// @tparam           E The expression type.
/// @tparam          Op The function object type (e.g., std::negate).
///
/// @treturns           The type of the unary operator for E, Op.
template <class E, template <class> class Op>
using unary_op_type = UnaryOp<E, Op<typename Traits<E>::ScalarType>>;

template <class E>
constexpr const unary_op_type<E, std::negate> operator-(E e) {
  return unary_op_type<E, std::negate>(e);
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_UNARY_OP_H
