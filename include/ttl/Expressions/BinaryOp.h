// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_BINARY_OP_H
#define TTL_EXPRESSIONS_BINARY_OP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/promote.h>
#include <ttl/util/is_equivalent.h>
#include <functional>

namespace ttl {
namespace expressions {
/// BinaryOp represents an element-wise combination of two expressions.
///
/// This expression combines two expressions that have equivalent free types to
/// result in an expression that has a free type equal to that on the left hand
/// side.
///
/// @tparam          Op The element-wise binary operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
class BinaryOp;

/// The expression Traits for BinaryOp expressions.
///
/// The binary op expression just exports its left-hand-side expression
/// types. This is a somewhat arbitrary decision---it could export its right
/// hand side as well.
///
/// It overrides the ScalarType based on type promotion rules for the underlying
/// scalar types.
///
/// @tparam          Op The type of the operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
struct expression_traits<BinaryOp<Op, L, R>> : expression_traits<L>
{
  using scalar_type = promote<L, R>;
};

/// The BinaryOp expression implementation.
///
/// The BinaryOp captures its left hand side and right hand side expressions,
/// and a function object or lambda for the operation, and implements the
/// operator[] operation to evaluate an index.
///
/// @tparam          Op The element-wise binary operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
class BinaryOp : Expression<BinaryOp<Op, L, R>>
{
  static_assert(util::is_equivalent<free_type<L>, free_type<R>>::value,
                "BinaryOp expressions do not have equivalent index types.");
 public:
  BinaryOp(L lhs, R rhs) : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class I>
  constexpr scalar_type<BinaryOp> operator[](I i) const {
    return op_(lhs_[i], rhs_[i]);
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

template <class L, class R>
using AddOp = BinaryOp<std::plus<promote<L, R>>, L, R>;

template <class L, class R>
using SubtractOp = BinaryOp<std::minus<promote<L, R>>, L, R>;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_BINARY_OP_H
