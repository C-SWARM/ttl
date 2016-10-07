// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_BINARY_OP_H
#define TTL_EXPRESSIONS_BINARY_OP_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>

namespace ttl {
namespace expressions {
/// BinaryOp represents an element-wise combination of two expressions.
///
/// @precondition
///   is_equivalent<L::External, R::External>::value == true
/// @precondition
///
/// @postcondition
///   BinaryOp<...>::External = L::External
/// @postcondition
///
/// This expression combines two expressions that have equivalent External shape
/// to result in an expression that has an External shape equal to that on the
/// left hand side. BinaryOp operations do not have any contracted dimensions.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
/// @tparam          Op The element-wise binary operation.
template <class L, class R, class Op>
class BinaryOp;

/// The expression Traits for BinaryOp expressions.
///
/// The binary op expression just exports its left-hand-side expression
/// types. This is a somewhat arbitrary decision---it could export its right
/// hand side as well.
///
/// It overrides the ScalarType based on type promotion rules.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
/// @tparam           R The type of the operation.
template <class L, class R, class Op>
struct Traits<BinaryOp<L, R, Op>> : public Traits<L> {
  using ScalarType = promote<L, R>;
};

/// The BinaryOp expression implementation.
///
/// The BinaryOp captures its left hand side and right hand side expressions,
/// and a function object or lambda for the operation, and implements the
/// operator[] operation to evaluate an index.
template <class L, class R, class Op>
class BinaryOp : Expression<BinaryOp<L, R, Op>>
{
 public:
  BinaryOp(L lhs, R rhs) : lhs_(lhs), rhs_(rhs), op_() {
  }

  constexpr auto operator[](IndexSet<Traits<BinaryOp>::Rank> i) const
    -> typename Traits<BinaryOp>::ScalarType
  {
    return op_(lhs_[i], rhs_[detail::shuffle<Traits<BinaryOp>::Rank,
                             typename Traits<L>::IndexType,
                             typename Traits<R>::IndexType>(i)]);
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

/// Convenience metafunction to compute the type of a BinaryOp when the Op is a
/// function object type from the STL (like std::plus).
///
/// @tparam           L The type of the left-hand-side expression.
/// @tparam           R The type of the right-hand-side expression.
/// @tparam          Op The type of the binary function object.
///
/// @treturn            The type of the BinaryOp for L, R, Op.
///
/// @{
template <class L, class R, class Op>
struct binary_op_type_impl;

template <class L, class R, template <class> class Op>
struct binary_op_type_impl<L, R, Op<void>> {
 private:
  using LeftScalarType_ = typename Traits<L>::ScalarType;
  using RightScalarType_ = typename Traits<R>::ScalarType;
  using ScalarType_ = decltype(Op<LeftScalarType_>()(LeftScalarType_(), RightScalarType_()));

 public:
  using type = BinaryOp<L, R, Op<ScalarType_>>;
};

template <class L, class R, class Op>
using binary_op_type = typename binary_op_type_impl<L, R, Op>::type;
/// @}

template <class L, class R, class = check_compatible<L, R>>
constexpr auto operator+(L lhs, R rhs)
  -> BinaryOp<L, R, std::plus<promote<L, R>>>
{
  return BinaryOp<L, R, std::plus<promote<L, R>>>(lhs, rhs);
}

template <class L, class R, class = check_compatible<L, R>>
constexpr auto operator-(L lhs, R rhs)
  -> BinaryOp<L, R, std::minus<promote<L, R>>>
{
  return BinaryOp<L, R, std::minus<promote<L, R>>>(lhs, rhs);
}
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_BINARY_OP_H
