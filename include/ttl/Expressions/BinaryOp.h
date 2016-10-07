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
template <class L, class R, class Op>
struct Traits<BinaryOp<L, R, Op>>
{
  static constexpr int Dimension = Traits<L>::Dimension;
  static constexpr int Rank = Traits<L>::Rank;
  using Scalar = typename Traits<L>::Scalar;
  using IndexPack = typename Traits<L>::IndexPack;
  using Type = BinaryOp<L, R, Op>;
  using ExpressionType = Expression<Type>;
};

template <class L, class R, class Op>
class BinaryOp : Expression<BinaryOp<L, R, Op>> {
 public:
  static constexpr int Rank = Traits<BinaryOp>::Rank;
  using Scalar = typename Traits<BinaryOp>::Scalar;

  BinaryOp(const L& lhs, const R& rhs, Op&& op) : lhs_(lhs), rhs_(rhs), op_(op) {
  }

  constexpr Scalar operator[](IndexSet<Rank> i) const {
    using LP = typename Traits<L>::IndexPack;
    using RP = typename Traits<R>::IndexPack;
    return op_(lhs_[i], rhs_[detail::shuffle<Rank, LP, RP>(i)]);
  }

 private:
  const L& lhs_;
  const R& rhs_;
  const Op op_;
};

template <class L, class R, class = check_compatible<L, R>>
inline constexpr auto operator+(L&& lhs, R&& rhs)
  -> BinaryOp<L, R, std::plus<typename Traits<L>::Scalar>> // @todo delete for C++14
{
  using Scalar = typename Traits<L>::Scalar;
  using BinaryOp = BinaryOp<L, R, std::plus<typename Traits<L>::Scalar>>;
  return BinaryOp(lhs, rhs, std::plus<Scalar>());
}

//
template <class L, class R, class = check_compatible<L, R>>
inline constexpr auto operator-(L&& lhs, R&& rhs)
  -> BinaryOp<L, R, std::minus<typename Traits<L>::Scalar>> // @todo delete for C++14
{
  using Scalar = typename Traits<L>::Scalar;
  using BinaryOp = BinaryOp<L, R, std::minus<typename Traits<L>::Scalar>>;
  return BinaryOp(lhs, rhs, std::minus<Scalar>());
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_BINARY_OP_H
