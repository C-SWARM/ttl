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
template <class E, class Op>
struct Traits<UnaryOp<E, Op>>
{
  static constexpr int Dimension = Traits<E>::Dimension;
  static constexpr int Rank = Traits<E>::Rank;
  using Scalar = typename Traits<E>::Scalar;
  using IndexPack = typename Traits<E>::IndexPack;
  using Type = UnaryOp<E, Op>;
  using ExpressionType = Expression<Type>;
};

template <class E, class Op>
class UnaryOp : Expression<UnaryOp<E, Op>> {
 public:
  static constexpr int Rank = Traits<UnaryOp>::Rank;
  using Scalar = typename Traits<UnaryOp>::Scalar;

  UnaryOp(const E& e, Op&& op) : e_(e), op_(op) {
  }

  constexpr Scalar operator[](IndexSet<Rank> i) const {
    using Index = typename Traits<E>::IndexPack;
    return op_(e_[i]);
  }

 private:
  const E& e_;
  const Op op_;
};

template <class E>
inline constexpr auto operator-(E&& e)
  -> UnaryOp<E, std::negate<typename Traits<E>::Scalar>> // @todo delete for C++14
{
  using Scalar = typename Traits<E>::Scalar;
  using UnaryOp = UnaryOp<E, std::negate<typename Traits<E>::Scalar>>;
  return UnaryOp(std::forward<E>(e), std::negate<Scalar>());
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_UNARY_OP_H
