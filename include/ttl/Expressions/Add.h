// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_ADD_EXP_H
#define TTL_EXPRESSIONS_ADD_EXP_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>

namespace ttl {
namespace expressions {
/// Add represents an element-wise combination of two expressions.
///
/// @precondition
///   is_equivalent<L::External, R::External>::value == true
/// @precondition
///
/// @postcondition
///   Add<...>::External = L::External
/// @postcondition
///
/// This expression combines two expressions that have equivalent External shape
/// to result in an expression that has an External shape equal to that on the
/// left hand side. Add operations do not have any contracted dimensions.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
/// @tparam          Op The element-wise operation.
template <class L, class R, template <class> class Op>
class Add;

/// The expression Traits for Add expressions.
template <class L, class R, template <class> class Op>
struct Traits<Add<L, R, Op>>
{
  static constexpr int Dimension = Traits<L>::Dimension;
  static constexpr int Rank = Traits<L>::Rank;
  using Scalar = typename Traits<L>::Scalar;
  using IndexPack = typename Traits<L>::IndexPack;
  using Type = Add<L, R, Op>;
  using ExpressionType = Expression<Type>;
};

template <class L, class R, template <class> class Op>
class Add : Expression<Add<L, R, Op>> {
 public:
  static constexpr int Rank = Traits<Add>::Rank;
  using Scalar = typename Traits<Add>::Scalar;

  Add(const L& lhs, const R& rhs) : lhs_(lhs), rhs_(rhs) {
  }

  constexpr Scalar operator[](IndexSet<Rank> i) const {
    return Op<Scalar>(lhs_[i], rhs_[i]);
  }

 private:
  const L& lhs_;
  const R& rhs_;
};

template <class L, class R, class = check_compatible<L, R>>
inline constexpr Add<L, R, std::plus> operator+(L&& lhs, R&& rhs) {
  return Add<L, R, std::plus>(lhs, rhs);
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_ADD_EXP_H
