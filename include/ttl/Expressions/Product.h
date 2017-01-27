// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/contract.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/promote.h>

namespace ttl {
namespace expressions {

/// Forward declare the tensor product for the traits implementation.
template <class L, class R>
class Product;

/// The expression Traits for Product.
///
/// The tensor product promotes the scalar type from its left and right hand
/// sides, and performs a set disjoint union to expose its free_type.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
struct traits<Product<L, R>>
{
  using l_outer_type = expressions::outer_type<L>;
  using r_outer_type = expressions::outer_type<R>;

  using outer_type = set_xor<l_outer_type, r_outer_type>;
  using inner_type = set_and<l_outer_type, r_outer_type>;
  using scalar_type = promote<L, R>;
  using dimension = typename traits<L>::dimension;
  using rank = typename std::tuple_size<outer_type>::type;
};

/// The Product expression implementation.
///
/// A tensor product combines two expressions using multiplication, potentially
/// mixed with contraction (accumulation) over some shared dimensions of the two
/// expressions.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
class Product : public Expression<Product<L, R>>
{
  static_assert(dimension<L>::value == dimension<R>::value,
                "Cannot combine expressions with different dimensionality");
  static_assert(is_expression<L>::value, "Operand is not Expression");
  static_assert(is_expression<R>::value, "Operand is not Expression");

 public:
  constexpr Product(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs) {
  }

  /// The eval() operation for the product forwards to the contraction routine.
  ///
  /// In TTL, contraction requires that we take an outer index---something like
  /// (i,j)---extend it with "slots" for inner hidden dimensions---like
  /// (i,j,K,L)---and then iterate over all of the values for the inner hidden
  /// dimensions---(i,j,0,0), (i,j,0,1), etc)---accumulating the results.
  ///
  /// @param          i The incoming index.
  /// @returns          The scalar contraction of the hidden dimensions in the
  ///                   expression.
  template <class I>
  constexpr auto eval(I i) const noexcept {
    return contract<Product>(i, [&](auto index){
        return lhs_.eval(index) * rhs_.eval(index);
      });
  }

 private:
  L lhs_;                                    //!< The left-hand-side expression
  R rhs_;                                    //!< The right-hand-side expression
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
