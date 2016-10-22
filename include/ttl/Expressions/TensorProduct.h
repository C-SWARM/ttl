// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/index_type.h>
#include <ttl/Expressions/promote.h>

namespace ttl {
namespace expressions {

/// Forward declare the tensor product for the traits implementation.
template <class L, class R>
class TensorProduct;

/// The expression Traits for TensorProduct.
///
/// The tensor product promotes the scalar type from its left and right hand
/// sides, and performs a set disjoint union to expose its free_type.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
struct traits<TensorProduct<L, R>>
{
  using free_type = outer_type<typename traits<L>::free_type,
                               typename traits<R>::free_type>;
  using scalar_type = promote<L, R>;
  using dimension = typename traits<L>::dimension;
};

/// The TensorProduct expression implementation.
///
/// A tensor product combines two expressions using multiplication, potentially
/// mixed with contraction (accumulation) over some shared dimensions of the two
/// expressions.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
class TensorProduct : Expression<TensorProduct<L, R>>
{
  static_assert(dimension<L>::value == dimension<R>::value,
                "Cannot combine expressions with different dimensionality");
  static_assert(is_expression<L>::value, "Operand is not Expression");
  static_assert(is_expression<R>::value, "Operand is not Expression");

  /// The type of the inner dimensions, needed during contraction.
  ///
  /// @todo C++14 scope this inside of get
  using hidden_type = intersection<typename traits<L>::free_type,
                                   typename traits<R>::free_type>;
 public:
  TensorProduct(const L& lhs, const R& rhs) : lhs_(lhs), rhs_(rhs) {
  }

  /// The eval() operation for the product forwards to the contraction routine.
  ///
  /// In TTL, contraction requires that we take an outer index---something like
  /// (i,j)---extend it with "slots" for inner hidden dimensions---like
  /// (i,j,K,L)---and then iterate over all of the values for the inner hidden
  /// dimensions---(i,j,0,0), (i,j,0,1), etc)---accumulating the results.
  ///
  /// @tparam     Index The actual type of the incoming index.
  /// @param      index The incoming index.
  /// @returns          The scalar contraction of the hidden dimensions in the
  ///                   expression.
  template <class Index>
  constexpr scalar_type<TensorProduct> eval(Index i) const {
    return contract<std::tuple_size<Index>::value>(std::tuple_cat(i, hidden_type()));
  }

  /// Used as a leaf call during contraction.
  template <class Index>
  constexpr scalar_type<TensorProduct> apply(Index index) const {
    return lhs_.eval(index) * rhs_.eval(index);
  }

 private:
  /// Contraction helper template.
  ///
  /// This recursive template allows us to carry out the loop generation through
  /// static recursion.
  ///
  /// @tparam       Index The index type of the product.
  /// @tparam           n The next dimension being contracted.
  /// @tparam           N The total number of dimensions being contracted.
  template <class Index, int n, int N = std::tuple_size<Index>::value>
  struct contract_impl
  {
    /// Iterate through a single dimension, generating indices and recursing.
    ///
    /// As an example, we might have an index space of (I,J,K) and the @p index
    /// might be (i,j,[]) and we're supposed to iterate through the K
    /// dimension. We want a loop that generates (i,j,0), (i,j,1), (i,j,2),
    /// etc., and evaluates the expression for those values, accumulating the
    /// results.
    ///
    /// @param        e The tensor product expression.
    /// @param    index The partially generated index to fill in.
    static scalar_type<TensorProduct> op(const TensorProduct& e, Index index) {
      scalar_type<TensorProduct> s(0);
      for (int i = 0; i < dimension<TensorProduct>::value; ++i) {
        std::get<n>(index).set(i);
        s += contract_impl<Index, n + 1>::op(e, index);
      }
      return s;
    }
  };

  /// Contraction helper base case.
  ///
  /// This base case matches for fully generated indices, and should just
  /// evaluate the tensor produce for that index.
  ///
  /// @tparam     Index The index type that has been generated.
  /// @tparam         N The number of dimensions in Index (used to end
  ///                   recursion).
  template <class Index, int N>
  struct contract_impl<Index, N, N>
  {
    /// The leaf operation for contraction.
    ///
    /// This leaf simply evaluates the TensorProduct expression (which could be
    /// quite complex and evaluated through recursive expansion) for this @p
    /// index.
    ///
    /// @param        e The actual product expression.
    /// @param    index The fully generated index to evaluate.
    static constexpr scalar_type<TensorProduct>
    op(const TensorProduct& e, Index index) {
      return e.apply(index);
    }
  };

  /// Contract a set of dimensions in an expression.
  ///
  /// The purpose of this operation is to provide the entry point for a sequence
  /// of summations. We're basically taking an index that is defined as a
  /// superset of the free_type indices, and iterating over all of the
  /// contracted dimensions, accumulating their products.
  ///
  /// @code
  ///   C(i,k) = A(i,j) * B(j,k);
  ///
  ///   for (i: 0..D-1)  // <-- this is performed by "evaluation" in TensorBind
  ///     for (k: 0..D-1)
  ///       C(i,k) = 0;       // <-- this is where contraction starts
  //        for (j: 0..D-1)
  ///         C(i,k) += A(i,j)*B(j,k)
  /// @code
  ///
  /// @tparam           n The index of the dimension to start contracting.
  /// @tparam       Index The index type of the product, including contracted
  ///                     dimensions.
  ///
  /// @param        index The prefix of the actual index that we'd like to
  ///                     contract.
  ///
  /// @returns            The contracted scalar.
  template <int n, class Index>
  constexpr scalar_type<TensorProduct> contract(Index index) const {
    return contract_impl<Index, n>::op(*this, index);
  }

  const L& lhs_;                             //!< The left-hand-side expression
  const R& rhs_;                             //!< The right-hand-side expression
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
