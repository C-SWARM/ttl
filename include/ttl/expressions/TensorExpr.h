// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_EXP_H
#define TTL_EXPRESSIONS_TENSOR_EXP_H

#include <ttl/Pack.h>
#include <ttl/Expression.h>
#include <ttl/Index.h>
#include <ttl/detail/check.h>
#include <ttl/detail/linearize.h>
#include <ttl/detail/shuffle.h>
#include <utility>
#include <iostream>

namespace ttl {
namespace expressions {

template <template <int, class, int> class Tensor, // underlying Tensor type
          int Rank, class Scalar, int Dimension,   // tensor parameters
          class... Indices>                        // external indices
class TensorExpr :
    public Expression<TensorExpr<Tensor, Rank, Scalar, Dimension, Indices...>,
                      Pack<Indices...>, Pack<>>
{
  using TensorType = Tensor<Rank, Scalar, Dimension>;
  using  IndexType = Pack<Indices...>;
  using   ExprType = Expression<TensorExpr<Tensor, Rank, Scalar, Dimension,
                                           Indices...>,
                                Pack<Indices...>,
                                Pack<>>;

 public:
  explicit TensorExpr(TensorType& t, Indices...) : ExprType(*this), t_(t) {
  }

  /// Default assignment, move, and copy should work fine when the
  /// right-hand-side is a tensor expression of the same type, meaning it has
  /// the same underlying tensor shape (Rank, Dimension, and Scalar type) and is
  /// being indexed through the same indices.
  ///
  /// @{
  TensorExpr(TensorExpr&&) = default;
  TensorExpr& operator=(TensorExpr&& rhs) {
    t_ = rhs.t_;
    return *this;
  };

  TensorExpr(const TensorExpr&) = default;
  TensorExpr& operator=(const TensorExpr& rhs) {
    t_ = rhs.t_;
    return *this;
  }
  /// @}

  /// The index operator maps the index array using the normal interpretation of
  /// multidimensional indexing, using index 0 as the most-significant-index.
  ///
  /// @code
  ///        Rank = n = sizeof(index)
  ///   Dimension = k
  ///       index = {a, b, c, ..., z}
  ///           i = a*k^(n-1) + b*k^(n-2) + c*k^(n-3) + ... + z*k^0
  /// @code
  ///
  /// @param      index The index array, which must be the same length as the
  ///                   Rank of this TensorExpr.
  ///
  /// @returns          The scalar value at the linearized offset.
  ///
  /// @{
  constexpr Scalar operator[](std::array<int, Rank> i) const {
    return t_[detail::linearize<Dimension, Rank>(i)];
  }

  Scalar& operator[](std::array<int, Rank> i) {
    return t_[detail::linearize<Dimension, Rank>(i)];
  }
  /// @}

  /// The key function of this TensorExpr type is assignment from expressions
  /// that are not simply copies of the underlying tensor. This can be a shuffle
  /// operation, like a transpose, where the right hand side expression is a
  /// tensor expression but the index set is reordered, or it can be a more
  /// complex expression class.
  ///
  /// @{
  template <class... U,
            class = detail::check<sizeof...(U) == Rank>,
            class = detail::check<is_equivalent<IndexType, Pack<U...>>::value>>
  TensorExpr&
  operator=(const TensorExpr<Tensor, Rank, Scalar, Dimension, U...>& rhs) {
    using RHS = TensorExpr<Tensor, Rank, Scalar, Dimension, U...>;
    apply<0, RHS, Pack<U...>>::op(*this, rhs, {0});
    return *this;
  }

 private:
  template <int N, class RHS, class RType>
  struct apply {
    static void op(TensorExpr& lhs, const RHS& rhs, std::array<int, Rank> index)
    {
      for (int i = 0; i < Dimension; ++i) {
        index[N] = i;
        apply<N + 1, RHS, RType>::op(lhs, rhs, index);
      }
    }
  };

  template <class RHS, class RType>
  struct apply<Rank, RHS, RType> {
    static void op(TensorExpr& lhs, const RHS& rhs, std::array<int, Rank> i)
    {
      auto j = detail::shuffle<Rank, IndexType, RType>(i);
      lhs[i] = rhs[j];
    }
  };

  TensorType& t_;
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_EXP_H
