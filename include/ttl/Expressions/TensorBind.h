// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_BIND_H
#define TTL_EXPRESSIONS_TENSOR_BIND_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Tensor.h>
#include <ttl/detail/check.h>
#include <ttl/detail/linearize.h>
#include <ttl/detail/shuffle.h>
#include <ttl/Expressions/Expression.h>
#include <utility>
#include <iostream>

namespace ttl {
namespace expressions {

/// The expression that represents binding a Tensor to an Index map.
///
/// This expression is the leaf expression for all tensor operations, and
/// results from expressions that look like `x(i)`, `A(j, k, l)`, etc. It allows
/// multidimensional indexing into tensors, tensor assignment, and natively
/// supports shuffle operations.
///
/// @tparam      Scalar The underlying scalar type of the tensor.
/// @tparam   Dimension The underlying tensor dimension.
/// @tparam   IndexPack The index map for this expression.
template <class Scalar, int Dimension, class IndexPack>
class TensorBind;

/// The expression Traits for TensorBind expressions.
///
/// @tparam           S The scalar type for the expression.
/// @tparam           D The dimensionality of the expression.
/// @tparam           I The indices bound to this expression.
template <class S, int D, class I>
struct Traits<TensorBind<S, D, I>>
{
  static constexpr int Dimension = D;
  static constexpr int      Rank = size<I>::value;

  using         Scalar = S;
  using      IndexPack = I;
  using           Type = TensorBind<S, D, I>;
  using     TensorType = Tensor<Rank, S, D>;
  using ExpressionType = Expression<Type>;
};

template <class Scalar, int Dimension, class IndexPack>
class TensorBind : public Expression<TensorBind<Scalar, Dimension, IndexPack>>
{
 public:
  /// Import some names that we need.
  static constexpr int Rank = Traits<TensorBind>::Rank;
  using TensorType = typename Traits<TensorBind>::TensorType;

  /// A TensorBind expression just keeps a reference to the Tensor it wraps.
  ///
  /// @tparam         t The underlying tensor.
  explicit TensorBind(TensorType& t) : t_(t) {
  }

  /// Default assignment, move, and copy should work fine when the
  /// right-hand-side is a tensor expression of the same type, meaning it has
  /// the same underlying tensor shape (Rank, Dimension, and Scalar type) and is
  /// being indexed through the same indices.
  ///
  /// @{
  TensorBind(TensorBind&&) = default;
  TensorBind& operator=(TensorBind&& rhs) {
    t_ = rhs.t_;
    return *this;
  };

  TensorBind(const TensorBind&) = default;
  TensorBind& operator=(const TensorBind& rhs) {
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
  ///                   Rank of this TensorBind.
  ///
  /// @returns          The scalar value at the linearized offset.
  constexpr Scalar operator[](IndexSet<Rank> i) const {
    return t_[detail::linearize<Dimension, Rank>(i)];
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// This is the core functionality provided by the TensorBind expression. Its
  /// job is to loop through the index space defined by its IndexPack and
  /// evaluate the right hand side for the remapped index pack.
  ///
  /// @code
  ///   A(i, j) = B(j, i)
  ///
  ///   for (i in 0..Dimension-1)
  ///     for (j in 0..Dimension-1)
  ///       a = linearize(i, j)
  ///       b = linearize(j, i)
  ///       A[a] = B[b]
  ///
  /// @code
  ///
  /// The only way that I can figure out to do this is through recursion. We use
  /// the recursive apply template to force that expansion to happen at runtime
  /// and insure that there is the opportunity for inlining and optimization.
  ///
  /// @tparam         E Right-hand-side expression..., must implement Traits.
  /// @tparam    (anon) Restrict this operation to expressions that match.
  template <class E, class = check_compatible<TensorBind, E>>
  TensorBind& operator=(const E& rhs) {
    apply<0, E>::op(*this, rhs, {0});
    return *this;
  }

 private:
  Scalar& operator[](IndexSet<Rank> i) {
    return t_[detail::linearize<Dimension, Rank>(i)];
  }

  template <int N, class R>
  struct apply {
    static void op(TensorBind& lhs, const R& rhs, IndexSet<Rank> i) {
      for (i[N] = 0; i[N] < Dimension; ++i[N]) {
        apply<N + 1, R>::op(lhs, rhs, i);
      }
    }
  };

  template <class R>
  struct apply<Rank, R> {
    static void op(TensorBind& lhs, const R& rhs, IndexSet<Rank> i) {
      using RP = typename Traits<R>::IndexPack;
      auto j = detail::shuffle<Rank, IndexPack, RP>(i);
      lhs[i] = rhs[j];
    }
  };

  TensorType& t_;
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
