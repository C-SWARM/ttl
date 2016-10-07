// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>

namespace ttl {
namespace expressions {
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
class TensorProduct;

/// The expression Traits for TensorProduct.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
struct Traits<TensorProduct<L, R>> {
  /// Traits specific to the TensorProduct
  /// @{
  using LeftType = L;
  using RightType = R;
  using LeftOuterType = typename Traits<L>::IndexType;
  using RightOuterType = typename Traits<R>::IndexType;
  using OuterType = symdif<LeftOuterType, RightOuterType>;
  using ProductType = unite<LeftOuterType, RightOuterType>;
  static constexpr int ProductSize = size<ProductType>::value;
  static constexpr int OuterSize =  size<OuterType>::value;
  /// @}

  /// External traits
  /// @{
  using ScalarType = promote<L, R>;
  using IndexType = OuterType;
  static constexpr int Rank = size<IndexType>::value;
  static constexpr int Dimension = Traits<L>::Dimension;
  /// @}
};


template <int N, class E, int M = Traits<E>::OuterSize>
struct contract {
  using L = typename Traits<E>::LeftType;
  using R = typename Traits<E>::RightType;
  using S = typename Traits<E>::ScalarType;
  static constexpr int Size = Traits<E>::ProductSize;
  static constexpr int Dimension = Traits<E>::Dimension;

  static S op(const L& lhs, const R& rhs, IndexSet<Size> i) {
    S s();
    for (i[N] = 0; i[N] < Dimension; ++i[N]) {
      s += contract<N + 1, E, M>::op(lhs, rhs, i);
    }
    return s;
  }
};

template <class E, int M>
struct contract<M, E, M> {
  using L = typename Traits<E>::LeftType;
  using R = typename Traits<E>::RightType;
  using S = typename Traits<E>::ScalarType;
  static constexpr int Size = Traits<E>::ProductSize;

  static S op(const L& lhs, const R& rhs, IndexSet<Size> i) {
    return 0;
  }
};

/// The TensorProduct expression implementation.
template <class L, class R>
class TensorProduct : Expression<TensorProduct<L, R>>
{
 public:
  TensorProduct(L lhs, R rhs) : lhs_(lhs), rhs_(rhs) {
  }

  auto operator[](IndexSet<Traits<TensorProduct>::OuterSize> i) const
    -> typename Traits<TensorProduct>::ScalarType
  {
    static constexpr int OuterSize = Traits<TensorProduct>::OuterSize;
    static constexpr int ProductSize = Traits<TensorProduct>::ProductSize;

    // Extend the index set with enough space for the inner, contracted size,
    // copy the outer dimensions into it, and then perform the contraction of
    // the inner indices.
    IndexSet<ProductSize> inner;
    std::copy(i, i + OuterSize, inner);
    return contract<OuterSize, TensorProduct>::op(lhs_, rhs_, inner);
  }

 private:
  L lhs_;
  R rhs_;
};


} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
