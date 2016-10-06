// -*- C++ -*-
#ifndef TTL_TENSOR_IMPL_H
#define TTL_TENSOR_IMPL_H

#include <ttl/Tensor.h>
#include <ttl/detail/check.h>
#include <ttl/detail/pow.h>
#include <ttl/Expressions/TensorBind.h>
#include <array>
#include <algorithm>

namespace ttl {
template <int Rank, typename Scalar, int Dimension>
class Tensor
{
 public:
  Tensor() = default;
  Tensor(Tensor&&) = delete;
  Tensor& operator=(Tensor&& rhs) = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor& rhs) = default;

  explicit Tensor(Scalar s) {
    std::fill(value_, value_ + Size_, s);
  }

  /// The tensor only supports simple linear addressing.
  ///
  /// Multidimensional indexing is all handled through a TensorBind that binds
  /// an Pack to the tensor, and knows how to transform a multidimensional
  /// index into a linear index.
  ///
  /// @{
  constexpr Scalar operator[](int i) const { return value_[i]; }
  Scalar& operator[](int i) { return value_[i]; }
  /// @}

  /// Create a tensor indexing expression for this tensor.
  ///
  /// This operation will bind the tensor to an Pack, which will allow us
  /// to make type inferences about the types of operations that should be
  /// available, as well as generate code to actually index the tensor in loops
  /// and to evaluate its elements.
  ///
  /// @code
  ///   static constexpr Index<'i'> i;
  ///   static constexpr Index<'j'> j;
  ///   static constexpr Index<'k'> k;
  ///   Tensor<3, double, 3> T;
  ///   auto expr = T(i, j, k);
  ///   T(i, j, k) = U(k, i, j);
  /// @code
  ///
  /// @tparam   Indices The set of indices to bind to the tensor dimensions.
  /// @tparam    [anon] A type-check (sizeof...(Indices) == Rank) metaprogram.
  ///
  /// @param indices... The actual set of indices to bind (e.g., (i, j, k)).
  ///
  /// @returns          A tensor indexing expression.
  template <class... Indices,
            class = detail::check<Rank == sizeof...(Indices)>>
  auto operator()(Indices... indices)
    -> expressions::TensorBind<Scalar, Dimension, Pack<Indices...>>
  {
    return expressions::TensorBind<Scalar, Dimension, Pack<Indices...>>(*this);
  }

 private:
  static constexpr int Size_ = detail::pow(Dimension, Rank);
  Scalar value_[Size_];
};
}

#endif // TTL_TENSOR_H
