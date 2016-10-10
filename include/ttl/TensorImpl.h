// -*- C++ -*-
#ifndef TTL_TENSOR_IMPL_H
#define TTL_TENSOR_IMPL_H

#include <ttl/Tensor.h>
#include <ttl/Expressions/TensorBind.h>
#include <ttl/util/pow.h>
#include <tuple>
#include <algorithm>

namespace ttl {
template <int Rank, typename Scalar, int Dimension>
class Tensor
{
  static constexpr auto Size_ = util::pow(Dimension, Rank);

 public:
  Tensor() = default;
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&& rhs) = default;
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor& rhs) = default;

  explicit Tensor(Scalar s) {
    std::fill(value_, value_ + Size_, s);
  }


  /// Simple linear addressing.
  /// @}
  constexpr Scalar operator[](int i) const {
    return value_[i];
  }

  Scalar& operator[](int i) {
    return value_[i];
  }
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
  /// @tparam    (anon) A type-check (sizeof...(Indices) == Rank) metaprogram.
  ///
  /// @param     (anon) The actual set of indices to bind (e.g., (i,j,k)).
  ///
  /// @returns          A tensor indexing expression.
  template <class... Indices>
  expressions::TensorBind<Tensor, std::tuple<Indices...>> operator()(Indices...)
  {
    static_assert(Rank == sizeof...(Indices), "Tensor indexing mismatch.");
    return expressions::TensorBind<Tensor, std::tuple<Indices...>>(*this);
  }

 private:
  Scalar value_[Size_];
};

template <int Rank, typename Scalar, int Dimension>
class Tensor<Rank, Scalar*, Dimension>
{
  static constexpr auto Size_ = util::pow(Dimension, Rank);

 public:
  Tensor() = delete;
  Tensor(Tensor&& rhs) = delete;
  Tensor(const Tensor&) = delete;

  Tensor& operator=(Tensor&& rhs) {
    std::copy(rhs.value_, rhs.value_ + Size_, value_);
    return *this;
  }

  Tensor& operator=(const Tensor& rhs) {
    std::copy(rhs.value_, rhs.value_ + Size_, value_);
    return *this;
  }

  Tensor(Scalar* value) : value_(*(Scalar(*)[Size_])value) {
  }

  Tensor(Scalar* value, Scalar s) : value_(*(Scalar(*)[Size_])value) {
    std::fill(value_, value_ + Size_, s);
  }

  /// Simple linear addressing.
  /// @}
  constexpr Scalar operator[](int i) const {
    return value_[i];
  }

  Scalar& operator[](int i) {
    return value_[i];
  }
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
  /// @tparam    (anon) A type-check (sizeof...(Indices) == Rank) metaprogram.
  ///
  /// @param     (anon) The actual set of indices to bind (e.g., (i,j,k)).
  ///
  /// @returns          A tensor indexing expression.
  template <class... Indices>
  expressions::TensorBind<Tensor, std::tuple<Indices...>> operator()(Indices...)
  {
    static_assert(Rank == sizeof...(Indices), "Tensor indexing mismatch.");
    return expressions::TensorBind<Tensor, std::tuple<Indices...>>(*this);
  }

 private:
  Scalar (&value_)[Size_];
};
} // namespace ttl

#endif // #ifndef TTL_TENSOR_IMPL_H
