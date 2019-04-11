// -------------------------------------------------------------------*- C++ -*-
// Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
// University of Notre Dame
// Indiana University
// University of Washington
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------
/// @file  ttl/TensorImpl.h
/// @brief Contains the CPU-based implementation of the Tensor template.
// -----------------------------------------------------------------------------
#ifndef TTL_TENSOR_IMPL_H
#define TTL_TENSOR_IMPL_H

#include <ttl/Tensor.h>
#include <ttl/Expressions.h>
#include <ttl/Storage/StackStorage.hpp>
#include <ttl/Storage/ExternalStorage.hpp>
#include <ttl/util/linearize.h>
#include <ttl/util/multi_array.h>
#include <ttl/util/pow.h>
#include <cassert>
#include <tuple>
#include <algorithm>

namespace ttl {
/// Tensor template implementation.
///
/// @tparam        Rank The tensor's rank.
/// @tparam   Dimension The tensor's dimension.
/// @tparam      Scalar The tensor's scalar storage (only used explicitly when
///                     casting to the derived tensor type).
template <int Rank, int Dimension, class Scalar>
class Tensor : public StackStorage<std::remove_const_t<Scalar>, util::pow(Dimension, Rank)>
{
 public:
  /// The size is the number of scalars in the tensor.
  static constexpr size_t size() {
    return util::pow(Dimension, Rank);
  }

  static constexpr int R = Rank;
  static constexpr int D = Dimension;
  using T = std::remove_const_t<Scalar>;
  using Storage = StackStorage<T, size()>;

  /// Standard default constructor.
  ///
  /// The default constructor leaves the tensor data uninitialized. To
  /// initialize a tensor to 0 use Tensor<R,D,T> A = {};
  constexpr Tensor() noexcept = default;
  constexpr Tensor(const Tensor&) noexcept = default;
  constexpr Tensor(Tensor&&) noexcept = default;

  /// Normal assignment and move operators.
  constexpr Tensor& operator=(const Tensor&) noexcept = default;
  constexpr Tensor& operator=(Tensor&&) noexcept = default;

  /// Construct a tensor from an initializer list.
  ///
  /// @code
  ///  Tensor<R,D,T> A = {0,1,...};
  /// @code
  ///
  /// @param       list The initializer list for the tensor.
  constexpr Tensor(std::initializer_list<T> list) noexcept : Storage(list) {
  }

  /// We can always assign to tensors from initializer lists.
  ///
  /// This works for all Tensor types, independent from the storage class.
  ///
  /// @code
  ///  Tensor<R,D,T> A;
  ///  A = {0,1,...};
  /// @code
  ///
  /// @param       list The initializer list for the tensor.
  ///
  /// @returns          A reference to the Tensor.
  constexpr Tensor& operator=(std::initializer_list<T> list) noexcept {
    Storage::operator=(list);
    return *this;
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   Tensor<R,D,S> A = B(i,j)
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(const expressions::Expression<E>& rhs) noexcept {
    using expressions::make_bind;
    using expressions::outer_t;
    using Index = outer_t<E>;
    make_bind<Index>(*this) = rhs;
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   Tensor<R,D,S> A = B(i,j)
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(expressions::Expression<E>&& rhs) noexcept {
    using expressions::make_bind;
    using expressions::outer_t;
    using Index = outer_t<E>;
    make_bind<Index>(*this) = std::move(rhs);
  }

  /// Fill a tensor with a scalar value.
  ///
  /// @code
  ///   auto T = Tensor<R,D,int>().fill(1);
  /// @code
  ///
  /// @tparam         T The type of the scalar, must be compatible with S.
  /// @param     scalar The actual scalar.
  /// @returns          A reference to the tensor so that fill() can be
  ///                   chained.
  template <class S>
  Tensor& fill(S scalar) noexcept {
    using std::begin;
    std::fill_n(Storage::begin(), size(), scalar);
    return *this;
  }

  /// Multidimensional indexing into the tensor, used during evaluation.
  ///
  /// @code
  ///   Tensor<R,D,int> A
  ///   Index I = {0,1,...}
  ///   int i = A.eval(I)
  /// @code
  ///
  /// @tparam     Index The tuple type for the index.
  /// @param      index The multidimension index to access.
  /// @returns          The scalar value at the linearized @p index.
  template <class Index>
  constexpr const auto eval(Index index) const noexcept {
    constexpr int N = std::tuple_size<Index>::value;
    static_assert(R == N, "Index size does not match tensor rank");
    return Storage::get(util::linearize<D>(index));
  }

  /// Multidimensional indexing into the tensor, used during evaluation.
  ///
  /// @code
  ///   Tensor<R,D,int> A
  ///   Index I = {0,1,...}
  ///   A.eval(I) = 42
  /// @code
  ///
  /// @tparam     Index The tuple type for the index.
  /// @param      index The multidimension index to access.
  /// @returns          The a reference to the scalar value at the linearized @p
  ///                   index.
  template <class Index>
  constexpr auto& eval(Index index) noexcept {
    constexpr int N = std::tuple_size<Index>::value;
    static_assert(R == N, "Index size does not match tensor rank");
    return Storage::get(util::linearize<D>(index));
  }

  /// Bind a Bind expression to a tensor.
  ///
  /// This is the core operation that provides the tensor syntax that we want to
  /// provide. The user binds a tensor to a set of indices, and the TTL
  /// expression template structure can
  ///
  /// 1) Type check expressions.
  /// 2) Generate loops over indices during evaluation and contraction.
  ///
  /// @code
  ///  Index<'i'> i;
  ///  Index<'j'> j;
  ///  Index<'k'> k;
  ///  Tensor<R,D,T> A, B, C;
  ///  C(i,j) = A(i,k)*B(k,j)
  /// @code
  ///
  /// @tparam...  Index The index types to bind to the tensor.
  /// @param... indices The index values.
  /// @returns          A Bind expression that can serves as the leaf
  ///                   expression in TTL expressions.
  template <class... Index>
  constexpr const auto operator()(Index&&... indices) const noexcept {
    constexpr int N = sizeof...(Index);
    static_assert(R == N, "Index size does not match tensor rank");
    auto i = std::make_tuple(std::forward<Index>(indices)...);
    return expressions::make_bind(i, *this);
  }

  /// Bind a Bind expression to a tensor.
  ///
  /// This is the core operation that provides the tensor syntax that we want to
  /// provide. The user binds a tensor to a set of indices, and the TTL
  /// expression template structure can
  ///
  /// 1) Type check expressions.
  /// 2) Generate loops over indices during evaluation and contraction.
  ///
  /// @code
  ///  Index<'i'> i;
  ///  Index<'j'> j;
  ///  Index<'k'> k;
  ///  Tensor<R,D,T> A, B, C;
  ///  C(i,j) = A(i,k)*B(k,j)
  /// @code
  ///
  /// @tparam...      I The index types to bind to the tensor.
  /// @param... indices The index values.
  /// @returns          A Bind expression that can serves as the leaf
  ///                   expression in TTL expressions.
  template <class... Index>
  constexpr auto operator()(Index&&... indices) noexcept {
    constexpr int N = sizeof...(Index);
    static_assert(R == N, "Index size does not match tensor rank");
    auto i = std::make_tuple(std::forward<Index>(indices)...);
    return expressions::make_bind(i, *this);
  }

  /// Provide multidimensional array notation for direct element access.
  ///
  /// @param          i The index into the 1st dimension.
  ///
  /// @returns          An object suitable for further indexing or use in scalar
  ///                   contexts if it's totally indexed.
  constexpr auto operator[](int i) const noexcept {
    return util::make_multi_array<R,D>(*this)[i];
  }

  /// Provide multidimensional array notation for direct element access.
  ///
  /// @param          i The index into the 1st dimension.
  ///
  /// @returns          An object suitable for further indexing or use in scalar
  ///                   contexts if it's totally indexed.
  constexpr auto operator[](int i) noexcept {
    return util::make_multi_array<R,D>(*this)[i];
  }
};

/// Special-case Rank 0 tensors.
///
/// They are basically just scalars and should be treated as such wherever
/// possible.
template <int D, class S>
class Tensor<0,D,S>
{
 public:
  constexpr Tensor() noexcept = default;

  constexpr Tensor(S s) noexcept : data(s) {
  }

  constexpr auto operator=(S s) noexcept {
    return (data = s);
  }

  constexpr auto operator()() const noexcept {
    return data;
  }

  constexpr auto& operator()() noexcept {
    return data;
  }

  constexpr operator S() const noexcept {
    return data;
  }

  S data;
};

} // namespace ttl

#endif // #ifndef TTL_TENSOR_IMPL_H
