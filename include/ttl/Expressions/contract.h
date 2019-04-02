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
#ifndef TTL_EXPRESSIONS_CONTRACT_H
#define TTL_EXPRESSIONS_CONTRACT_H

#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>

namespace ttl {
namespace expressions {
namespace detail {
/// The recursive template class that evaluates tensor expressions.
///
/// The fundamental goal of ttl is to generate loops over tensor dimension,
/// evaluating the right-hand-side of the expression for each of the
/// combinations of inputs on the left hand side.
///
/// @code
///   for i : 0..D-1
///    for j : 0..D-1
///      ...
///        for n : 0..D-1
///           lhs(i,j,...,n) = rhs(i,j,...,n)
/// @code
///
/// We use recursive template expansion to generate these "loops" statically, by
/// dynamically enumerating the index dimensions. There is presumably a static
/// enumeration technique, as our bounds are all known statically.
///
/// @tparam           n The current dimension that we need to traverse.
/// @tparam           M The total number of free indices to enumerate.
/// @tparam           D The dimensionality of the space.
template <int n, int M, int D>
struct forall_impl
{
  /// The evaluation routine just iterates through the values of the nth
  /// dimension of the tensor, recursively calling the template.
  ///
  /// @tparam   Index The index we're generating.
  /// @tparam      Op The lambda to evaluate for each index.
  ///
  /// @param    index The partially constructed index.
  /// @param       op The operator we're going to evaluate for each index.
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      forall_impl<n + 1, M, D>::op(index, std::forward<Op>(op));
    }
  }
};

/// The base case for tensor evaluation.
///
/// @tparam         M The total number of dimensions to enumerate.
template <int M, int D>
struct forall_impl<M, M, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    op(index);
  }

  /// Specialize for the case where the index space is empty (i.e., the
  /// left-hand-side is a scalar as in an inner product).
  ///
  /// @code
  ///   c() = a(i) * b(i)
  /// @code
  template <class Op>
  static void op(Op&& op) {
    op();
  }
};

/// The recursive contraction template.
///
/// This template is instantiated to generate a loop for each inner dimension
/// for the expression. Each loop accumulates the result of the nested loops'
/// outputs.
template <class E,
          int n = std::tuple_size<outer_type<E>>::value,
          int M = std::tuple_size<concat<outer_type<E>, inner_type<E>>>::value>
struct contract_impl
{
  static constexpr auto D = dimension<E>();

  template <class Index, class Op>
  static auto op(Index index, Op&& op) noexcept {
    decltype(op(index)) s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      s += contract_impl<E, n+1>::op(index, std::forward<Op>(op));
    }
    return s;
  }
};

/// The contraction base case evaluates the lambda on the current index.
template <class E, int M>
struct contract_impl<E, M, M>
{
  template <class Index, class Op>
  static constexpr auto op(Index index, Op&& op) noexcept {
    return op(index);
  }
};

/// Simple local utility to take an external index, select the subset of indices
/// that appear in the Expression's outer type, and extend it with indices for
/// the Expression's inner type.
template <class E, class Index>
constexpr auto extend(Index i) {
  using Outer = outer_type<E>;
  using Inner = inner_type<E>;
  return std::tuple_cat(transform(Outer{}, i), Inner{});
}
} // namespace detail

/// The external entry point for contraction takes the external index set and
/// the lambda to apply in the inner loop, and instantiates the recursive
/// template to expand the inner loops.
///
/// @tparam           E The type of the expression being contracted.
/// @tparam          Op The type of the operation to evaluate in the inner loop.
///
/// @param            i The partial index generated externally.
/// @param           op The lambda expression to evaluate in the inner loop.
///
/// @returns            The fully contracted scalar value, i.e., the sum of the
///                     inner loop invocations.
template <class E, class Index, class Op>
constexpr auto contract(Index i, Op&& op) noexcept {
  using impl = detail::contract_impl<E>;
  return impl::op(detail::extend<E>(i), std::forward<Op>(op));
}

/// The external entry point for evaluating an expression.
///
template <class E, class Op>
constexpr void forall(Op&& op) noexcept {
  using Index = outer_type<E>;
  constexpr int M = std::tuple_size<Index>::value;
  constexpr int D = dimension<E>();
  detail::forall_impl<0,M,D>::op(Index{}, std::forward<Op>(op));
}
} // namespace expressions
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_CONTRACT_H
