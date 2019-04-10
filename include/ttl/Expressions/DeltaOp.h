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
/// This header defines and implements the delta expression.
///
/// The delta expression differs from the delta tensor in that it has no storage
/// allocated with it. It is merely an expression tree node that returns 1 when
/// all of the index values match and 0 otherwise.
///
/// A delta tensor can be allocated by assigning a delta expression to a
/// tensor.
///
/// @code
///   Tensor<3,4,double> D = delta<4>(i,j,k);
/// @code
///
/// The delta expression cannot infer the proper dimensionality of the
/// expression node, and it needs to be able to implement the dimension trait,
/// so the user must statically specify the dimensionality in the expression.
///
/// @todo We can implement the dimension trait with a universal dimension so
///       that assertions about the dimensionality will always match.
///
/// The scalar type of the delta expression is integer, but that type is
/// automatically promoted in all use cases.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_DELTA_H
#define TTL_EXPRESSIONS_DELTA_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {
template <int n, int N>
struct is_diagonal
{
  template <class Other, class Index>
  static constexpr bool op(Other d, Index index) noexcept {
    return (std::get<n>(index) == d && is_diagonal<n+1,N>::op(d, index));
  }
};

template <int N>
struct is_diagonal<N, N>
{
  template <class Other, class Index>
  static constexpr bool op(Other, Index) noexcept {
    return true;
  }
};
} // namespace detail

template <int D, class Index>
class DeltaOp : public Expression<DeltaOp<D, Index>>
{
 public:
  constexpr int eval(Index index) const noexcept {
    constexpr int Rank = std::tuple_size<Index>::value;
    return detail::is_diagonal<0, Rank>::op(std::get<0>(index), index);
  }

  template <class Other>
  constexpr int eval(Other index) const noexcept {
    return eval(transform<Index>(index));
  }
};

template <int D>
class DeltaOp<D, std::tuple<>> : public Expression<DeltaOp<D, std::tuple<>>>
{
 public:
  template <class Index>
  constexpr int eval(Index) const noexcept {
    return 1;
  }
};

template <int D, class Index>
struct traits<DeltaOp<D, Index>>
{
  using     outer_type = Index;
  using    scalar_type = int;
  using dimension_type = std::integral_constant<int, D>;
  using      rank_type = typename std::tuple_size<Index>::type;
};
} // namespace expressions

template <int D = -1, class... Index>
auto delta(Index... i) {
  return expressions::DeltaOp<D, std::tuple<Index...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_DELTA_H
