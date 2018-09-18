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
/// The recursive contraction template.
///
/// This template is instantiated to generate a loop for each inner dimension
/// for the expression. Each loop accumulates the result of the nested loops'
/// outputs.
template <class E,
          int n = std::tuple_size<outer_type<E>>::value,
          int M = std::tuple_size<concat<outer_type<E>, inner_type<E>>>::value,
          int D = dimension<E>::value>
struct contract_impl
{
  static_assert(D > 0, "Contraction requires explicit dimensionality");

  template <class Index, class F>              // c++14 auto (icc 16 complains)
  CUDA static auto op(Index index, F&& f) noexcept {
    decltype(f(index)) s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index) = i;
      s += contract_impl<E, n+1>::op(index, std::forward<F>(f));
    }
    return s;
  }
};

/// The contraction base case evaluates the lambda on the current index.
template <class E, int M, int D>
struct contract_impl<E, M, M, D>
{
  template <class Index, class F>              // c++14 auto (icc 16 complains)
  CUDA static constexpr auto op(Index index, F&& f) noexcept {
    return f(index);
  }
};

/// Simple local utility to take an external index, select the subset of indices
/// that appear in the Expression's outer type, and extend it with indices for
/// the Expression's inner type.
template <class E,
          class Index>           // Index c++14 auto (icc 16 complains)
CUDA constexpr auto extend(Index i) {
  return std::tuple_cat(transform(outer_type<E>{}, i), inner_type<E>{});
}
} // namespace detail

/// The external entry point for contraction takes the external index set and
/// the lambda to apply in the inner loop, and instantiates the recursive
/// template to expand the inner loops.
///
/// @tparam           E The type of the expression being contracted.
///
/// @param            i The partial index generated externally.
/// @param            f The lambda expression to evaluate in the inner loop.
///
/// @returns            The fully contracted scalar value, i.e., the sum of the
///                     inner loop invocations.
template <class E,
          class Index, class F>              // c++14 auto (icc 16 complains)
CUDA auto contract(Index i, F&& f) noexcept {
  using impl = detail::contract_impl<E>;
  return impl::op(detail::extend<E>(i), std::forward<F>(f));
}
} // namespace contract
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_CONTRACT_H
