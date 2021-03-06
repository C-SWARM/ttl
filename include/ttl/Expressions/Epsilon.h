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
/// This header defines and implements the epsilon expression.
///
/// https://en.wikipedia.org/wiki/Levi-Civita_symbol
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_EPSILON_H
#define TTL_EXPRESSIONS_EPSILON_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {

template <int n, int N>
struct sort;

template <int i, int n, int N>
struct insert
{
  template <class Index>
  static int op(Index index, int parity) {
    int s = std::get<i>(index);
    int t = std::get<i-1>(index);

    // If the two elements are equal this is not a permutation and we can early
    // terminate with 0.
    if (s == t) {
      return 0;
    }

    // If the current element is larger than the previous element then we can
    // continue our sort at the next outermost element.
    if (s > t) {
      return sort<n+1,N>::op(index, parity);
    }

    // Swap the elements and the parity.
    std::get<i>(index) = t;
    std::get<i-1>(index) = s;
    parity = ((parity == 1) ? -1 : 1);

    // Continue insertion for this outer loop (i)
    return insert<i-1,n,N>::op(index, parity);
  }
};

/// We've performed the nth insertion pass, so move on to the next.
template <int n, int N>
struct insert<0,n,N>
{
  template <class Index>
  static constexpr int op(const Index index, int parity) {
    return sort<n+1,N>::op(index, parity);
  }
};

/// Process the nth element in the Index.
template <int n, int N>
struct sort
{
  template <class Index>
  static constexpr int op(const Index index, int parity) {
    return insert<n,n,N>::op(index, parity);
  }
};

/// Sort base case when we've processed all of the elements in the Index.
template <int N>
struct sort<N,N>
{
  template <class Index>
  static constexpr int op(const Index, int parity) {
    return parity;
  }
};
} // namespace detail

template <int D, class Index>
class Epsilon : public Expression<Epsilon<D,Index>>
{
  static constexpr int sort(const Index index) noexcept {
    return detail::sort<0, std::tuple_size<Index>::value>::op(index, 1);
  }

 public:
  template <class I>
  static constexpr int eval(const I index) noexcept {
    return sort(ttl::expressions::transform<Index>(index));
  }
};

template <int D, class Index>
struct traits<Epsilon<D, Index>>
{
  using outer_type = Index;
  using scalar_type = int;
  using dimension = std::integral_constant<int, D>;
  using rank = typename std::tuple_size<Index>::type;
};
} // namespace expressions


template <int D = -1, class... I>
auto epsilon(I... i) {
  return expressions::Epsilon<D,std::tuple<I...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_EPSILON_H
