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
/// This header defines and implements the zero expression.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_ZERO_H
#define TTL_EXPRESSIONS_ZERO_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <tuple>

namespace ttl {
namespace expressions {
template <int D, class Index>
class Zero : public Expression<Zero<D,Index>>
{
 public:
  template <class I>
  CUDA constexpr int eval(I) const noexcept {
    return 0;
  }
};

template <int D, class Index>
struct traits<Zero<D, Index>>
{
  using outer_type = Index;
  using scalar_type = int;
  using dimension = std::integral_constant<int, D>;
  using rank = typename std::tuple_size<Index>::type;
};
} // namespace expressions

template <int D = -1, class... I>
CUDA auto zero(I... i) {
  return expressions::Zero<D, std::tuple<I...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_ZERO_H
