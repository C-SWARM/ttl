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
#ifndef TTL_EXPRESSIONS_FORCE_H
#define TTL_EXPRESSIONS_FORCE_H

#include <ttl/Expressions/traits.h>

namespace ttl {
namespace expressions {
/// The force operation forces the evaluation of a tensor expression.
///
/// The returned value is the tensor resulting from the evaluation of the
/// expression. If the expression is a raw tensor then it will return a copy of
/// the tensor. The returned value is stack allocated.
///
/// @tparam  Expression The expression type.
///
/// @param            e The expression.
///
/// @returns            A tensor with the result of evaluating the
///                     expression. If the expression is a tensor or a reference
///                     to a tensor, then this returns a copy of the tensor. If
///                     the expression is an r-value reference to a tensor then
///                     it returns the tensor through the move operation.
template <class Expression>
tensor_type<Expression> force(Expression&& e) {
  static constexpr int N = dimension_t<Expression>::value;
  static_assert(N >= 0, "forced expression needs dimension");
  return std::forward<Expression>(e);
}
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_FORCE_H
