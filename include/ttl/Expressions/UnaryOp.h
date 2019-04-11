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
#ifndef TTL_EXPRESSIONS_UNARY_OP_H
#define TTL_EXPRESSIONS_UNARY_OP_H

#include <ttl/Expressions/Expression.h>
#include <functional>

namespace ttl {
namespace expressions {
/// UnaryOp represents a unary operation on an expression.
///
/// @tparam           E The Expression.
template <class E, class Op>
class UnaryOp;

/// The expression Traits for UnaryOp expressions.
///
/// This just exports the traits of the underlying expression.
template <class E, class Op>
struct traits<UnaryOp<E, Op>> : public traits<E> {
  using scalar_type = std::result_of_t<Op(scalar_t<E>)>;
};

template <class E, class Op>
class UnaryOp : public Expression<UnaryOp<E, Op>>
{
 public:
  constexpr UnaryOp(E e, Op op) noexcept : e_(std::move(e)), op_(std::move(op))
  {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(e_.eval(std::move(index)));
  }

 private:
  E e_;
  Op op_;
};

template <class E, class Op>
constexpr UnaryOp<E, Op> make_unary_op(E e, Op op) noexcept {
  return { std::move(e), std::move(op) };
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_UNARY_OP_H
