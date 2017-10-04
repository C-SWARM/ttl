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
#ifndef TTL_EXPRESSIONS_SCALAR_OP_H
#define TTL_EXPRESSIONS_SCALAR_OP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/promote.h>
#include <functional>
#include <type_traits>

namespace ttl {
namespace expressions {
/// ScalarOp represents a multiplication between a scalar and an expression.
///
/// Either the left-hand-side or the right-hand-side might be a scalar. The
/// scalar operation must preserve this ordering.
///
/// Ops supported: *, /, %
template <class Op, class L, class R, bool = std::is_arithmetic<L>::value>
class ScalarOp;

/// The expression Traits for ScalarOp expressions.
///
/// The scalar op just exports its Expression's Traits, with the scalar type
/// adjusted as necessary.
template <class Op, class L, class R>
struct traits<ScalarOp<Op, L, R, true>> : public traits<R>
{
  using scalar_type = promote<L, R>;
};

template <class Op, class L, class R>
struct traits<ScalarOp<Op, L, R, false>> : public traits<L>
{
  using scalar_type = promote<L, R>;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the left-hand-side
/// operation.
template <class Op, class L, class R>
class ScalarOp<Op, L, R, true> : public  Expression<ScalarOp<Op, L, R, true>>
{
  static_assert(is_expression_t<R>::value, "Operand is not Expression");
 public:
  constexpr ScalarOp(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(lhs_, rhs_.eval(index));
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

/// The ScalarOp expression implementation.
///
/// This version of the scalar op matches when the scalar is the right-hand-side
/// operation.
template <class Op, class L, class R>
class ScalarOp<Op, L, R, false> : public Expression<ScalarOp<Op, L, R, false>>
{
  static_assert(is_expression_t<L>::value, "Operand is not Expression");

 public:
  constexpr ScalarOp(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(lhs_.eval(index), rhs_);
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

template <class L, class R>
using DivideOp = ScalarOp<std::divides<promote<L, R>>, L, R>;

template <class L, class R>
using ModulusOp = ScalarOp<std::modulus<promote<L, R>>, L, R>;

template <class L, class R>
using MultiplyOp = ScalarOp<std::multiplies<promote<L, R>>, L, R>;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_SCALAR_OP_H
