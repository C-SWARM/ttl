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
#ifndef TTL_EXPRESSIONS_BINARY_OP_H
#define TTL_EXPRESSIONS_BINARY_OP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/promote.h>
#include <functional>

namespace ttl {
namespace expressions {
/// BinaryOp represents an element-wise combination of two expressions.
///
/// This expression combines two expressions that have equivalent free types to
/// result in an expression that has a free type equal to that on the left hand
/// side.
///
/// @tparam          Op The element-wise binary operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
class BinaryOp;

/// The expression Traits for BinaryOp expressions.
///
/// The binary op expression just exports its left-hand-side expression
/// types. This is a somewhat arbitrary decision---it could export its right
/// hand side as well.
///
/// It overrides the ScalarType based on type promotion rules for the underlying
/// scalar types.
///
/// @tparam          Op The type of the operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
struct traits<BinaryOp<Op, L, R>> : public traits<L>
{
  using scalar_type = promote<L, R>;
};

/// The BinaryOp expression implementation.
///
/// The BinaryOp captures its left hand side and right hand side expressions,
/// and a function object or lambda for the operation, and implements the
/// get operation to evaluate an index.
///
/// @tparam          Op The element-wise binary operation.
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class Op, class L, class R>
class BinaryOp : public Expression<BinaryOp<Op, L, R>>
{
  static_assert(is_expression_t<L>::value, "Operand is not Expression");
  static_assert(is_expression_t<R>::value, "Operand is not Expression");
  static_assert(equivalent<outer_type<L>, outer_type<R>>::value,
                "BinaryOp expressions do not have equivalent index types.");
  static_assert(dimension<L>() == dimension<R>(),
                "Cannot operate on expressions of differing dimension");
 public:
  constexpr BinaryOp(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs), op_() {
  }

  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return op_(lhs_.eval(index), rhs_.eval(index));
  }

 private:
  L lhs_;
  R rhs_;
  Op op_;
};

template <class L, class R>
using AddOp = BinaryOp<std::plus<promote<L, R>>, L, R>;

template <class L, class R>
using SubtractOp = BinaryOp<std::minus<promote<L, R>>, L, R>;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_BINARY_OP_H
