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
#ifndef TTL_EXPRESSIONS_OPERATORS_H
#define TTL_EXPRESSIONS_OPERATORS_H

/// This file binds operators to expressions.
#include <ttl/Expressions/BinaryOp.h>
#include <ttl/Expressions/ScalarOp.h>
#include <ttl/Expressions/UnaryOp.h>

namespace ttl {
namespace expressions {

template <class L, class R>
constexpr auto operator+(L lhs, R rhs) noexcept {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
                                                          return l + r;
                                                        });
}

template <class L, class R>
constexpr auto operator-(L lhs, R rhs) noexcept {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
                                                          return l - r;
                                                        });
}

template <class L, class R>
constexpr const auto operator/(L lhs, R rhs) {
  return DivideOp<L, R>(lhs, rhs);
}

template <class L, class R>
constexpr const auto operator%(L lhs, R rhs) {
  return ModulusOp<L, R>(lhs, rhs);
}

template <class R>
constexpr const auto operator-(R rhs) {
  return NegateOp<R>(rhs);
}

/// Product needs to select between the scalar multiply and the tensor product,
/// based on the left and right types.
template <class L, class R,
          bool = std::is_arithmetic<L>::value or std::is_arithmetic<R>::value>
struct ProductOp {
  using type = MultiplyOp<L, R>;
};

template <class L, class R>
struct ProductOp<L, R, false> {
  using type = Product<L, R>;
};

template <class L, class R>
constexpr const auto operator*(L lhs, R rhs) {
  return typename ProductOp<L, R>::type(lhs, rhs);
}

} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_OPERATORS_H
