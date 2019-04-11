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
#include <ttl/Expressions/UnaryOp.h>

namespace ttl {
namespace expressions {
template <class L, class R>
constexpr auto
operator+(L lhs, R rhs) noexcept {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
    return l + r;
  });
}

template <class L, class R>
constexpr auto
operator-(L lhs, R rhs) noexcept {
  return make_binary_op(std::move(lhs), std::move(rhs), [](auto l, auto r) {
    return l - r;
  });
}

template <class L, class R,
          class = std::enable_if_t<std::is_arithmetic<R>::value>>
constexpr auto
operator/(L lhs, R rhs) noexcept {
  return make_unary_op(std::move(lhs), [r=std::move(rhs)](auto l) {
    return l / r;
  });
}

template <class L, class R,
          class = std::enable_if_t<std::is_arithmetic<R>::value>>
constexpr auto
operator%(L lhs, R rhs) noexcept {
  return make_unary_op(std::move(lhs), [r=std::move(rhs)](auto l) {
    return l % r;
  });
}

template <class R>
constexpr auto
operator-(R rhs) noexcept {
  return make_unary_op(std::move(rhs), [](auto r) {
    return -r;
  });
}

template <class L, class R>
constexpr auto
make_product(L lhs, R rhs, std::false_type, std::false_type) noexcept {
  return Product<L, R>(lhs, rhs);
}

template <class L, class R>
constexpr auto
make_product(L lhs, R rhs, std::true_type, std::false_type) noexcept {
  return make_unary_op(std::move(rhs), [lhs](auto r) {
    return lhs * r;
  });
}

template <class L, class R>
constexpr const auto
make_product(L lhs, R rhs, std::false_type, std::true_type) noexcept {
  return make_unary_op(std::move(lhs), [rhs](auto l) {
    return l * rhs;
  });
}

template <class L, class R>
constexpr const auto
operator*(L lhs, R rhs) noexcept {
  return make_product(std::move(lhs), std::move(rhs),
                      typename std::is_arithmetic<L>::type{},
                      typename std::is_arithmetic<R>::type{});
}
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_OPERATORS_H
