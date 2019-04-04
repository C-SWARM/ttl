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
/// This header defines and implements the identity expression.
///
/// The identity expression for an even rank N is just the product of N/2 delta
/// expressions. The indices for the delta expressions are just the indices of
/// the identity split in half and interleaved.
///
/// ttl::identity() = 1
/// ttl::identity(i,j) = delta(i,j)
/// ttl::identity(i,j,k,l) = delta(i,k) * delta(j,l)
/// ttl::identity(i,j,k,l,m,n) = delta(i,l) * delta(j,m) * delta(k,n)
///
/// Our strategy for building an identity is to shuffle the indices into the
/// right order and then recursively spawning the delta expressions by popping
/// two indices at a time.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_IDENTITY_H
#define TTL_EXPRESSIONS_IDENTITY_H

#include <ttl/Expressions/Bind.h>
#include <ttl/Expressions/DeltaOp.h>

namespace ttl {
namespace mp {
template <class T>
struct list {
  using car = std::tuple<>;
  using cdr = std::tuple<>;
};

template <class T0, class... T>
struct list<std::tuple<T0, T...>> {
  using car = std::tuple<T0>;
  using cdr = std::tuple<T...>;
};

template <class T>
using car = typename list<T>::car;

template <class T>
using cdr = typename list<T>::cdr;

template <class T, size_t n>
struct take_n {
  using next = take_n<cdr<T>, n - 1>;
  using lhs = cat_t<car<T>, typename next::lhs>;
  using rhs = typename next::rhs;
};

template <class T>
struct take_n<T, 0u> {
  using lhs = std::tuple<>;
  using rhs = T;
};

template <class T, class U>
struct merge {
  using  next = merge<cdr<T>, cdr<U>>;
  using first = cat_t<car<T>, car<U>>;
  using  type = cat_t<first, typename next::type>;
};

template <>
struct merge<std::tuple<>, std::tuple<>> {
  using type = std::tuple<>;
};

template <class T>
struct shuffle {
  using split = take_n<T, std::tuple_size<T>::value / 2>;
  using   lhs = typename split::lhs;
  using   rhs = typename split::rhs;
  using  type = typename merge<lhs, rhs>::type;
};

template <class T>
using shuffle_t = typename shuffle<T>::type;
} // namespace mp

namespace expressions {
template <int D>
constexpr auto identity(std::tuple<>) {
  return DeltaOp<D, std::tuple<>>();
}

template <int D, class T0, class T1, class... T>
constexpr auto identity(std::tuple<T0, T1, T...>) {
  using delta = DeltaOp<D,std::tuple<T0,T1>>;
  return delta() * identity<D>(std::tuple<T...>{});
}
} // namespace expressions

template <int D = -1, class... Index>
constexpr auto identity(Index...) {
  static_assert(sizeof...(Index) % 2 == 0, "The identity must have even rank.");
  using type = mp::shuffle_t<std::tuple<Index...>>;
  return expressions::identity<D>(type{}).to(std::tuple<Index...>{});
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_IDENTITY_H
