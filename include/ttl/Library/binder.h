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
#ifndef TTL_LIBRARY_BINDER_H
#define TTL_LIBRARY_BINDER_H

#include <ttl/Index.h>
#include <ttl/Expressions/Bind.h>
#include <ttl/mp/cat.hpp>
#include <tuple>

namespace ttl {
namespace lib {
template <char N>
struct bind_impl
{
  static_assert(N > 0, "Unexpected index size");
  using next = typename bind_impl<N-1>::type;
  using type = mp::cat_t<std::tuple<Index<N>>, next>;
};

template <>
struct bind_impl<1>
{
  using type = std::tuple<Index<1>>;
};

/// This binds a tensor anonymously.
///
/// Some of the library APIs allow users to operate directly on Tensors, rather
/// than requiring bound expressions, however the library implementations are
/// designed to work on bound expressions.
///
/// This function will turn a Tensor into a bound expression, using a sequence
/// of library-defined indices (these indices don't have useful names and, while
/// they can match with other indices, they shouldn't be used in compound
/// expressions).
///
/// @tparam           E The type of the expression we're binding (probably a
///                     Tensor template).
///
/// @param            e The expression that we're binding.
///
/// @return             A fully bound expression.
template <class E>
constexpr auto bind(E&& e) {
  using namespace ttl::expressions;
  constexpr int Rank = rank_t<E>::value;
  return Bind<E,typename bind_impl<Rank>::type>(std::forward<E>(e));
}
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_BINDER_H
