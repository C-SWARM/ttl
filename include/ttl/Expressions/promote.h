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
#ifndef TTL_EXPRESSIONS_PROMOTE_H
#define TTL_EXPRESSIONS_PROMOTE_H

#include <ttl/Expressions/Expression.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {

/// Template for promoting scalar types.
///
/// We use multiplication as the default promotion operator. This might not be
/// the best choice, but we're going with it for now.
template <class L, class R,
          bool = std::is_arithmetic<L>::value,
          bool = std::is_arithmetic<R>::value>
struct promote {
  using type = decltype(L() * R());             // both scalars
};

template <class L, class R>
struct promote<L, R, true, false>
{
  using type = typename promote<L, scalar_t<R>>::type;
};

template <class L, class R>
struct promote<L, R, false, true>
{
  using type = typename promote<scalar_t<L>, R>::type;
};

template <class L, class R>
struct promote<L, R, false, false>
{
  using type = typename promote<scalar_t<L>, scalar_t<R>>::type;
};
} // namespace detail

template <class L, class R>
using promote_t = typename detail::promote<L, R>::type;
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_PROMOTE_H
