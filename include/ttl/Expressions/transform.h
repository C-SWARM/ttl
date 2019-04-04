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
#ifndef TTL_EXPRESSIONS_TRANSFORM_H
#define TTL_EXPRESSIONS_TRANSFORM_H

#include <ttl/mp/index_of.hpp>
#include <tuple>

namespace ttl {
namespace expressions {
namespace detail {

/// Select, at compile time, which value to use for a tuple element.
///
/// In the transform template we would really like to be able to write something
/// like:
///
/// @code
///    if (is_integral<T>) : std::get<n>(to) : std::get<index_of(T, From>>(from);
/// @code
///
/// Unfortunately, when T is an integral type then it won't be found in the
/// index_of test in the from index. This ends up causing a compilation error
/// because, e.g., std::get<1>(std::tuple<>{}) doesn't compile. In order to
/// avoid this problem we use this helper template such that we can partially
/// specialize to avoid trying to expand that illegal get<> when the type is
/// integral.
///
/// @returns          (is_integral<T>::value) ? to : find(from)
template <class T, bool Integral = std::is_integral<T>::value>
struct select
{
  /// This specialization finds the index of the type T in the From pack, and
  /// returns its value. We check to make sure that the index is actually
  /// found.
  template <class From>
  static constexpr T op(const T, const From from) {
    constexpr int n = mp::index_of<T, From>::value;
    constexpr int N = std::tuple_size<From>::value;
    static_assert(n < N, "Index space is incompatible");
    return std::get<n>(from);
  }
};

template <class T>
struct select<T, true>
{
  /// The integral type version is really easy, we can just return the value
  /// we're passed.
  template <class From>
  static constexpr T op(const T to, From) {
    return to;
  }
};

/// The transform implementation creates a tuple in the target space.
///
/// We implement this template by iterating over each tuple, and concatenating
/// the values together.
template <class Index, int n = 0, int N = std::tuple_size<Index>::value>
struct transform_impl
{
  template <class From>
  static constexpr auto op(Index to, From from) noexcept {
    return std::tuple_cat(head(std::get<n>(to), from), tail(to, from));
  }

 private:
  template <class From, class T>
  static constexpr auto head(T to, From from) noexcept {
    return std::make_tuple(select<T>::op(to, from));
  }

  template <class From>
  static constexpr auto tail(Index to, From from) noexcept {
    return transform_impl<Index, n+1>::op(to, from);
  }
};

// Base case is empty pack.
template <template <class...> class Pack, class... I, int N>
struct transform_impl<Pack<I...>, N, N>
{
  template <class From>
  static constexpr auto op(Pack<I...>, From) noexcept {
    return Pack<>{};
  }
};

} // namespace detail

template <class To, class From>
constexpr To transform(To to, From from) noexcept {
  return detail::transform_impl<To>::op(to, from);
}

template <class To, class From>
constexpr To transform(From from) noexcept {
  return detail::transform_impl<To>::op(To{}, from);
}
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_TRANSFORM_H
