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
#ifndef TTL_UTIL_LINEARIZE_H
#define TTL_UTIL_LINEARIZE_H

#include <ttl/util/pow.h>
#include <tuple>

namespace ttl {
namespace util {
namespace detail {

/// Linearization of an index means processing each term in the index.
///
/// @tparam           D The dimensionality of the data.
/// @tparam           T The type of the index (a tuple of indices).
/// @tparam           i The term we're processing.
/// @tparam           N The number of terms in the index---we need this to
///                     terminate template recursion.
template <int D, class T, int i = 0, int N = std::tuple_size<T>::value>
struct linearize_impl
{
  static constexpr int op(T index) noexcept {
    return head(index) + tail(index);
  }

 private:
  static constexpr int head(T index) noexcept {
    return int(std::get<i>(index)) * util::pow(D, N - i - 1);
  }

  static constexpr int tail(T index) noexcept {
    return linearize_impl<D, T, i + 1, N>::op(index);
  }
};

/// Recursive base case for linearization is when we've processed all of the
/// terms in the index.
template <int D, class T, int N>
struct linearize_impl<D, T, N, N> {
  static constexpr int op(T) noexcept {
    return 0;
  }
};
} // namespace detail

template <int D, class T>
constexpr int linearize(T index) noexcept {
  return detail::linearize_impl<D, T>::op(index);
}

} // namespace util
} // namespace ttl


#endif // #ifndef TTL_UTIL_LINEARIZE_H
