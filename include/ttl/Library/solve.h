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
#ifndef TTL_LIBRARY_SOLVE_H
#define TTL_LIBRARY_SOLVE_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/LinearAlgebra.hpp>
#include <ttl/Library/binder.h>
#include <ttl/Library/matrix.h>

namespace ttl {
namespace lib {
template <class A, class B>
struct solve_impl
{
  template <class X>
  static int op(A a, B b, X& x) noexcept {
    constexpr auto N = matrix_dimension(a);
    return detail::solve<N>(a, b, x);
  }

  static auto op(A a, B b) {
    constexpr auto N = matrix_dimension(a);
    return detail::solve<N>(a, b);
  }
};
} // namespace lib

template <class A, class B>
auto solve(A a, B b) {
  return lib::solve_impl<A,B>::op(a, b);
}

template <class E, int R, int D, class S>
auto solve(E A, const Tensor<R,D,S>& b) {
  return solve(A, lib::bind(b));
}

template <int R, int D, class S, class T>
auto solve(const Tensor<R,D,S>& A, const Tensor<R/2,D,T>& b) {
  return solve(lib::bind(A), lib::bind(b));
}

template <class A, class B, class X>
int solve(A a, B b, X& x) noexcept {
  return lib::solve_impl<A,B>::op(a, b, x);
}

template <class E, int R, int D, class S, class T>
int solve(E A, const Tensor<R,D,S>& b, Tensor<R/2,D,T>& x) noexcept {
  return solve(A, lib::bind(b), x);
}

template <int R, int D, class S, class T>
int solve(const Tensor<R,D,S>& A, const Tensor<R/2,D,T>& b,
          Tensor<R/2,D,T>& x) noexcept {
  return solve(lib::bind(A), lib::bind(b), x);
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
