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
#ifndef TTL_LIBRARY_INVERSE_H
#define TTL_LIBRARY_INVERSE_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/LinearAlgebra.hpp>
#include <ttl/Library/binder.h>
#include <ttl/Library/determinant.h>
#include <ttl/Library/fp_utils.h>
#include <ttl/Library/matrix.h>

namespace ttl {
namespace lib {
template <int N>
struct inverse_impl
{
  template <class Matrix, class Inverse>
  static int op(Matrix&& A, Inverse& inv) noexcept {
    using expressions::force;
    return detail::inverse<N>(as_matrix(force(std::forward<Matrix>(A))),
                              as_matrix(inv));
  }
};

/// Analytically compute 2x2 inverse.
template <>
struct inverse_impl<2>
{
  template <class Matrix, class Inverse>
  static int op(Matrix&& A, Inverse& inv) noexcept {
    auto d = det(A);
    if (!FPNEZ(d)) {
      return -1;
    }
    auto rd = 1/d;
    inv(0,0) =  rd * A(1,1);
    inv(0,1) = -rd * A(0,1);
    inv(1,0) = -rd * A(1,0);
    inv(1,1) =  rd * A(0,0);
    return 0;
  }
};

/// Analytically compute 3x3 inverse.
template <>
struct inverse_impl<3>
{
  template <class Matrix, class Inverse>
  static int op(Matrix&& A, Inverse& inv) noexcept {
    auto d = det(A);
    if (!FPNEZ(d)) {
      return -1;
    }
    auto rd = 1/d;
    inv(0,0) =  rd * (A(2,2) * A(1,1) - A(2,1) * A(1,2)); //a22a11-a21a12
    inv(0,1) = -rd * (A(2,2) * A(0,1) - A(2,1) * A(0,2)); //a22a01-a21a02
    inv(0,2) =  rd * (A(1,2) * A(0,1) - A(1,1) * A(0,2)); //a12a01-a11a02
    inv(1,0) = -rd * (A(2,2) * A(1,0) - A(2,0) * A(1,2)); //a22a10-a20a12
    inv(1,1) =  rd * (A(2,2) * A(0,0) - A(2,0) * A(0,2)); //a22a00-a20a02
    inv(1,2) = -rd * (A(1,2) * A(0,0) - A(1,0) * A(0,2)); //a12a00-a10a02
    inv(2,0) =  rd * (A(2,1) * A(1,0) - A(2,0) * A(1,1)); //a21a10-a20a11
    inv(2,1) = -rd * (A(2,1) * A(0,0) - A(2,0) * A(0,1)); //a21a00-a20a01
    inv(2,2) =  rd * (A(1,1) * A(0,0) - A(1,0) * A(0,1)); //a11a00-a10a01
    return 0;
  }
};

template <class Matrix, class Inverse>
int inverse(Matrix&& A, Inverse& inv) noexcept {
  using expressions::rank;
  using expressions::dimension;
  static_assert(rank(A) == rank(inv), "A and inv must have the same rank");
  static_assert(dimension(A) == dimension(inv), "A and inv must have the same dimension");

  static constexpr auto N = matrix_dimension(A);
  return inverse_impl<N>::op(std::forward<Matrix>(A), inv);
}
} // namespace lib

template <class Matrix, class Inverse>
int inverse(Matrix&& A, Inverse& inv, bool zero = true) noexcept {
  if (zero) inv = {};
  return lib::inverse(std::forward<Matrix>(A), inv);
}

template <class Matrix>
auto inverse(Matrix&& A) {
  expressions::tensor_type<Matrix> inv = {};
  if (int i = lib::inverse(std::forward<Matrix>(A), inv)) {
    throw i;
  }
  return inv;
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_INVERSE_H
