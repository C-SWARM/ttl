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
template <class MatrixExpression, int N>
struct inverse_impl
{
  template <int M, class T>
  static int op(MatrixExpression A, Tensor<2,M,T>& inv) noexcept {
    static_assert(N == M, "Dimensions must match");
    return detail::inverse<M, T>(A, inv);
  }

  static expressions::tensor_type<MatrixExpression> op(MatrixExpression A) {
    using expressions::force;
    expressions::tensor_type<MatrixExpression> inv = {};
    if (int i = op(force(A), inv)) {
      throw i;
    }
    return inv;
  }
};

/// Analytically expand 2x2 inverse.
template <class MatrixExpression>
struct inverse_impl<MatrixExpression, 2>
{
  template <int M, class T>
  static int op(MatrixExpression A, Tensor<2,M,T>& inv) noexcept {
    auto d = det(A);
    if (!FPNEZ(d)) {
      return 1;
    }
    auto rd = 1/d;
    inv(0,0) =  rd * A(1,1);
    inv(0,1) = -rd * A(0,1);
    inv(1,0) = -rd * A(1,0);
    inv(1,1) =  rd * A(0,0);
    return 0;
  }

  static expressions::tensor_type<MatrixExpression> op(MatrixExpression A) {
    using expressions::force;
    expressions::tensor_type<MatrixExpression> inv;
    if (int i = op(force(A), inv)) {
      throw i;
    }
    return inv;
  }
};

/// Analytically expand 3x3 inverse.
template <class MatrixExpression>
struct inverse_impl<MatrixExpression, 3>
{
  template <int M, class T>
  static int op(MatrixExpression A, Tensor<2,M,T>& inv) noexcept {
    auto d = det(A);
    if (!FPNEZ(d)) {
      return 1;
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

  static expressions::tensor_type<MatrixExpression> op(MatrixExpression A) {
    using expressions::force;
    expressions::tensor_type<MatrixExpression> inv;
    if (int i = op(force(A), inv)) {
      throw i;
    }
    return inv;
  }
};
} // namespace lib

template <bool Zero, class Matrix, class Inverse>
int inverse(Matrix&& A, Inverse& inv) noexcept {
  using namespace lib;
  static constexpr auto N = matrix_dimension(A);
  if (Zero) inv = {};
  return detail::inverse<N>(as_matrix(std::forward<Matrix>(A)), as_matrix(inv));
}

template <int R, int N, class T, bool Zero = true>
int inverse(Tensor<R,N,T>&& A, Tensor<R,N,T>& inv) noexcept {
  return inverse<Zero>(std::move(A), inv);
}

template <int R, int N, class T, bool Zero = true>
int inverse(const Tensor<R,N,T>& A, Tensor<R,N,T>& inv) noexcept {
  using expressions::force;
  return inverse<Zero>(force(A), inv);
}

template <int R, int N, class T, class MatrixExpression, bool Zero = true>
int inverse(MatrixExpression A, Tensor<R,N,T>& inv) noexcept {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(A) == R, "Expression A must compatible with inv");
  static_assert(dimension(A) == N, "Dimensions must match");
  return inverse<Zero>(force(A), inv);
}

template <int R, int N, class T>
auto inverse(const Tensor<R,N,T>& A) {
  using expressions::force;
  expressions::tensor_type<Tensor<R,N,T>> inv = {};
  if (int i = inverse<false>(force(A), inv)) {
    throw i;
  }
  return inv;
}

template <int R, int N, class T>
auto inverse(Tensor<R,N,T>&& A) {
  expressions::tensor_type<Tensor<R,N,T>> inv = {};
  if (int i = inverse<false>(std::move(A), inv)) {
    throw i;
  }
  return inv;
}

template <class MatrixExpression>
auto inverse(MatrixExpression A) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(A) == 2, "Expression A must be a matrix");
  return inverse(force(A));
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_INVERSE_H
