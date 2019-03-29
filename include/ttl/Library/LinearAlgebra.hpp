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
// Basic LUP code contributed by Andrew Lumsdaine
// -----------------------------------------------------------------------------

#ifndef TTL_LIBRARY_LINEAR_ALGEBRA_HPP
#define TTL_LIBRARY_LINEAR_ALGEBRA_HPP

#include "ttl/Library/matrix.h"
#include <numeric>                              // std::iota
#include <type_traits>
#include <vector>

#include <iostream>

namespace ttl {
namespace lib {
namespace detail {
/// Run the pivoting algorithm on a rank 2 tensor (i.e., matrix).
///
/// The pivoting operation will restructure the matrix and thus we require a
/// "real" matrix as `A` rather than simply a rank 2 tensor expression.
///
/// @tparam      Matrix The Matrix type.
/// @tparam Permutation The type of the permutation array.
/// @tparam       Index The type of the index into a Tensor dimension.
///
/// @param[in/out]    A The rank 2 tensor to pivot.
/// @param[out]    perm The permutation.
/// @param[in]        j The column that we are pivoting on.
template <class Matrix, class Permutation, class Index>
static inline void
pivot(Matrix& A, Permutation& perm, const Index j)
{
  constexpr auto M = expressions::dimension(A);
  using          T = expressions::scalar_type<Matrix>;

  // Find the maximum magnitude in column j
  using std::abs;
  using std::swap;
  Index i = j;
  T max = 0.0;
  for (auto ii = j; ii < M; ++ii) {
    if (abs(A(ii, j)) > max) {
      max = abs(A(ii, j));
      i = ii;
    }
  }

  // Record the permutation
  swap(perm[j], perm[i]);

  // Swap the row i with the row j
  if (i != j) {
    for (auto jj = 0; jj < M; ++jj) {
      // operator()(i,j) isn't naturally l-value (can't take its address)
      T t = A(i, jj);
      A(i, jj) = A(j, jj);
      A(j, jj) = t;
    }
  }
}

/// Factor an expression.
///
/// @tparam      Matrix The matrix type.
/// @tparam Permutation The type to store the permutation.
///
/// @param[in/out]    A The rank 2 tensor expression to factor.
/// @param[out]    perm The permutation.
///
/// @returns          0 If the operation succeeded.
///            positive Indicates that the factorization would have tried to
///                     divide by zero in the returned row.
template <class Matrix, class Permutation>
static inline auto
lu_ikj_pp(Matrix& A, Permutation& perm)
{
  constexpr auto M = expressions::dimension(A);
  using          T = expressions::scalar_type<Matrix>;
  using      Index = decltype(M);

  for (auto i = 0; i < M; ++i) {
    pivot(A, perm, i);

    for (auto k = 0; k < i; ++k) {
      T z = A(i, k) / A(k, k);
      A(i, k) = z;

      for (auto j = k + 1; j < M; ++j) {
        A(i, j) = A(i, j) - z * A(k, j);
      }
    }

    if (A(i,i) == T{0}) {
      return i + 1;
    }
  }

  return Index{0};
}

/// Solve a system.
///
/// This will perform LUP on the matrix `A`, permute the vector `b`, and then
/// perform the solve via lower and upper substitution. The result is returned
/// in b.
///
/// It will return `0` on success and a positive number, `j`, between `1 and `M`
/// to indicate if we have a diagonal element in `U(j-1,j-1)` that was `0`
/// during factorization. If it returns non-zero then the results of `A` and `b`
/// are invalidated.
///
/// The result is returned in the space for `b`.
///
/// The Matrix A is any class that has a `rank(2)`, however it must have storage
/// allocated behind it. The Vector `b` is any class that has a `rank(1)`, but
/// also must have storage.
///
///
template <class Matrix, class Vector>
static inline int
solve(Matrix&& A, Vector& b) noexcept
{
  // Check the ranks to make sure that we're getting a matrix and a vector.
  using expressions::rank;
  static_assert(rank(A) == 2, "solve requires a matrix A");
  static_assert(rank(b) == 1, "solve requires a vector b");

  // Check the dimensions to make sure that they match.
  using expressions::dimension;
  static constexpr auto M = dimension(A);
  static_assert(M == dimension(b), "Dimension mismatch");

  // Make sure that we can deal with the scalar type
  using T = expressions::scalar_type<Matrix>;
  static_assert(std::is_floating_point<T>::value,
                "Can only solve floating point systems");

  // 1. Allocate and initialize a permutation matrix.
  using Index = decltype(dimension(A));
  using std::begin;
  using std::end;
  Index perm[M];
  std::iota(begin(perm), end(perm), 0);

  // 2. Perform LU factorization on the matrix, recording the permutation. If
  //    this fails then we have a singular matrix.
  if (auto i = lu_ikj_pp(A, perm)) {
    return i;
  }

  // 3. Permute the vector, `b`.
  //    @todo we could do this in-place.
  Vector x;
  for (Index i = 0; i < M; ++i) {
    x(perm[i]) = b(i);
  }

  // Lower triangular
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < i; ++j) {
      x(i) = x(i) - A(i, j) * x(j);
    }
  }

  // Upper triangular
  for (auto i = M - 1; i >= 0; --i) {
    for (auto j = i + 1; j < M; ++j) {
      x(i) = x(i) - A(i, j) * x(j);
    }
    x(i) = x(i) / A(i, i);
  }

  b = std::move(x);
  return 0;
}

template <class Expression, class Inverse>
static inline int
invert(Expression e, Inverse& inv, bool zero = false) noexcept
{
  // using expressions::dimension;
  // using expressions::rank;
  // using T = expressions::scalar_type<Expression>;
  // static_assert(std::is_floating_point<T>::value,
  //               "Can only invert floating point systems");
  // static constexpr auto M = matrix_dimension(e);

  // auto A = to_matrix(e);

  // std::vector<decltype(M)> perm(M);
  // if (auto i = lu_ikj_pp(A, perm)) {
  //   return i;
  // }

  // if (zero) {
  //   inv = {};
  // }

  // for (auto i = 0; i < M; ++i) {
  //   assert(perm[i] < e);
  //   inv(perm[i], i) = 1;
  // }

  return 0;
}
} // namespace detail
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_LINEAR_ALGEBRA_HPP
