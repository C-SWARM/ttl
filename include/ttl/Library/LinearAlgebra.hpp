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

#include <numeric>                              // std::iota

namespace ttl {
namespace lib {
namespace detail {
/// Run the pivoting algorithm on a rank 2 tensor (i.e., matrix).
///
/// The pivoting operation will restructure the matrix and thus we require a
/// "real" matrix as `A` rather than simply a rank 2 tensor expression.
///
/// @tparam           M The size of the matrix.
/// @tparam      Matrix The Matrix type.
/// @tparam Permutation The type of the permutation array.
///
/// @param[in/out]    A The rank 2 tensor to pivot.
/// @param[out]    perm The permutation.
/// @param[in]        j The column that we are processing.
///
/// @returns            The row index that we swapped, for debugging purposes.
template <int M, class Matrix, class Permutation>
static inline int
pivot(Matrix&& A, Permutation&& perm, const int j)
{
  // Find the maximum magnitude in column j
  using std::abs;
  using std::swap;
  auto i = j;
  auto max = abs(A(j,j));
  for (auto ii = j + 1; ii < M; ++ii) {
    if (abs(A(ii, j)) > max) {
      max = abs(A(ii, j));
      i = ii;
    }
  }

  for (auto jj = 0; jj < M; ++jj) {
    swap(A(i, jj), A(j, jj));
  }

  swap(perm(j), perm(i));

  return i;
}

/// Factor an expression.
///
/// @tparam           M The dimension of the matrix.
/// @tparam      Matrix The matrix type.
/// @tparam Permutation The type to store the permutation.
///
/// @param[in/out]    A The rank 2 tensor expression to factor.
/// @param[out]    perm The permutation.
///
/// @returns          0 If the operation succeeded.
///            positive Indicates that the factorization would have tried to
///                     divide by zero in the returned row.
template <int M, class Matrix, class Permutation>
static inline int
lu_kij_pp(Matrix&& A, Permutation&& perm)
{
  for (auto k = 0; k < M - 1; ++k) {
    pivot<M>(A, perm, k);
    for (auto i = k + 1; i < M; ++i) {
      auto z = A(i, k) /= A(k, k);
      for (auto j = k + 1; j < M; ++j) {
        A(i, j) -= z * A(k, j);
      }
    }
  }

  // Check the diagonal for 0s. We do this here rather than terminating pivoting
  // in the loop to avoid an early loop exit and possible GPU divergence.
  int e = 0;
  for (auto i = 0; i < M; ++i) {
    e = (not e and A(i, i) == 0) ? i : e;
  }
  return e;
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
/// @tparam           M The size of the matrix and vector.
/// @tparam      Matrix The matrix type, must support `operator()(int,int)`.
/// @tparam      Vector The vector type, must support `operator()(int)`.
///
/// @param[in/out]    A The matrix, will be factored and permuted into LUP.
/// @param[in/out]    b The vector, will be written with the solution.
///
/// @returns          0 On success.
///            non-zero The matrix is singular in that one of the diagonal
///                     elements is identically 0. The returned value is the row
///                     id that is 0 (1-based indexing for the id).
template <int M, class Matrix, class Vector>
static inline int
solve(Matrix&& A, Vector&& b) noexcept
{
  // 1. Perform LU factorization on the matrix. If this fails then we have a
  //    singular matrix. This permutes the vector at the same time.
  if (auto i = lu_kij_pp<M>(A, b)) {
    return i;
  }

  // 2. Lower triangular solve.
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < i; ++j) {
      b(i) -= A(i, j) * b(j);
    }
  }

  // 3. Upper triangular solve.
  for (auto i = M - 1; i >= 0; --i) {
    for (auto j = i + 1; j < M; ++j) {
      b(i) -= A(i, j) * b(j);
    }
    b(i) /= A(i, i);
  }

  return 0;
}

template <int M, class Matrix, class Inverse>
static inline int
inverse(Matrix&& A, Inverse&& inv) noexcept
{
  // 1. Allocate and initialize a permutation.
  struct Permutation {
    Permutation() {
      std::iota(std::begin(data), std::end(data), 0);
    }

    int& operator()(int i) {
      return data[i];
    }

    void operator()(Inverse& inv) {
      using T = std::remove_reference_t<decltype(inv(0,0))>;
      for (auto i = 0; i < M; ++i) {
        inv(i, data[i]) = T{1};
      }
    }

    int data[M];
  } perm;

  // 2. Perform LU factorization on the matrix, and test for failure.
  if (auto i = lu_kij_pp<M>(A, perm)) {
    return i;
  }

  // 3. Permute the Identity matrix.
  perm(inv);

  // 4. Lower triangular solve.
  for (auto k = 0; k < M; ++k) {
    for (auto i = 0; i < M; ++i) {
      for (auto j = 0; j < i; ++j) {
        inv(i, k) -= A(i, j) * inv(j, k);
      }
    }
  }

  // 5. Upper triangular solve.
  for (auto k = 0; k < M; ++k) {
    for (auto i = M - 1; i >= 0; --i) {
      for (auto j = i + 1; j < M; ++j) {
        inv(i, k) -= A(i, j) * inv(j, k);
      }
      inv(i, k) /= A(i, i);
    }
  }

  return 0;
}
} // namespace detail
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_LINEAR_ALGEBRA_HPP
