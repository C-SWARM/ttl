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
#include <ttl/Library/LinearAlgebra.hpp>
#include <ttl/Library/matrix.h>
#include <ttl/Expressions/force.h>
#include <ttl/Expressions/traits.h>

namespace ttl {
namespace lib {
/// The core solve operation captures A and b as a Matrix and Vector and
/// forwards to the linear algebra solver.
template <class Matrix, class Vector>
int solve(Matrix&& A, Vector&& b) {
  using expressions::rank_t;
  using expressions::dimension_t;
  static_assert(rank_t<Matrix>::value == 2 * rank_t<Vector>::value,
                "A must be a matrix");
  static_assert(dimension_t<Matrix>::value == dimension_t<Vector>::value,
                "A and b must be the same size");

  using namespace lib;
  static constexpr auto M = matrix_dimension_t<Matrix>::value;
  return detail::solve<M>(as_matrix(std::forward<Matrix>(A)),
                          as_vector(std::forward<Vector>(b)));
}
} // namespace lib

/// The three-argument interface copies or evaluates A and b and returns the
/// solution in x.
template <class Matrix, class B, class X>
int solve(Matrix&& A, B&& b, X& x) noexcept {
  using expressions::rank_t;
  using expressions::dimension_t;
  static_assert(rank_t<B>::value == rank_t<X>::value,
                "b and x must be the same rank");
  static_assert(dimension_t<B>::value == dimension_t<X>::value,
                "b and x must be the same size");

  using expressions::force;
  x = force(std::forward<B>(b));
  return lib::solve(force(std::forward<Matrix>(A)), x);
}

/// The two-argument interface copies or evaluates A and b and returns a
/// solution.
template <class Matrix, class Vector>
auto solve(Matrix&& A, Vector&& b) {
  using expressions::force;
  auto x = force(std::forward<Vector>(b));
  if (auto i = lib::solve(force(std::forward<Matrix>(A)), x)) {
    throw i;
  }
  return x;
}
} // namespace lib

#endif // #define TTL_LIBRARY_SOLVE_H
