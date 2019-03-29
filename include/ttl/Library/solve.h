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
#include <ttl/Library/binder.h>
#include <ttl/Expressions/force.h>
#include <ttl/Expressions/traits.h>

namespace ttl {
template <int N, class T>
using Matrix = Tensor<2, N, T>;

template <int N, class T>
using Vector = Tensor<1, N, T>;

template <int N, class T>
int solve(const Matrix<N,T>& A, const Vector<N,T>& b, Vector<N,T>& x) noexcept {
  return lib::detail::solve(expressions::force(A), (x = b));
}

template <int N, class T>
int solve(Matrix<N,T>&& A, const Vector<N,T>& b, Vector<N,T>& x) noexcept {
  return lib::detail::solve(std::move(A), (x = b));
}

template <int N, class T>
int solve(const Matrix<N,T>& A, Vector<N,T>&& b, Vector<N,T>& x) noexcept {
  return lib::detail::solve(expressions::force(A), (x = std::move(b)));
}

template <int N, class T>
int solve(Matrix<N,T>&& A, Vector<N,T>&& b, Vector<N,T>& x) noexcept {
  return lib::detail::solve(std::move(A), (x = std::move(b)));
}

template <int N, class T, class M>
int solve(M A, const Vector<N,T>& b, Vector<N,T>& x)
  noexcept
{
  return solve(expressions::force(A), b, x);
}

template <int N, class T, class M>
int solve(M A, Vector<N,T>&& b, Vector<N,T>& x)
  noexcept
{
  return solve(expressions::force(A), std::move(b), x);
}

template <int N, class T, class V>
int solve(const Matrix<N,T>& A, V b, Vector<N,T>& x)
  noexcept
{
  return solve(A, expressions::force(b), x);
}

template <int N, class T, class V>
int solve(Matrix<N,T>&& A, V b, Vector<N,T>& x)
  noexcept
{
  return solve(std::move(A), expressions::force(b), x);
}

template <int N, class T, class M, class V>
int solve(M A, V b,
          Vector<N,T>& x) noexcept
{
  return solve(expressions::force(A), expressions::force(b), x);
}

template <int N, class T>
Vector<N,T> solve(const Matrix<N,T>& A, const Vector<N,T>& b) {
  Vector<N,T> x = b;
  if (auto i = lib::detail::solve(expressions::force(A), x)) {
    throw i;
  }
  return x;
}

template <int N, class T>
Vector<N,T> solve(Matrix<N,T>&& A, const Vector<N,T>& b) {
  Vector<N,T> x = b;
  if (auto i = lib::detail::solve(std::move(A), x)) {
    throw i;
  }
  return x;
}

template <int N, class T>
Vector<N,T> solve(const Matrix<N,T>& A, Vector<N,T>&& b) {
  Vector<N,T> x = std::move(b);
  if (auto i = lib::detail::solve(expressions::force(A), x)) {
    throw i;
  }
  return x;
}

template <int N, class T>
Vector<N,T> solve(Matrix<N,T>&& A, Vector<N,T>&& b) {
  Vector<N,T> x = std::move(b);
  if (auto i = lib::detail::solve(std::move(A), x)) {
    throw i;
  }
  return x;
}

template <class M, int N, class T>
Vector<N,T> solve(M A, const Vector<N,T>& b) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(A) == 2, "Expression must be a matrix");
  static_assert(dimension(A) == N, "Dimensions must match");
  return solve(force(A), b);
}

template <class M, int N, class T>
Vector<N,T> solve(M A, Vector<N,T>&& b) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(A) == 2, "Expression must be a matrix");
  static_assert(dimension(A) == N, "Dimensions must match");
  return solve(force(A), std::move(b));
}

template <int N, class T, class V>
Vector<N,T> solve(const Matrix<N,T>& A, V b) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(b) == 1, "Expression must be a vector");
  static_assert(dimension(b) == N, "Dimensions must match");
  return solve(A, force(b));
}

template <int N, class T, class V>
Vector<N,T> solve(Matrix<N,T>&& A, V b) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(b) == 1, "Expression must be a vector");
  static_assert(dimension(b) == N, "Dimensions must match");
  return solve(std::move(A), force(b));
}

template <class M, class V>
auto solve(M A, V b) {
  using expressions::rank;
  using expressions::dimension;
  using expressions::force;
  static_assert(rank(A) == 2, "Expression must be a vector");
  static_assert(rank(b) == 1, "Expression must be a vector");
  static_assert(dimension(A) == dimension(b), "Dimensions must match");
  return solve(force(A), force(b));
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
