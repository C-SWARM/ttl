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
#ifndef TTL_LIBRARY_MATRIX_H
#define TTL_LIBRARY_MATRIX_H

#include <ttl/Expressions/force.h>
#include <ttl/Expressions/traits.h>
#include <ttl/util/pow.h>
#include <ttl/util/log2.h>
#include <iostream>
#include <iomanip>

namespace ttl {
namespace lib {
template <class E>
struct matrix_dimension_t {
  static constexpr int   Rank = expressions::rank_t<E>::value;
  static constexpr int l2Rank = util::log2<Rank>::value;
  static constexpr int      N = expressions::dimension_t<E>::value;
  static constexpr int  value = util::pow(N, l2Rank);
};

template <int N, class T>
constexpr auto as_matrix(Tensor<2,N,T>&& A) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i, int j) -> S& {
    return A[i][j];
  };
}

template <int N, class T>
constexpr auto as_matrix(Tensor<2,N,T>& A) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i, int j) -> S& {
    return A[i][j];
  };
}

template <int N, class T>
constexpr auto as_matrix(const Tensor<2,N,T>& A) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i, int j) -> S {
    return A[i][j];
  };
}

template <int N, class T>
constexpr auto as_matrix(Tensor<4,N,T>&& A) {
  using S = expressions::scalar_type<Tensor<4,N,T>>;
  return [&](int i, int j) -> S& {
    int m = i / N;
    int n = i % N;
    int o = j / N;
    int p = j % N;
    return A[m][n][o][p];
  };
}

template <int N, class T>
constexpr auto as_matrix(Tensor<4,N,T>& A) {
  using S = expressions::scalar_type<Tensor<4,N,T>>;
  return [&](int i, int j) -> S& {
    int m = i / N;
    int n = i % N;
    int o = j / N;
    int p = j % N;
    return A[m][n][o][p];
  };
}

template <int N, class T>
constexpr auto as_matrix(const Tensor<4,N,T>& A) {
  using S = expressions::scalar_type<Tensor<4,N,T>>;
  return [&](int i, int j) -> S {
    int m = i / N;
    int n = i % N;
    int o = j / N;
    int p = j % N;
    return A[m][n][o][p];
  };
}

template <int N, class T>
constexpr auto as_vector(Tensor<1,N,T>&& x) {
  using S = expressions::scalar_type<Tensor<1,N,T>>;
  return [&](int i) -> S& {
    return x[i];
  };
}

template <int N, class T>
constexpr auto as_vector(Tensor<1,N,T>& x) {
  using S = expressions::scalar_type<Tensor<1,N,T>>;
  return [&](int i) -> S& {
    return x[i];
  };
}

template <int N, class T>
constexpr auto as_vector(const Tensor<1,N,T>& x) {
  using S = expressions::scalar_type<Tensor<1,N,T>>;
  return [&](int i) -> S {
    return x[i];
  };
}

template <int N, class T>
constexpr auto as_vector(Tensor<2,N,T>&& x) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i) -> S& {
    int m = i / N;
    int n = i % N;
    return x[m][n];
  };
}

template <int N, class T>
constexpr auto as_vector(Tensor<2,N,T>& x) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i) -> S& {
    int m = i / N;
    int n = i % N;
    return x[m][n];
  };
}

template <int N, class T>
constexpr auto as_vector(const Tensor<2,N,T>& x) {
  using S = expressions::scalar_type<Tensor<2,N,T>>;
  return [&](int i) -> S {
    int m = i / N;
    int n = i % N;
    return x[m][n];
  };
}

template <int M, class Matrix>
std::ostream& printMatrix(std::ostream& os, Matrix&& A) {
  os << std::left << std::setw(4) << " ";
  for (auto j = 0; j < M; ++j) {
    os << std::left << std::setw(20) << j;
  }
  os << "\n";

  for (auto i = 0; i < M; ++i) {
    os << std::left << std::setw(4) << i;
    for (auto j = 0; j < M; ++j) {
      os << std::left << std::setw(20) << A(i,j);
    }
    os << "\n";
  }
  os << "\n";
  return os;
}
} // namespace lib
} // namespace ttl

#endif // #define TTL_LIBRARY_MATRIX_H
