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
#ifndef TTL_LIBRARY_LINEAR_ALGEBRA_HPP
#define TTL_LIBRARY_LINEAR_ALGEBRA_HPP

#include <ttl/config.h>

#if defined(TTL_WITH_LAPACK) && defined(TTL_HAVE_MKL_H)
#include <mkl.h>
#elif defined(TTL_WITH_LAPACK)
#include <lapacke.h>
#else
#include <ttl/contrib/resice.hpp>
#endif

namespace ttl {
namespace lib {
namespace detail {
#ifdef TTL_WITH_LAPACK
#ifdef TTL_HAVE_MKL_H
using ipiv_t = MKL_INT;
#else
using ipiv_t = lapack_int;
#endif

template <int N>
CUDA_BOTH static inline int getrf(double data[N*N], ipiv_t ipiv[N]) {  
  #ifdef __CUDA_ARCH__
    return 1;
  #else  
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR,N,N,data,N,ipiv);
  #endif
}

template <int N>
CUDA_BOTH static inline int getri(double data[N*N], ipiv_t ipiv[N]) {
  #ifdef __CUDA_ARCH__
    return 1;
  #else  
    return LAPACKE_dgetri(LAPACK_COL_MAJOR,N,data,N,ipiv);
  #endif
}

template <int N>
CUDA_BOTH static inline int gesv(double a[N*N], double b[N], ipiv_t pivot[N]) {
  #ifdef __CUDA_ARCH__
    return 1;
  #else  
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, N, 1, a, N, pivot, b, N);
  #endif
}

template <int N>
CUDA_BOTH static inline int gesv(float a[N*N], float b[N], ipiv_t pivot[N]) {
  #ifdef __CUDA_ARCH__
    
  #else  
    return LAPACKE_sgesv(LAPACK_COL_MAJOR, N, 1, a, N, pivot, b, N);
  #endif  
}

template <int N, class A, class B, class X>
CUDA_BOTH static inline int solve(A a, B b, X& x) noexcept {
  // explicitly force a transpose into a temporary tensor on the stack... this
  // prevents lapacke from having to transpose back and forth to column major
  using namespace ttl::expressions;
  #ifdef __CUDA_ARCH__
    return 1;
  #else  
    ipiv_t ipiv[N];
  #endif 
  
  auto ta = force(transpose(a));
  x = force(b);
  int i = gesv<N>(ta.data, x.data, ipiv);
  return i;
}

template <int N, class A, class B>
CUDA_BOTH static inline auto solve(A a, B b) {
  // explicitly force a transpose into a temporary tensor on the stack... this
  // prevents lapacke from having to transpose back and forth to column major
  using namespace ttl::expressions;
  ipiv_t ipiv[N];
  auto ta = force(transpose(a));
  auto tb = force(b);

  #ifdef __CUDA_ARCH__
    gesv<N>(ta.data, tb.data, ipiv);
  #else
    if (int i = gesv<N>(ta.data, tb.data, ipiv)) {
      throw i;
    }
  #endif // check for cuda architecture
  return tb;
}

template <int N, class E, class M>
CUDA_BOTH static inline int invert(E e, M& m) noexcept {
  // explicitly force a transpose into a temporary tensor on the stack... this
  // prevents lapacke from having to transpose back and forth to column major
  using namespace ttl::expressions;
  ipiv_t ipiv[N];
  auto A = force(transpose(e));
  if (int i = getrf<N>(A.data, ipiv)) {
    return i;
  }
  if (int i = getri<N>(A.data, ipiv)) {
    return i;
  }
  m = force(transpose(bind(A)));
  return 0;
}
#else
template <int N, class E, class M>
CUDA_BOTH static inline int invert(E e, M& m) noexcept {
  static constexpr Index<'i'> i;
  static constexpr Index<'j'> j;
  ttl::expressions::tensor_type<E> A = force(e);
  Tensor<2, N, ttl::expressions::scalar_type<E>> b = identity<N>(i,j);
  long ipiv[N];
  return reseni_rovnic(A.data, m.data, b.data, N, N, 2, ipiv);
}

template <int N, class A, class B, class X>
CUDA_BOTH static inline int solve(A a, B b, X& x) noexcept {
  ttl::expressions::tensor_type<A> fA = force(a);
  ttl::expressions::tensor_type<B> fb = force(b);
  long ipiv[N];
  return reseni_rovnic(fA.data, x.data, fb.data, N, 1, 2, ipiv);
}

template <int N, class A, class B>
CUDA_BOTH static inline auto solve(A a, B b) {
  ttl::expressions::tensor_type<A> fA = force(a);
  ttl::expressions::tensor_type<B> fb = force(b);
  ttl::expressions::tensor_type<B> x;
  long ipiv[N];
  #ifdef __CUDA_ARCH__
    reseni_rovnic(fA.data, x.data, fb.data, N, 1, 2, ipiv);
  #else
    if (int i = reseni_rovnic(fA.data, x.data, fb.data, N, 1, 2, ipiv)) {
      throw i;
    }
  #endif // check for cuda architecture
  return x;
}
#endif
}
}
}

#endif // #define TTL_LIBRARY_LINEAR_ALGEBRA_HPP
