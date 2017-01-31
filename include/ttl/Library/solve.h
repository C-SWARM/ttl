// -*- C++ -*-
#ifndef TTL_LIBRARY_SOLVE_H
#define TTL_LIBRARY_SOLVE_H

#include <ttl/ttl.h>

#ifdef HAVE_MKL_H
#include <mkl.h>
using ipiv_t = MKL_INT;
#else
#include <lapacke.h>
using ipiv_t = lapack_int;
#endif

namespace ttl {
namespace lib {
namespace detail {
template <int D>
int solve(float *a, float *b) {
  ipiv_t pivot[D];
  return LAPACKE_sgesv(LAPACK_COL_MAJOR, D, 1, a, D, pivot, b, D);
}

template <int D>
int solve(double *a, double *b) {
  ipiv_t pivot[D];
  return LAPACKE_dgesv(LAPACK_COL_MAJOR, D, 1, a, D, pivot, b, D);
}
} // namespace detail
} // namespace lib

template <int D, class S, class T>
Tensor<1,D,T> solve(Tensor<2,D,S> A, Tensor<1,D,T> b) {
  lib::detail::solve<D>(A.data, b.data);
  return b;
}

template <class E, class V>
auto solve(const E e, const V v) {
  return solve(expressions::force(e), expressions::force(v));
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
