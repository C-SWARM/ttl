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

template <int D, class S, class T>
Tensor<1,D,T> solve(Tensor<2,D,S> Tr, Tensor<1,D,T> b) {
  lib::detail::solve<D>(Tr.data, b.data);
  return b;
}
} // namespace lib

template <class E, class V>
auto solve(const E e, const V v) {
    return lib::solve(expressions::force(transpose(e)), expressions::force(v));
}

template <int D, class S, class T>
auto solve(Tensor<2,D,S> A, Tensor<1,D,T> b) {
  Index<'i'> i;
  Index<'j'> j;
  return solve(A(i,j), b(j));
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
