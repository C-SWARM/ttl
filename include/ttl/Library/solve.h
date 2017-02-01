// -*- C++ -*-
#ifndef TTL_LIBRARY_SOLVE_H
#define TTL_LIBRARY_SOLVE_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>
#include <ttl/Library/matrix.h>

#ifdef HAVE_MKL_H
#include <mkl.h>
using ipiv_t = MKL_INT;
#else
#include <lapacke.h>
using ipiv_t = lapack_int;
#endif

namespace ttl {
namespace lib {
template <class E, class V, int N = matrix_dimension<E>()>
struct solve_impl
{
  static int gesv(double A[N*N], double b[N], ipiv_t pivot[N]) {
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, N, 1, A, N, pivot, b, N);
  }

  static int gesv(float A[N*N], float b[N], ipiv_t pivot[N]) {
    return LAPACKE_sgesv(LAPACK_COL_MAJOR, N, 1, A, N, pivot, b, N);
  }

  static auto op(E e, V v) {
    // explicitly force a transpose into a temporary tensor on the stack... this
    // prevents lapacke from having to transpose back and forth to column major
    using namespace ttl::expressions;
    ipiv_t ipiv[N];
    auto A = force(transpose(e));
    auto b = force(v);
    if (int i = gesv(A.data, b.data, ipiv)) {
      assert(false);                            // error message?
    }
    return b;
  }
};
} // namespace lib

template <class E, class V>
auto solve(E e, V v) {
  return lib::solve_impl<E,V>::op(e, v);
}

template <class E, int R, int D, class S>
auto solve(E A, const Tensor<R,D,S>& b) {
  return solve(A, lib::bind(b));
}

template <int R, int D, class S, class T>
auto solve(const Tensor<R,D,S>& A, const Tensor<R/2,D,T>& b) {
  return solve(lib::bind(A), lib::bind(b));
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
