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
template <class A, class B, int N = matrix_dimension<A>()>
struct solve_impl
{
  static int gesv(double a[N*N], double b[N], ipiv_t pivot[N]) {
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, N, 1, a, N, pivot, b, N);
  }

  static int gesv(float a[N*N], float b[N], ipiv_t pivot[N]) {
    return LAPACKE_sgesv(LAPACK_COL_MAJOR, N, 1, a, N, pivot, b, N);
  }

  static auto op(A a, B b) {
    // explicitly force a transpose into a temporary tensor on the stack... this
    // prevents lapacke from having to transpose back and forth to column major
    using namespace ttl::expressions;
    ipiv_t ipiv[N];
    auto ta = force(transpose(a));
    auto tb = force(b);
    if (int i = gesv(ta.data, tb.data, ipiv)) {
      throw i;
    }
    return tb;
  }

  template <class X>
  static int op(A a, B b, X& x) noexcept {
    // explicitly force a transpose into a temporary tensor on the stack... this
    // prevents lapacke from having to transpose back and forth to column major
    using namespace ttl::expressions;
    ipiv_t ipiv[N];
    auto ta = force(transpose(a));
    x = force(b);
    int i = gesv(ta.data, x.data, ipiv);
    return i;
  }
};
} // namespace lib

template <class A, class B>
auto solve(A a, B b) {
  return lib::solve_impl<A,B>::op(a, b);
}

template <class E, int R, int D, class S>
auto solve(E A, const Tensor<R,D,S>& b) {
  return solve(A, lib::bind(b));
}

template <int R, int D, class S, class T>
auto solve(const Tensor<R,D,S>& A, const Tensor<R/2,D,T>& b) {
  return solve(lib::bind(A), lib::bind(b));
}

template <class A, class B, class X>
int solve(A a, B b, X& x) noexcept {
  return lib::solve_impl<A,B>::op(a, b, x);
}

template <class E, int R, int D, class S, class T>
int solve(E A, const Tensor<R,D,S>& b, Tensor<R/2,D,T>& x) noexcept {
  return solve(A, lib::bind(b), x);
}

template <int R, int D, class S, class T>
int solve(const Tensor<R,D,S>& A, const Tensor<R/2,D,T>& b,
          Tensor<R/2,D,T>& x) noexcept {
  return solve(lib::bind(A), lib::bind(b), x);
}
}

#endif // #define TTL_LIBRARY_SOLVE_H
