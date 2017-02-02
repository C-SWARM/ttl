// -*- C++ -*-
#ifndef TTL_LIBRARY_INVERSE_H
#define TTL_LIBRARY_INVERSE_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>
#include <ttl/Library/determinant.h>
#include <ttl/Library/fp_utils.h>
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
template <class E, int N = matrix_dimension<E>()>
struct inverse_impl
{
  static int getrf(double data[N*N], ipiv_t ipiv[N]) {
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR,N,N,data,N,ipiv);
  }

  static int getri(double data[N*N], ipiv_t ipiv[N]) {
    return LAPACKE_dgetri(LAPACK_COL_MAJOR,N,data,N,ipiv);
  }

  template <class M>
  static int op(E e, M& m) noexcept {
    // explicitly force a transpose into a temporary tensor on the stack... this
    // prevents lapacke from having to transpose back and forth to column major
    using namespace ttl::expressions;
    ipiv_t ipiv[N];
    auto A = force(transpose(e));
    if (int i = getrf(A.data, ipiv)) {
      return i;
    }
    if (int i = getri(A.data, ipiv)) {
      return i;
    }
    m = force(transpose(bind(A)));
    return 0;
  }

  static auto op(E e) {
    ttl::expressions::tensor_type<E> m;
    if (int i = op(e, m)) {
      throw i;
    }
    return m;
  }
};

/// Analytically expand 2x2 inverse.
template <class E>
struct inverse_impl<E, 2>
{
  template <class M>
  static int op(E f, M& m) noexcept {
    auto d = det(f);
    if (!FPNEZ(d)) {
      return 1;
    }
    auto rd = 1/d;
    m = {rd*f(1,1), -rd*f(0,1), -rd*f(1,0), rd*f(0,0)};
    return 0;
  }

  static auto op(E f) {
    ttl::expressions::tensor_type<E> m;
    if (int i = op(f, m)) {
      throw i;
    }
    return m;
  }
};

/// Analytically expand 3x3 inverse.
template <class E>
struct inverse_impl<E, 3>
{
  template <class M>
  static int op(E f, M& m) noexcept {
    auto d = det(f);
    if (!FPNEZ(d)) {
      return 1;
    }

    auto t00 = f(2,2)*f(1,1) - f(2,1)*f(1,2); //a22a11-a21a12
    auto t01 = f(2,2)*f(0,1) - f(2,1)*f(0,2); //a22a01-a21a02
    auto t02 = f(1,2)*f(0,1) - f(1,1)*f(0,2); //a12a01-a11a02
    auto t10 = f(2,2)*f(1,0) - f(2,0)*f(1,2); //a22a10-a20a12
    auto t11 = f(2,2)*f(0,0) - f(2,0)*f(0,2); //a22a00-a20a02
    auto t12 = f(1,2)*f(0,0) - f(1,0)*f(0,2); //a12a00-a10a02
    auto t20 = f(2,1)*f(1,0) - f(2,0)*f(1,1); //a21a10-a20a11
    auto t21 = f(2,1)*f(0,0) - f(2,0)*f(0,1); //a21a00-a20a01
    auto t22 = f(1,1)*f(0,0) - f(1,0)*f(0,1); //a11a00-a10a01
    auto rd = 1/d;
    m = {rd*t00, -rd*t01,  rd*t02,
         -rd*t10,  rd*t11, -rd*t12,
         rd*t20, -rd*t21,  rd*t22};
    return 0;
  }

  static auto op(E f) {
    ttl::expressions::tensor_type<E> m;
    if (int i = op(f, m)) {
      throw i;
    }
    return m;
  }
};
} // namespace lib

template <class E>
auto inverse(E e) {
  return lib::inverse_impl<E>::op(e);
}

template <int R, int D, class S>
auto inverse(const Tensor<R,D,S>& T) {
  return inverse(lib::bind(T));
}

template <class E, class M>
int inverse(E e, M& out) noexcept {
  return lib::inverse_impl<E>::op(e,out);
}

template <int R, int D, class S, class T>
int inverse(const Tensor<R,D,S>& A, Tensor<R,D,T>& out) noexcept {
  return inverse(lib::bind(A),out);
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_INVERSE_H
