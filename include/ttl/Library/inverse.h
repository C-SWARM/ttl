// -*- C++ -*-
#ifndef TTL_LIBRARY_INVERSE_H
#define TTL_LIBRARY_INVERSE_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>
#include <ttl/Library/determinant.h>
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

  static auto op(E e) {
    // explicitly force a transpose into a temporary tensor on the stack... this
    // prevents lapacke from having to transpose back and forth to column major
    using namespace ttl::expressions;
    ipiv_t ipiv[N];
    auto A = force(transpose(e));
    if (int i = getrf(A.data, ipiv)) {
      assert(false);
    }
    if (int i = getri(A.data, ipiv)) {
      assert(false);
    }
    return force(transpose(bind(A)));
  }
};

/// Analytically expand 2x2 inverse.
template <class E>
struct inverse_impl<E, 2>
{
  static ttl::expressions::tensor_type<E> op(E f) {
    auto d = 1/det(f);
    return {d*f(1,1), -d*f(0,1), -d*f(1,0), d*f(0,0)};
  }
};

/// Analytically expand 3x3 inverse.
template <class E>
struct inverse_impl<E, 3>
{
  static ttl::expressions::tensor_type<E> op(E f) {
    auto t00 = f(2,2)*f(1,1) - f(2,1)*f(1,2); //a22a11-a21a12
    auto t01 = f(2,2)*f(0,1) - f(2,1)*f(0,2); //a22a01-a21a02
    auto t02 = f(1,2)*f(0,1) - f(1,1)*f(0,2); //a12a01-a11a02
    auto t10 = f(2,2)*f(1,0) - f(2,0)*f(1,2); //a22a10-a20a12
    auto t11 = f(2,2)*f(0,0) - f(2,0)*f(0,2); //a22a00-a20a02
    auto t12 = f(1,2)*f(0,0) - f(1,0)*f(0,2); //a12a00-a10a02
    auto t20 = f(2,1)*f(1,0) - f(2,0)*f(1,1); //a21a10-a20a11
    auto t21 = f(2,1)*f(0,0) - f(2,0)*f(0,1); //a21a00-a20a01
    auto t22 = f(1,1)*f(0,0) - f(1,0)*f(0,1); //a11a00-a10a01
    auto d = 1/det(f);
    return {d*t00, -d*t01,  d*t02,
           -d*t10,  d*t11, -d*t12,
            d*t20, -d*t21,  d*t22};
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
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_INVERSE_H
