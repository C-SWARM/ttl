// -*- C++ -*-
#ifndef TTL_LIBRARY_INVERSE_H
#define TTL_LIBRARY_INVERSE_H

#include <ttl/Tensor.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/force.h>
#include <ttl/Library/determinant.h>

namespace ttl {
namespace detail {
constexpr int log2(int val) {
  return ((sizeof(val) * 8 - 1) - __builtin_clz(val)) + (!!(val & (val - 1)));
}

template <class E,
          int N = expressions::traits<expressions::rinse<E>>::rank::value,
          int D = expressions::traits<expressions::rinse<E>>::dimension::value>
using square_dimension = std::integral_constant<int, log2(N) * D>;

template <int N>
struct inverse_impl;

/// Analytically expand 3x3 inverse.
template <>
struct inverse_impl<3>
{
  using I0 = Index<'\0'>;
  using I1 = Index<'\1'>;
  using i = std::tuple<I0,I1>;

  template<class E>
  static expressions::tensor_type<E> op(E&& e) {
    auto f = expressions::force(std::forward<E>(e));
    auto t00 = f.eval(i{2,2})*f.eval(i{1,1}) - f.eval(i{2,1})*f.eval(i{1,2}); //a22a11-a21a12
    auto t01 = f.eval(i{2,2})*f.eval(i{0,1}) - f.eval(i{2,1})*f.eval(i{0,2}); //a22a01-a21a02
    auto t02 = f.eval(i{1,2})*f.eval(i{0,1}) - f.eval(i{1,1})*f.eval(i{0,2}); //a12a01-a11a02
    auto t10 = f.eval(i{2,2})*f.eval(i{1,0}) - f.eval(i{2,0})*f.eval(i{1,2}); //a22a10-a20a12
    auto t11 = f.eval(i{2,2})*f.eval(i{0,0}) - f.eval(i{2,0})*f.eval(i{0,2}); //a22a00-a20a02
    auto t12 = f.eval(i{1,2})*f.eval(i{0,0}) - f.eval(i{1,0})*f.eval(i{0,2}); //a12a00-a10a02
    auto t20 = f.eval(i{2,1})*f.eval(i{1,0}) - f.eval(i{2,0})*f.eval(i{1,1}); //a21a10-a20a11
    auto t21 = f.eval(i{2,1})*f.eval(i{0,0}) - f.eval(i{2,0})*f.eval(i{0,1}); //a21a00-a20a01
    auto t22 = f.eval(i{1,1})*f.eval(i{0,0}) - f.eval(i{1,0})*f.eval(i{0,1}); //a11a00-a10a01
    return 1/ttl::det(f) * expressions::tensor_type<E>{t00, -t01,  t02,
                                                      -t10,  t11, -t12,
                                                       t20, -t21,  t22}(I0(),I1());
  }
};

template <>
struct inverse_impl<9>
{
  template<class E>
  static expressions::tensor_type<E> op(E&& e) {
    expressions::tensor_type<E> temp{};
    return temp;
  }
};
}

template <class E>
constexpr expressions::tensor_type<E> inverse(E&& e) {
  return detail::inverse_impl<detail::square_dimension<E>::value>::op(std::forward<E>(e));
}
}

#endif // #ifndef TTL_LIBRARY_INVERSE_H
