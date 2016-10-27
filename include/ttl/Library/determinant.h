// -*- C++ -*-
#ifndef TTL_LIBRARY_DETERMINANT_H
#define TTL_LIBRARY_DETERMINANT_H

#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/force.h>

namespace ttl {
namespace detail {
template <int R, int D, class S>
struct det;

template <class S>
struct det<2,2,S>
{
  using i = std::tuple<Index<'\0'>,Index<'\1'>>;
  template <class E>
  static S op(E&& e) {
    auto f = expressions::force(std::forward<E>(e));
    auto t = f.eval(i{0,0})*f.eval(i{1,1});
    auto s = f.eval(i{0,1})*f.eval(i{1,0});
    return t - s;
  }
};

template <class S>
struct det<2, 3, S>
{
  using i = std::tuple<Index<'\0'>,Index<'\1'>>;
  template <class E>
  static S op(E&& e) {
    auto f = expressions::force(std::forward<E>(e));
    auto t0 = f.eval(i{0,0})*f.eval(i{1,1})*f.eval(i{2,2});
    auto t1 = f.eval(i{1,0})*f.eval(i{2,1})*f.eval(i{0,2});
    auto t2 = f.eval(i{2,0})*f.eval(i{0,1})*f.eval(i{1,2});
    auto s0 = f.eval(i{0,0})*f.eval(i{1,2})*f.eval(i{2,1});
    auto s1 = f.eval(i{1,1})*f.eval(i{2,0})*f.eval(i{0,2});
    auto s2 = f.eval(i{2,2})*f.eval(i{0,1})*f.eval(i{1,0});
    return (t0 + t1 + t2) - (s0 + s1 + s2);
  }
};
} // namespace detail

template <class E>
constexpr expressions::scalar_type<E> det(E&& e) {
  return detail::det<expressions::rank<E>::value,
                     expressions::dimension<E>::value,
                     expressions::scalar_type<E>>::op(std::forward<E>(e));
}
}

#endif // #ifndef TTL_LIBRARY_DETERMINANT_H
