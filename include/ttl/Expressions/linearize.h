// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_LINEARIZE_H
#define TTL_EXPRESSIONS_LINEARIZE_H

#include <ttl/util/pow.h>
#include <tuple>

namespace ttl {
namespace expressions {
namespace detail {

/// Linearization of an index means processing each term in the index.
///
/// @tparam           D The dimensionality of the data.
/// @tparam           T The type of the index (a tuple of indices).
/// @tparam           i The term we're processing.
/// @tparam           N The number of terms in the index---we need this to
///                     terminate template recursion.
template <class D, class T, int i = 0, int N = std::tuple_size<T>::value>
struct linearize_impl
{
  static constexpr int op(T index) {
    return head(index) + tail(index);
  }

 private:
  static constexpr int head(T index) {
    return int(std::get<i>(index)) * util::pow(D::value, N - i - 1);
  }

  static constexpr int tail(T index) {
    return linearize_impl<D, T, i + 1, N>::op(index);
  }
};

/// Recursive base case for linearization is when we've processed all of the
/// terms in the index.
template <class D, class T, int N>
struct linearize_impl<D, T, N, N> {
  static constexpr int op(T) {
    return 0;
  }
};
} // namespace detail

template <class D, class T>
constexpr int linearize(T index) {
  return detail::linearize_impl<D, T>::op(index);
}

} // namespace expression
} // namespace ttl


#endif // #ifndef TTL_EXPRESSIONS_LINEARIZE_H
