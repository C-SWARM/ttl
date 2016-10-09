// -*- C++ -*-
#ifndef TTL_LIBRARY_DELTA_H
#define TTL_LIBRARY_DELTA_H

#include <ttl/Tensor.h>
#include <ttl/TensorImpl.h>
#include <ttl/detail/pow.h>

namespace ttl {
namespace detail {
/// Compute the linearized index of the Nth diagonal index in a tensor with
/// dimension D.
///
/// @code
///   (2, 2, 2) = 2*D^2, 2*D^1, 2*D^0
/// @code
///
/// @tparam           D The dimensionality of the manifold.
/// @tparam           R The number of terms we're adding.
///
/// @param            n The Nth diagonal we're trying to compute.
/// @param            i The current term we're adding.
///
/// @returns            The linear index of the Nth diagonal element.
template <int D, int R>
constexpr int diagonal(const int n, const int i = 0) {
  return (i < R) ? n * pow(D, i) + diagonal<D, R>(n, i + 1) : 0;
}

template <class T, class S,
          int D = tensor_traits<T>::dimension,
          int R = tensor_traits<T>::rank>
T make_delta(T&& t, const S s, const int n = 0) {
  if (n < D) {
    t[diagonal<D, R>(n)] = s;
    return make_delta(std::forward<T>(t), s, n + 1);
  }
  else {
    return t;
  }
}
} // namespace detail


/// Create a Delta tensor for a specific dimension and rank.
///
/// The Delta tensor has non-zero indices in its "diagonal". By default these
/// will be 1 (technically Scalar(1)) but the user can select whatever they want
/// dynamically.
template <int R, class S, int D>
constexpr Tensor<R, S, D> Delta(S s = S(1)) {
  return detail::make_delta(Tensor<R, S, D>(S(0)), s);
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_DELTA_H
