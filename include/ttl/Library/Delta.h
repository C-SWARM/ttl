// -*- C++ -*-
#ifndef TTL_LIBRARY_DELTA_H
#define TTL_LIBRARY_DELTA_H

#include <ttl/TensorImpl.h>
#include <ttl/detail/linearize.h>

namespace ttl {
namespace detail {

/// Hack to generate a diagonal index, which is just an array filled with the
/// same value. This isn't a huge problem because this is never a very deep
/// prospect in our domain (the manifold dimension is basically 3 or 4).
///
/// @todo Is there a std:: way to do this statically?
template <int N, int D, int R>
inline int diagonal() {
  IndexSet<R> i;
  i.fill(N);
  return linearize<D, R>(i);
}

/// Recursive template to fill a Delta matrix.
template <int N, class S, class T>
struct fill_delta_impl;

template <int N, int R, class S, int D>
struct fill_delta_impl<N, S, Tensor<R, S, D>> {
  using T = Tensor<R, S, D>;
  static T op(T&& rhs, S s) {
    rhs[diagonal<N,D,R>()] = s;
    return fill_delta_impl<N - 1, S, T>::op(std::forward<T>(rhs), s);
  }
};

template <int R, class S, int D>
struct fill_delta_impl<-1, S, Tensor<R, S, D>> {
  using T = Tensor<R, S, D>;
  static T op(T&& rhs, S s) {
    return rhs;
  }
};
} // namespace detail


/// Create a Delta tensor for a specific dimension and rank.
///
/// The Delta tensor has non-zero indices in its "diagonal". By default these
/// will be 1 (technically Scalar(1)) but the user can select whatever they want
/// dynamically.
template <int Rank, class Scalar, int Dimension,
          class = typename std::enable_if<(Rank > 1)>:: type>
inline constexpr auto Delta(Scalar s = Scalar(1))
  -> Tensor<Rank, Scalar, Dimension>            // @todo delete for C++14
{
  using T = Tensor<Rank, Scalar, Dimension>;
  return detail::fill_delta_impl<Dimension, Scalar, T>::op(T(), s);
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_DELTA_H
