// -*- C++ -*-
#ifndef TTL_DETAIL_LINEARIZE_H
#define TTL_DETAIL_LINEARIZE_H

#include <ttl/Index.h>
#include <ttl/detail/indexof.h>
#include <ttl/detail/pow.h>
#include <array>

namespace ttl {
namespace detail {

/// Linearize an index.
///
/// @code
///       R = n = sizeof(index)
///       D = k
///   index = {a, b, c, ..., z}
///     lhs = a*k^(n-1) + b*k^(n-2) + c*k^(n-3) + ... + z*k^0
/// @code
///
/// When we are traversing in parallel, then we are basically remapping the
/// index into the alternative space defined by the U parameter pack. A concrete
/// example follows.
///
/// @code
///       R = 3
///       D = 3
///       T = {i, j, k}
///       U = {j, i, k}
///   index = {1, 2, 3}
///     lhs = 1*3^2 + 2*3^1 + 3*3^0
///     rhs = 2*3^2 + 1*3^0 + 3*3^0
///
/// @code
///
/// @tparam           D The Index we're processing.
/// @tparam           D The dimensionality of the data.
/// @tparam           R The rank of the data.
template <int N, int D, int R>
struct linearize_impl {
  using next = linearize_impl<N + 1, D, R>;
  static constexpr int op(IndexSet<R> i, int accum) {
    return next::op(i, accum + i[N] * pow(D, R - N - 1));
  }
};

template <int D, int R>
struct linearize_impl<R, D, R> {
  static constexpr int op(IndexSet<R>, int accum) {
    return accum;
  }
};

template <int D, int R>
inline constexpr int linearize(IndexSet<R> in) {
  return linearize_impl<0, D, R>::op(in, 0);
}
} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_LINEARIZE_H
