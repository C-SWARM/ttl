// -*- C++ -*-
#ifndef TTL_UTIL_LOG2_H
#define TTL_UTIL_LOG2_H

namespace ttl {
namespace util {
/// Simple compile time template for computing the log2 of a power of 2.
///
/// This uses template specialization to ensure that we don't try and take the
/// log of a k where k is not a power of two.
///
/// @tparam         N
/// @tparam         B
template <int N, int B = N%2>
struct log2;

template <int N>
struct log2<N, 0> {
  static constexpr int value = 1 + log2<N/2>::value;
};

template <>
struct log2<1, 1> {
  static constexpr int value = 0;
};
} // namespace util
} // namespace tll

#endif // TTL_UTIL_LOG2_H
