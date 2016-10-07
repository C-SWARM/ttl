// -*- C++ -*-
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

#include <ttl/Pack.h>
#include <ttl/detail/and_type.h>
#include <ttl/detail/or_type.h>
#include <ttl/detail/xor_type.h>
#include <array>

namespace ttl {
template <char ID>
struct Index {
  static constexpr char id = ID;
};

/// Right now we're encoding runtime indices through integer arrays.
///
/// In the future it might make a lot of sense to extend the Index template with
/// an integer field so that we can use IndexPacks with std::get and std::set
/// overrides at runtime. This would make our Index structures uniform at
/// compile and runtime.
template <int N>
using IndexSet = std::array<int, N>;

template <class T, class U>
using intersect = typename detail::and_type<T, U>::type;

template <class T, class U>
using unite = typename detail::or_type<T, U>::type;

template <class T, class U>
using symdif = typename detail::xor_type<T, U>::type;

} // namespace ttl

#endif // TTL_INDEX_H
