// -*- C++ -*-
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

#include <ttl/Pack.h>
#include <ttl/detail/and_type.h>
#include <ttl/detail/or_type.h>
#include <ttl/detail/xor_type.h>

namespace ttl {
template <char ID>
struct Index {
  static constexpr char id = ID;
};

template <class T, class U>
struct intersect {
  using type = typename detail::and_type<T, U>::type;
};

template <class T, class U>
struct unite {
  using type = typename detail::or_type<T, U>::type;
};

template <class T, class U>
struct symdif {
  using type = typename detail::xor_type<T, U>::type;
};
} // namespace ttl

#endif // TTL_INDEX_H
