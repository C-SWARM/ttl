// -*- C++ -*-
#ifndef TTL_UTIL_MAKE_INDEX_H
#define TTL_UTIL_MAKE_INDEX_H

namespace ttl {
namespace util {
/// A simple constructor for 2d indexing
static constexpr auto make_ij(int i, int j) {
  return std::make_tuple(Index<'\0'>(i), Index<'\1'>(j));
}
}
}

#endif // #ifndef TTL_UTIL_MAKE_INDEX_H
