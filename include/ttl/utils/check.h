// -*- C++ -*-
#ifndef TTL_DETAIL_CHECK_LENGTH_H
#define TTL_DETAIL_CHECK_LENGTH_H

#include <type_traits>

namespace ttl {
namespace detail {
template <bool condition>
using check = typename std::enable_if<condition>::type;
} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_CHECK_LENGTH_H
