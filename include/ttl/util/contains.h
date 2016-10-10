// -*- C++ -*-
#ifndef TTL_UTIL_CONTAINS_H
#define TTL_UTIL_CONTAINS_H

#include <type_traits>

namespace ttl {
namespace util {

template <class T, class... U>
struct contains;

template <class T>
struct contains<T> : std::false_type {
};

template <class T, class... U>
struct contains<T, T, U...> : std::true_type {
};

template <class T, class U, class... V>
struct contains<T, U, V...> : contains<T, V...> {
};

} // namespace util
} // namespace ttl

#endif // #ifndef TTL_UTIL_CONTAINS_H
