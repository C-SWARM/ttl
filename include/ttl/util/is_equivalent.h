// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_IS_EQUIVALENT_H
#define TTL_EXPRESSIONS_IS_EQUIVALENT_H

#include <ttl/util/is_subset.h>
#include <type_traits>

namespace ttl {
namespace util {
template <class T, class U>
using is_equivalent = typename util::iif<is_subset<T,U>, is_subset<U,T>,
                                         std::false_type>::type;
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_IS_EQUIVALENT_H
