// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_FORCE_H
#define TTL_EXPRESSIONS_FORCE_H

#include <ttl/Expressions/traits.h>

namespace ttl {
namespace expressions {
template <class E>
auto force(E&& e) {
  return tensor_type<E>(std::forward<E>(e));
}
}
}

#endif // #ifndef TTL_EXPRESSIONS_FORCE_H
