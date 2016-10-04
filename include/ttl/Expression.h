// -*- C++ -*-
#ifndef TTL_EXPRESSION_H
#define TTL_EXPRESSION_H

#include <ttl/Index.h>
#include <type_traits>
#include <utility>

namespace ttl {
template <class T, class External, class Internal>
class Expression {
  static_assert(is_empty<intersect<External, Internal>>::value,
                "Expressions should have disjoint external and internal index "
                "sets.");
 public:
  template <class Index>
  auto eval(Index&& index) const {
    return e_.eval(std::forward<Index>(index));
  }

  Expression(T& e) : e_(e) {
  }

 private:
  T& e_;
};
} // namespace ttl

#endif // TTL_EXPRESSION_H
