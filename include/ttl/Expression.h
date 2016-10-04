// -*- C++ -*-
#ifndef TTL_EXPRESSION_H
#define TTL_EXPRESSION_H

namespace ttl {
template <typename T>
class Expression {
};

namespace expressions {
template <typename T>
class Bind : Expression<Bind<T>> {
  T& data_;
 public:
  Bind(T& data) : data_(data) {
  }
};

template <typename LHS, typename RHS>
class Add : Expression<Add<LHS, RHS>> {
  const Bind<LHS>& lhs_;
  const Bind<RHS>& rhs_;
 public:
  Add(const Bind<LHS>& lhs, const Bind<LHS>& rhs)
      : lhs_(lhs), rhs_(rhs) {
  }
};
}
}

#endif // TTL_EXPRESSION_H
