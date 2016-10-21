// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_INDEX_MAP_H
#define TTL_EXPRESSIONS_INDEX_MAP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>
#include <ttl/util/is_equivalent.h>

namespace ttl {
namespace expressions {
/// The expression that maps from one index space to another.
///
/// The basic purpose of this expression is to shuffle the free_type of its child
/// expression into a compatible free_type for the parent expression.
template <class E, class OuterType>
class IndexMap;

template <class E, class OuterType>
struct traits<IndexMap<E, OuterType>> : traits<E>
{
  using free_type = OuterType;
};

template <class E, class OuterType>
class IndexMap : public Expression<IndexMap<E, OuterType>> {
  static_assert(util::is_equivalent<free_type<E>, OuterType>::value,
                "Mapped index types are not equivalent.");
 public:
  explicit IndexMap(const E& e) : e_(e) {
  }

  template <class I>
  constexpr const scalar_type<IndexMap> get(I i) const {
    static_assert(util::is_equivalent<OuterType, I>::value,
                  "Unexpected outer type during index mapping");
    return e_.get(transform<free_type<IndexMap>>(i));
  }

 private:
  const E& e_;
};
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_INDEX_MAP_H
