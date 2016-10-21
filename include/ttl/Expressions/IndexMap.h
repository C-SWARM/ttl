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
template <class T, class OuterType, class InnerType>
class IndexMap;

template <class T, class OuterType, class InnerType>
struct traits<IndexMap<T, OuterType, InnerType>> : traits<T>
{
  using free_type = OuterType;
};

template <class T, class OuterType, class InnerType>
class IndexMap : public Expression<IndexMap<T, OuterType, InnerType>> {
 public:
  explicit IndexMap(const T& t) : t_(t) {
  }

  template <class I>
  constexpr const scalar_type<IndexMap> get(I i) const {
    static_assert(util::is_equivalent<OuterType, I>::value,
                  "Unexpected outer type during index mapping");
    return t_.get(transform<InnerType>(i));
  }

 private:
  const T& t_;
};
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_INDEX_MAP_H
