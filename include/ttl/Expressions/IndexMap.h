// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_INDEX_MAP_H
#define TTL_EXPRESSIONS_INDEX_MAP_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>

namespace ttl {
namespace expressions {
/// The expression that maps from one index space to another.
template <class E, class OuterType, class InnerType>
class IndexMap;

/// The IndexMap just masks the child's outer_type trait with the OuterType.
template <class E, class OuterType, class InnerType>
struct traits<IndexMap<E, OuterType, InnerType>> : public traits<E>
{
  using outer_type = OuterType;
};

/// The IndexMap implementation.
///
/// The basic purpose of this expression is to shuffle the outer_type of its child
/// expression into a compatible outer_type for the parent expression.
///
/// @tparam           T The type of the child expression.
/// @tparam   OuterType The type of the index space to export.
/// @tparam   InnerType The type of the child index space.
template <class E, class OuterType, class InnerType = outer_type<E>>
class IndexMap : public Expression<IndexMap<E, OuterType, InnerType>> {
 public:
  /// The index map is simply initialized with its child expression.
  constexpr IndexMap(E e) noexcept : e_(e) {
  }

  /// The index map eval() operation remaps the incoming index.
  ///
  /// The eval() operation takes an index, defined in the outer type, and maps
  /// it into the inner type. Interestingly, the incoming index does not
  /// necessarily match the OuterType for this class. The OuterType is used
  /// during the static upward traversal for type checking, while the eval()
  /// operation is used dynamically during evaluation.
  ///
  /// A simple case where the Index type is not equivalent to the OuterType is
  /// a matrix multiplication with a transpose.
  ///
  /// @code
  ///   C(i,k) = A(i,j)*B(k,j).to(j,k); // C = A*B^T
  /// @code
  ///
  /// In this context, the tensor product is creating indexes in the (i,k,j)
  /// space, and the IndexMap on the right hand side is only exporting the (j,k)
  /// space. Basically, B is not indexed by (i) so it is ignored. The index
  /// space transformation below will filter out those unused dimensions before
  /// forwarding them along to the child.
  ///
  /// @tparam     Index The type of the incoming index.
  /// @param          i The instance of the incoming instance.
  /// @returns          The scalar computed by the child expression for this
  ///                   index.
  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    static_assert(subset<OuterType, Index>::value, "Unexpected outer type during index mapping");
    return e_.eval(transform<InnerType>(index));
  }

 private:
  E e_;
};
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_INDEX_MAP_H
