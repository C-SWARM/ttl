// -*- C++ -*-------------------------------------------------------------------
/// This header defines and implements the identity expression.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_IDENTITY_H
#define TTL_EXPRESSIONS_IDENTITY_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {
} // namespace detail

template <int D, class Index>
class Identity : public Expression<Identity<D,Index>>
{
 public:
  static constexpr auto eval(Index i) {
    return 1;
  }
};

template <int D, class Index>
struct traits<Identity<D, Index>>
{
  using outer_type = Index;
  using scalar_type = int;
  using dimension = std::integral_constant<int, D>;
  using rank = typename std::tuple_size<Index>::type;
};
} // namespace expressions

template <int D = -1, class... I>
auto identity(I... i) {
  static_assert(sizeof...(I) % 2 == 0, "The identity must have even rank.");
  return expressions::Identity<D,std::tuple<I...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_IDENTITY_H
