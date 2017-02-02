// -*- C++ -*-------------------------------------------------------------------
/// This header defines and implements the zero expression.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_ZERO_H
#define TTL_EXPRESSIONS_ZERO_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <tuple>

namespace ttl {
namespace expressions {
template <int D, class Index>
class Zero : public Expression<Zero<D,Index>>
{
 public:
  template <class I>
  constexpr int eval(I) const noexcept {
    return 0;
  }
};

template <int D, class Index>
struct traits<Zero<D, Index>>
{
  using outer_type = Index;
  using scalar_type = int;
  using dimension = std::integral_constant<int, D>;
  using rank = typename std::tuple_size<Index>::type;
};
} // namespace expressions

template <int D = -1, class... I>
auto zero(I... i) {
  return expressions::Zero<D, std::tuple<I...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_ZERO_H
