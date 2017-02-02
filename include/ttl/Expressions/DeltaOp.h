// -*- C++ -*-------------------------------------------------------------------
/// This header defines and implements the delta expression.
///
/// The delta expression differs from the delta tensor in that it has no storage
/// allocated with it. It is merely an expression tree node that returns 1 when
/// all of the index values match and 0 otherwise.
///
/// A delta tensor can be allocated by assigning a delta expression to a
/// tensor.
///
/// @code
///   Tensor<3,4,double> D = delta<4>(i,j,k);
/// @code
///
/// The delta expression cannot infer the proper dimensionality of the
/// expression node, and it needs to be able to implement the dimension trait,
/// so the user must statically specify the dimensionality in the expression.
///
/// @todo We can implement the dimension trait with a universal dimension so
///       that assertions about the dimensionality will always match.
///
/// The scalar type of the delta expression is integer, but that type is
/// automatically promoted in all use cases.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_DELTA_H
#define TTL_EXPRESSIONS_DELTA_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/traits.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {
template <int n, int N>
struct is_diagonal
{
  template <class I, class Index>
  static constexpr bool op(I d, const Index index) noexcept {
    return (std::get<n>(index) == d && is_diagonal<n+1,N>::op(d, index));
  }
};

template <int N>
struct is_diagonal<N, N>
{
  template <class I, class Index>
  static constexpr bool op(I, const Index) noexcept {
    return true;
  }
};
} // namespace detail

template <int D, class Index>
class DeltaOp : public Expression<DeltaOp<D, Index>>
{
  static constexpr int N = std::tuple_size<Index>::value;

 public:
  constexpr int eval(const Index index) const noexcept {
    return detail::is_diagonal<0,N>::op(std::get<0>(index), index);
  }

  template <class I>
  constexpr int eval(const I index) const noexcept {
    return eval(ttl::expressions::transform<Index>(index));
  }
};

template <int D>
class DeltaOp<D, std::tuple<>> : public Expression<DeltaOp<D, std::tuple<>>>
{
 public:
  template <class I>
  constexpr int eval(I) const noexcept {
    return 1;
  }
};

template <int D, class Index>
struct traits<DeltaOp<D, Index>>
{
  using outer_type = Index;
  using scalar_type = int;
  using dimension = std::integral_constant<int, D>;
  using rank = typename std::tuple_size<Index>::type;
};
} // namespace expressions

template <int D = -1, class... I>
auto delta(I... i) {
  return expressions::DeltaOp<D, std::tuple<I...>>();
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_DELTA_H
