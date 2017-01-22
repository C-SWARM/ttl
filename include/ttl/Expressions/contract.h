// -*- C++ -*-------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_CONTRACT_H
#define TTL_EXPRESSIONS_CONTRACT_H

#include <ttl/Expressions/traits.h>

namespace ttl {
namespace expressions {
namespace detail {
/// The recursive contraction template.
///
/// This template is instantiated to generate a loop for each inner dimension
/// for the expression. Each loop accumulates the result of the nested loops'
/// outputs.
template <class E,
          int n = std::tuple_size<outer_type<E>>::value,
          int M = std::tuple_size<concat<outer_type<E>, inner_type<E>>>::value,
          int D = dimension<E>::value>
struct contract_impl
{
  template <class Index, class F>              // c++14 auto (icc 16 complains)
  static auto op(Index index, F&& f) noexcept {
    decltype(f(index)) s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      s += contract_impl<E, n+1>::op(index, std::forward<F>(f));
    }
    return s;
  }
};

/// The contraction base case evaluates the lambda on the current index.
template <class E, int M, int D>
struct contract_impl<E, M, M, D>
{
  template <class Index, class F>              // c++14 auto (icc 16 complains)
  static constexpr auto op(Index index, F&& f) noexcept {
    return f(index);
  }
};

/// Simple local utility to take an external index, select the subset of indices
/// that appear in the Expression's outer type, and extend it with indices for
/// the Expression's inner type.
template <class E,
          class Index>           // Index c++14 auto (icc 16 complains)
constexpr auto extend(Index i) {
  return std::tuple_cat(transform<outer_type<E>>(i), inner_type<E>{});
}
} // namespace detail

/// The external entry point for contraction takes the external index set and
/// the lambda to apply in the inner loop, and instantiates the recursive
/// template to expand the inner loops.
///
/// @tparam           E The type of the expression being contracted.
///
/// @param            i The partial index generated externally.
/// @param            f The lambda expression to evaluate in the inner loop.
///
/// @returns            The fully contracted scalar value, i.e., the sum of the
///                     inner loop invocations.
template <class E,
          class Index, class F>              // c++14 auto (icc 16 complains)
constexpr auto contract(Index i, F&& f) noexcept {
  return detail::contract_impl<E>::op(detail::extend<E>(i), std::forward<F>(f));
}
} // namespace contract
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_CONTRACT_H
