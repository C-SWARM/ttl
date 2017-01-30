// -*- C++ -*-------------------------------------------------------------------
/// This header defines and implements the identity expression.
///
/// The identity expression for an even rank N is just the product of N/2 delta
/// expressions. The indices for the delta expressions are just the indices of
/// the identity split in half and interleaved.
///
/// ttl::identity() = 1
/// ttl::identity(i,j) = delta(i,j)
/// ttl::identity(i,j,k,l) = delta(i,k) * delta(j,l)
/// ttl::identity(i,j,k,l,m,n) = delta(i,l) * delta(j,m) * delta(k,n)
///
/// Our strategy for building an identity is to shuffle the indices into the
/// right order and then recursively spawning the delta expressions by popping
/// two indices at a time.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_IDENTITY_H
#define TTL_EXPRESSIONS_IDENTITY_H

#include <ttl/Expressions/DeltaOp.h>
#include <ttl/Expressions/pack.h>
#include <type_traits>

namespace ttl {
namespace expressions {
namespace detail {

template <class Pack, int n = 0, int N = std::tuple_size<Pack>::value>
struct split
{
  using next = split<cdr<Pack>, n+2, N>;
  using lhs = concat<car<Pack>, typename next::lhs>;
  using rhs = typename next::rhs;
};

template <int N, template <class...> class Pack, class... LHS>
struct split<Pack<LHS...>, N, N>
{
  using lhs = Pack<>;
  using rhs = Pack<LHS...>;
};

template <class LHS, class RHS>
struct merge
{
  using next = merge<cdr<LHS>,cdr<RHS>>;
  using first = concat<car<LHS>,car<RHS>>;
  using type = concat<first, typename next::type>;
};

template <template <class...> class Pack>
struct merge<Pack<>, Pack<>>
{
  using type = Pack<>;
};

template <class Pack>
struct shuffle
{
  using lhs = typename split<Pack>::lhs;
  using rhs = typename split<Pack>::rhs;
  using type = typename merge<lhs,rhs>::type;
};

template <class... I>
using shuffle_t = typename shuffle<std::tuple<I...>>::type;

} // namespace detail

template <int D>
constexpr int identity(std::tuple<>) {
  return 1;
}

template <int D, class T0, class T1, class... T>
constexpr auto identity(std::tuple<T0, T1, T...>) {
  using delta = DeltaOp<D,std::tuple<T0,T1>>;
  return delta() * identity<D>(std::tuple<T...>{});
}
} // namespace expressions

template <int D = -1, class... I>
constexpr auto identity(I...) {
  static_assert(sizeof...(I) % 2 == 0, "The identity must have even rank.");
  using type = expressions::detail::shuffle_t<I...>;
  return expressions::identity<D>(type{});
}
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_IDENTITY_H
