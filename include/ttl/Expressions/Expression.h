// -*- C++ -*-
#ifndef TTL_EXPRESSION_H
#define TTL_EXPRESSION_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/detail/check.h>
#include <array>
#include <type_traits>

namespace ttl {
namespace expressions {

/// Type traits for the Expression curiously-recurring-template-pattern (CRTP).
///
/// We use an expression tree framework based on the standard CRTP mechanism,
/// where we define the Expression "superclass" parametrically based on the
/// "subclass" type. This allows us to forward to the subclass's behavior
/// without any semantic overhead, and thus allows the compiler to see the "real
/// types" in the expression.
///
/// Inside the Expression template we need to know things about the subclass
/// type, but at the time of the declaration the subclass type is
/// incomplete, so we can't say things like `Subclass::Scalar`. The standard
/// solution to this is a traits class (i.e., a level of indirection).
///
/// Each expression type must also provide an associated traits specialization
/// to provide at least the required functionality. This is class concept-based
/// design... the Expression system will work for any class that defines the
/// right concepts.
///
/// The default traits class tries to generate useful errors for classes that
/// haven't defined traits.
template <class E>
struct Traits;

/// The base expression class template.
///
/// Every expression will subclass this template, with it's specific subclass as
/// the E type parameter. This curiously-recurring-template-pattern provides
/// static polymorphic behavior, allowing us to build expression trees.
///
/// @tparam           E The type of the base expression.
template <class E>
class Expression {
 public:
  using Traits = Traits<E>;

  constexpr auto operator[](IndexSet<Traits::Rank> i) const
    -> typename Traits::ScalarType
  {
    return static_cast<const E&>(*this)[i];
  }
};

/// This helper metaprogram makes sure that the index packs associated with two
/// expression types are compatible, i.e., they have the same underlying index
/// types, even if they're not in the same order.
///
/// The normal usage is in a template list, e.g.,
///
/// @code
///  template <class L, class R, class = check_compatible<L, R>> class ...
/// @code
///
/// @{
template <class L, class R>
struct check_compatible_impl {
 private:
  using L_ = typename std::remove_reference<L>::type;
  using R_ = typename std::remove_reference<R>::type;
  using LI_ = typename Traits<L_>::IndexType;
  using RI_ = typename Traits<R_>::IndexType;

 public:
  static constexpr bool value = is_equivalent<LI_, RI_>::value;
  using type = detail::check<value>;
};

template <class L, class R>
using check_compatible = typename check_compatible_impl<L, R>::type;
/// @}

/// Template for promoting scalar types.
///
/// We use multiplication as the default
template <class L, class R,
          bool = std::is_arithmetic<L>::value,
          bool = std::is_arithmetic<R>::value>
struct promote_impl;

template <class L, class R>
struct promote_impl<L, R, true, true> {
  using type = decltype(L() * R());             // both scalars
};

template <class L, class R>
struct promote_impl<L, R, true, false> {
 private:
  using R_ = typename Traits<R>::ScalarType;    // resolve right
 public:
  using type = typename promote_impl<L, R_>::type;
};

template <class L, class R>
struct promote_impl<L, R, false, true> {
 private:
  using L_ = typename Traits<L>::ScalarType;    // resolve left
 public:
  using type = typename promote_impl<L_, R>::type;
};

template <class L, class R>
struct promote_impl<L, R, false, false> {
 private:
  using L_ = typename Traits<L>::ScalarType;    // resolve both
  using R_ = typename Traits<L>::ScalarType;
 public:
  using type = typename promote_impl<L_, R_>::type;
};

template <class L, class R>
using promote = typename promote_impl<L, R>::type;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSION_H
