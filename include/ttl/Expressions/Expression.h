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
  using Scalar = typename Traits<E>::Scalar;
  static constexpr int Rank = Traits<E>::Rank;

  constexpr Scalar operator[](IndexSet<Rank> i) const {
    return static_cast<const E&>(*this)[i];
  }
};

/// Check to see if two expressions have compatible index sets.
template <class L, class R>
struct is_compatible {
  using LSet_ = typename Traits<L>::IndexSet;
  using RSet_ = typename Traits<R>::IndexSet;
  static constexpr bool value = is_equivalent<LSet_, RSet_>::value;
};

template <class L, class R>
struct check_compatible {
  using value = typename detail::check<is_compatible<L, R>::value>::value;
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSION_H
