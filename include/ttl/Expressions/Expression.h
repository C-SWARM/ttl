// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_EXPRESSION_H
#define TTL_EXPRESSIONS_EXPRESSION_H

#include <ttl/Tensor.h>
#include <tuple>
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
struct expression_traits;

/// This specialization is used to try and print a hopefully useful error when
/// Tensors are used without indices.
template <int R, int D, class T>
struct expression_traits<Tensor<R, D, T>>
{
  using scalar_type = typename std::remove_pointer<T>::type;
  using dimension = std::integral_constant<int, D>;
};

/// The following traits are required for all expression types.
template <class E>
using free_type = typename expression_traits<E>::free_type;

template <class E>
using scalar_type = typename expression_traits<E>::scalar_type;

template <class E>
using dimension = typename expression_traits<E>::dimension;

/// Derived traits that are widely used.
template <class E>
using free_size = typename std::tuple_size<free_type<E>>::type;

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
  template <class I>
  constexpr typename expression_traits<E>::scalar_type operator[](I i) const {
    return static_cast<const E&>(*this)[i];
  }
};

template <class E>
struct is_expression_impl {
  using type = std::is_base_of<Expression<E>, E>;
};

template <class E>
using is_expression = typename is_expression_impl<E>::type;

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_EXPRESSION_H
