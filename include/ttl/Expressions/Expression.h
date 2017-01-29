// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_EXPRESSION_H
#define TTL_EXPRESSIONS_EXPRESSION_H

#include <ttl/Expressions/force.h>
#include <ttl/Expressions/traits.h>
#include <ostream>
#include <type_traits>

namespace ttl {
namespace expressions {

/// Forward declare the IndexMap template, since it is needed in the Expression
// template <class T, class Outer, class Inner> class IndexMap;
template <class T, class Index> class Bind;

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
  constexpr auto eval(I index) const {
    return static_cast<const E*>(this)->eval(index);
  }

  template <template <class...> class Pack, class... I>
  constexpr const auto to(Pack<I...> index) const {
    return make_bind(*this, index);
  }

  template <class... I>
  constexpr const auto to(I... index) const {
    return to(std::make_tuple(index...));
  }

  constexpr operator const scalar_type<E>() const {
    return eval(std::tuple<>{});
  }

  constexpr std::ostream& print(std::ostream& os) const {
    return static_cast<const E*>(this)->print(os);
  }
};

template <class E>
struct traits<Expression<E>> : public traits<E> {
};

namespace detail {
template <class E>
struct is_expression_impl {
  using type = std::is_base_of<Expression<E>, E>;
};
} // namespace detail

template <class E>
using is_expression = typename detail::is_expression_impl<E>::type;

} // namespace expressions
} // namespace ttl

template <class E,
          class = std::enable_if_t<ttl::expressions::is_expression<E>::value>>
std::ostream& operator<<(std::ostream& os, const E& e) {
  using Index = ttl::expressions::outer_type<E>;
  using Bind = ttl::expressions::Bind<const E,Index>;
  return Bind(e).print(os);
}

#endif // TTL_EXPRESSIONS_EXPRESSION_H
