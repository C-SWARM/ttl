// -------------------------------------------------------------------*- C++ -*-
// Copyright (c) 2017, Center for Shock Wave-processing of Advanced Reactive Materials (C-SWARM)
// University of Notre Dame
// Indiana University
// University of Washington
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_EXPRESSION_H
#define TTL_EXPRESSIONS_EXPRESSION_H

#include <ttl/Expressions/traits.h>
#include <ttl/mp/non_integer.hpp>
#include <ostream>
#include <tuple>

namespace ttl {
namespace expressions {

/// The expression that represents binding an index space to a subtree.
///
/// @tparam           E The subtree type.
/// @tparam       Index The index map for this expression.
template <class E, class Index>
class Bind;

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
  template <class Index>
  constexpr auto eval(Index index) const noexcept {
    return static_cast<const E*>(this)->eval(std::move(index));
  }

  template <class... Index>
  constexpr const auto to(std::tuple<Index...> index) const noexcept {
    return make_bind(std::move(index), std::move(*static_cast<const E*>(this)));
  }

  template <class... Index>
  constexpr const auto to(Index&&... index) const noexcept {
    return to(std::make_tuple(std::forward<Index>(index)...));
  }

  /// This operator is used to index an expression with integers.
  template <class... Index>
  constexpr const auto operator()(Index&&... index) const noexcept {
    using t = std::tuple<remove_cvref_t<Index>...>;
    constexpr auto N = std::tuple_size<mp::non_integer_t<t>>::value;
    static_assert(N == 0, "Tensor expression must be fully quantified by ()");
    return eval(outer_t<E>(std::forward<Index>(index)...));
  }

  constexpr operator scalar_t<E>() const noexcept {
    static_assert(rank<E>() == 0, "Tensor expression used in scalar context");
    return eval(std::tuple<>{});
  }

  constexpr std::ostream& print(std::ostream& os) const noexcept {
    return static_cast<const E*>(this)->print(os);
  }
};

template <class E>
struct traits<Expression<E>> : public traits<E> {
};

template <class E>
struct is_expression {
  using type = typename std::is_base_of<Expression<E>, E>::type;
  static constexpr bool value = std::is_base_of<Expression<E>, E>::value;
};

template <class E>
struct is_expression<Expression<E>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

template <class E>
using is_expression_t = typename is_expression<remove_cvref_t<E>>::type;

} // namespace expressions
} // namespace ttl

template <class E>
std::ostream& operator<<(std::ostream& os, const ttl::expressions::Expression<E>& e) {
  return e.print(os);
}

#endif // TTL_EXPRESSIONS_EXPRESSION_H
