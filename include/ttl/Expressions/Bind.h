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
/// This header includes the functionality for binding and evaluating tensor
/// expressions.
///
/// Its primary template is the Bind expression which is generated using
/// a Tensor's operator(), e.g., A(i,j). It also defines the Expression traits
/// for the Bind expression, as well as some ::detail metafunctions to
/// process type sets.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_TENSOR_BIND_H
#define TTL_EXPRESSIONS_TENSOR_BIND_H

#include <ttl/Index.h>
#include <ttl/Tensor.h>
#include <ttl/Expressions/execution.hpp>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>
#include <ttl/mp/duplicate.hpp>
#include <ttl/mp/non_integer.hpp>
#include <ttl/mp/subset.hpp>
#include <ttl/mp/unique.hpp>
#include <utility>

namespace ttl {
namespace expressions {
/// The expression that represents binding an index space to a subtree.
///
/// @tparam           E The subtree type.
/// @tparam       Index The index map for this expression.
template <class E, class Index>
class Bind;

/// The expression traits for Bind expressions.
///
/// @tparam           E The subtree expression type.
/// @tparam       Index The indices bound to this expression.
template <class E, class Index>
struct traits<Bind<E, Index>> : public traits<std::decay_t<E>>
{
 private:
  using index_t = mp::non_integer_t<Index>;

 public:
  using outer_type = mp::unique_t<index_t>;
  using inner_type = mp::duplicate_t<index_t>;
  using rank_type = typename std::tuple_size<outer_type>::type;
};

/// The Bind expression.
///
/// The bind expression is the expression that adapts a Tensor for use in a
/// Tensor Expression. It merges the underlying Tensor storage with an index
/// type, allowing the core syntax that differentiates a ttl Tensor from a basic
/// multidimensional array (or higher order matrix).
///
/// @tparam       Child The type of the child node in the syntax tree.
/// @tparam       Index The index type.
template <class Child, class Index>
class Bind : public Expression<Bind<Child, Index>>
{
 public:
  static constexpr int Rank = rank_t<Bind>::value;
  static constexpr int N = dimension_t<Bind>::value;

  /// A Bind expression keeps a reference to the E it wraps, and an index.
  ///
  /// @tparam         t The underlying tensor.
  constexpr Bind(Child t, Index i = Index{}) noexcept : t_(t), i_(i) {
  }

  template <class Outer>
  constexpr auto eval(Outer index) const {
    return contract<Bind>(index, [this](auto index) {
        return t_.eval(transform(index));
    });
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam         E Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  std::enable_if_t<is_expression<std::decay_t<E>>::value, Bind&>
  operator=(E&& rhs)
  {
    static_assert(dimension_t<E>::value ==  N or
                  dimension_t<E>::value == -1,
                  "Cannot operate on expressions of differing dimension");
    static_assert(mp::equivalent_t<outer_t<Bind>, outer_t<E>>::value,
                  "Attempted assignment of incompatible Expressions");
    forall<Bind>([&](auto i) {
        t_.eval(transform(i)) = std::forward<E>(rhs).eval(i);
    });
    return *this;
  }

  /// Assignment of a scalar to a fully specified scalar right hand side.
  template <class T>
  std::enable_if_t<std::is_arithmetic<std::decay_t<T>>::value, Bind&>
  operator=(T rhs)
  {
    static_assert(Rank == 0, "Cannot assign scalar to tensor");
    forall<Bind>([this,r=std::move(rhs)](auto i) {
      t_.eval(transform(std::move(i))) = r;
    });
    return *this;
  }

  /// Accumulate from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam       RHS Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class RHS>
  Bind& operator+=(RHS&& rhs) {
    static_assert(dimension_t<RHS>::value == N,
                  "Cannot operate on expressions of differing dimension");
    static_assert(mp::equivalent_t<outer_t<Bind>, outer_t<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    forall<Bind>([&](auto i) {
      t_.eval(transform(std::move(i))) += rhs.eval(i);
    });
    return *this;
  }

  /// Basic print-to-stream functionality.
  ///
  /// This iterates through the index space and prints the index and results to
  /// the stream.
  std::ostream& print(std::ostream& os) const {
    forall<Bind>([&](auto i) {
      print_pack(os, i) << ": " << eval(i) << "\n";
    });
    return os;
  }

 private:
  template <class Other>
  Index transform(Other index) const {
    return expressions::transform(i_, std::move(index));
  }

  Child t_;                                     ///<! The underlying tree.
  const Index i_;                               ///<! The bound index.
};

template <class Index, class Child>
constexpr Bind<Child, Index> make_bind(Index index, Child&& child) {
  return { std::forward<Child>(child), std::move(index) };
}

template <class Index, class Child>
constexpr Bind<Child, Index> make_bind(Child&& child) {
  return { std::forward<Child>(child) };
}
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
