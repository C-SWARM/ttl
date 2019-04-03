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
/// Bind currently strips the cvref keywords from the bound type.
///
/// @tparam           E The subtree expression type.
/// @tparam       Index The indices bound to this expression.
template <class E, class Index>
struct traits<Bind<E, Index>> : public traits<rinse<E>>
{
  using outer_type = unique<non_integral<Index>>;
  using inner_type = duplicate<non_integral<Index>>;
  using rank = typename std::tuple_size<outer_type>::type;
};

template <class E, class Index>
class Bind : public Expression<Bind<E, Index>>
{
  /// The Bind storage type is an expression, or a reference to a Tensor.
  using Child = iif<is_expression_t<E>, E, E&>;

 public:
  /// A Bind expression keeps a reference to the E it wraps, and a
  ///
  /// @tparam         t The underlying tensor.
  constexpr Bind(Child t, const Index i = Index{}) noexcept : t_(t), i_(i) {
  }

  static constexpr int Rank = rank_t<Bind>::value;
  static constexpr int N = dimension_t<Bind>::value;

  template <class OuterIndex>
  constexpr auto eval(OuterIndex index) const {
    return contract<Bind>(index, [this](auto index) {
        return t_.eval(transform(index));
    });
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam       RHS Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class RHS>
  Bind& operator=(RHS&& rhs) {
    static_assert(dimension_t<RHS>::value ==  N or
                  dimension_t<RHS>::value == -1,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<outer_type<Bind>, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    forall<Bind>([&](auto i) {
      t_.eval(transform(i)) = rhs.eval(i);
    });
    return *this;
  }

  /// Assignment of a scalar to a fully specified scalar right hand side.
  Bind& operator=(scalar_type<Bind> rhs) {
    static_assert(Rank == 0, "Cannot assign scalar to tensor");
    forall<Bind>([&](auto i) {
      t_.eval(transform(i)) = rhs;
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
    static_assert(equivalent<outer_type<Bind>, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    forall<Bind>([&](auto i) {
      t_.eval(transform(i)) += rhs.eval(i);
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
    return expressions::transform(i_, index);
  }

  Child t_;                                     ///<! The underlying tree.
  const Index i_;                               ///<! The bound index.
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
