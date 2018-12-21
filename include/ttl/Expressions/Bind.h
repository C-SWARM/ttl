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
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/contract.h>
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
  using Outer = outer_type<Bind>;

 public:
  /// A Bind expression keeps a reference to the E it wraps, and a
  ///
  /// @tparam         t The underlying tensor.
  constexpr Bind(Child t, const Index i = Index{}) noexcept : t_(t), i_(i) {
  }

  /// The index operator maps the index array using the normal interpretation of
  /// multidimensional indexing, using index 0 as the most-significant-index.
  ///
  /// @code
  ///        Rank = n = sizeof(index)
  ///   Dimension = k
  ///       index = {a, b, c, ..., z}
  ///           i = a*k^(n-1) + b*k^(n-2) + c*k^(n-3) + ... + z*k^0
  /// @code
  ///
  /// @param      index The index array, which must be the same length as the
  ///                   Rank of this Bind.
  ///
  /// @returns          The scalar value at the linearized offset.
  template <class I>
  CUDA constexpr auto eval(I i) const {
    return contract<Bind>(i, [this](auto index) {
        // intel 16.0 can't handle the "transform" symbol here without the
        // namespace
        return t_.eval(ttl::expressions::transform(i_, index));
      });
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam       RHS Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class RHS>
  CUDA Bind& operator=(RHS&& rhs) {
    static_assert(dimension<RHS>::value == dimension<Bind>::value or
                  dimension<RHS>::value == -1,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    apply<>::op(Outer{}, [&rhs,this](Outer i) {
        t_.eval(ttl::expressions::transform(i_, i)) = std::forward<RHS>(rhs).eval(i);
      });
    return *this;
  }

  /// Assignment of a scalar to a fully specified scalar right hand side.
  CUDA Bind& operator=(scalar_type<Bind> rhs) {
    static_assert(rank<Bind>::value == 0, "Cannot assign scalar to tensor");
    apply<>::op(Outer{}, [rhs,this](Outer i) {
        t_.eval(ttl::expressions::transform(i_, i)) = rhs;
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
  CUDA Bind& operator+=(RHS&& rhs) {
    static_assert(dimension<E>::value == dimension<RHS>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    apply<>::op(Outer{}, [&rhs,this](const Outer i) {
        t_.eval(ttl::expressions::transform(i_, i)) += std::forward<RHS>(rhs).eval(i);
      });
    return *this;
  }

  /// Basic print-to-stream functionality.
  ///
  /// This iterates through the index space and prints the index and results to
  /// the stream.
  CUDA_HOST std::ostream& print(std::ostream& os) const {
    apply<>::op(Outer{}, [&os,this](const Outer i) {
        print_pack(os, i) << ": " << eval(i) << "\n";
      });
    return os;
  }

 private:
  /// The recursive template class that evaluates tensor expressions.
  ///
  /// The fundamental goal of ttl is to generate loops over tensor dimension,
  /// evaluating the right-hand-side of the expression for each of the
  /// combinations of inputs on the left hand side.
  ///
  /// @code
  ///   for i : 0..D-1
  ///    for j : 0..D-1
  ///      ...
  ///        for n : 0..D-1
  ///           lhs(i,j,...,n) = rhs(i,j,...,n)
  /// @code
  ///
  /// We use recursive template expansion to generate these "loops" statically, by
  /// dynamically enumerating the index dimensions. There is presumably a static
  /// enumeration technique, as our bounds are all known statically.
  ///
  /// @note I tried to use a normal recursive function but there were typing
  ///       errors for the case where M=0 (i.e., the degenerate scalar tensor
  ///       type).
  ///
  /// @tparam           n The current dimension that we need to traverse.
  /// @tparam           M The total number of free indices to enumerate.
  template <int n = 0, int M = std::tuple_size<Outer>::value>
  struct apply
  {
    static constexpr int D = dimension<E>::value;
    static_assert(D > 0, "Apply requires explicit dimensionality");

    /// The evaluation routine just iterates through the values of the nth
    /// dimension of the tensor, recursively calling the template.
    ///
    /// @tparam      Op The lambda to evaluate for each index.
    /// @param    index The partially constructed index.
    template <class Op>
    static constexpr void op(Outer index, Op&& f) {
      for (int i = 0; i < D; ++i) {
        std::get<n>(index) = i;
        apply<n + 1>::op(index, std::forward<Op>(f));
      }
    }
  };

  /// The base case for tensor evaluation.
  ///
  /// @tparam         M The total number of dimensions to enumerate.
  template <int M>
  struct apply<M, M>
  {
    template <class Op>
    static constexpr void op(Outer index, Op&& f) {
      f(index);
    }

    /// Specialize for the case where the index space is empty (i.e., the
    /// left-hand-side is a scalar as in an inner product).
    ///
    /// @code
    ///   c() = a(i) * b(i)
    /// @code
    template <class Op>
    static constexpr void op(Op&& f) {
      f(Outer{});
    }
  };

  Child t_;                                     ///<! The underlying tree.
  const Index i_;                               ///<! The bound index.
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
