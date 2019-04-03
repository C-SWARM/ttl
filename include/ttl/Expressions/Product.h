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
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Expressions/execution.hpp>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/promote.h>

namespace ttl {
namespace expressions {

/// Forward declare the tensor product for the traits implementation.
template <class L, class R>
class Product;

/// The expression Traits for Product.
///
/// The tensor product promotes the scalar type from its left and right hand
/// sides, and performs a set disjoint union to expose its free_type.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
struct traits<Product<L, R>>
{
 private:
  static_assert(is_expression_t<L>::value, "Operand is not Expression");
  static_assert(is_expression_t<R>::value, "Operand is not Expression");

  using l_outer_type = expressions::outer_type<L>;
  using r_outer_type = expressions::outer_type<R>;

  static constexpr int l_dim = expressions::dimension<L>();
  static constexpr int r_dim = expressions::dimension<R>();
  static_assert(l_dim == r_dim or
                l_dim == -1 or
                r_dim == -1,
                "Cannot combine expressions with different dimensionality");
  static constexpr int dim = ((l_dim < r_dim) ? r_dim : l_dim);

 public:
  using outer_type = set_xor<l_outer_type, r_outer_type>;
  using inner_type = set_and<l_outer_type, r_outer_type>;
  using scalar_type = promote<L, R>;
  using dimension = std::integral_constant<int, dim>;
  using rank = typename std::tuple_size<outer_type>::type;
};

/// The Product expression implementation.
///
/// A tensor product combines two expressions using multiplication, potentially
/// mixed with contraction (accumulation) over some shared dimensions of the two
/// expressions.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
class Product : public Expression<Product<L, R>>
{
 public:
  constexpr Product(L lhs, R rhs) noexcept : lhs_(lhs), rhs_(rhs) {
  }

  /// The eval() operation for the product forwards to the contraction routine.
  ///
  /// In TTL, contraction requires that we take an outer index---something like
  /// (i,j)---extend it with "slots" for inner hidden dimensions---like
  /// (i,j,K,L)---and then iterate over all of the values for the inner hidden
  /// dimensions---(i,j,0,0), (i,j,0,1), etc)---accumulating the results.
  ///
  /// @param          i The incoming index.
  /// @returns          The scalar contraction of the hidden dimensions in the
  ///                   expression.
  template <class I>
  constexpr auto eval(I i) const noexcept {
    return contract<Product>(i, [&](auto index) {
        return lhs_.eval(index) * rhs_.eval(index);
      });
  }

 private:
  L lhs_;                                    //!< The left-hand-side expression
  R rhs_;                                    //!< The right-hand-side expression
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
