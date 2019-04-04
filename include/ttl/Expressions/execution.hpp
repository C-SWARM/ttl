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
#ifndef TTL_EXPRESSIONS_EXECUTION_HPP
#define TTL_EXPRESSIONS_EXECUTION_HPP

#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>

namespace ttl {
namespace expressions {
template <int n>
using int_constant = std::integral_constant<int, n>;

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
/// @tparam           n The current dimension that we need to traverse.
/// @tparam           M The total number of free indices to enumerate.
/// @tparam           D The dimensionality of the space.
template <class n, class M, int D>
struct forall_impl;

template <int n, int M, int D>
struct forall_impl<int_constant<n>, int_constant<M>, D>
{
  /// The evaluation routine just iterates through the values of the nth
  /// dimension of the tensor, recursively calling the template.
  ///
  /// @tparam   Index The index we're generating.
  /// @tparam      Op The lambda to evaluate for each index.
  ///
  /// @param    index The partially constructed index.
  /// @param       op The operator we're going to evaluate for each index.
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    using next = forall_impl<int_constant<n + 1>, int_constant<M>, D>;
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      next::op(index, std::forward<Op>(op));
    }
  }
};

/// The base case for tensor evaluation.
///
/// @tparam         M The total number of dimensions to enumerate.
template <int M, int D>
struct forall_impl<int_constant<M>, int_constant<M>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    op(index);
  }

  /// Specialize for the case where the index space is empty (i.e., the
  /// left-hand-side is a scalar as in an inner product).
  ///
  /// @code
  ///   c() = a(i) * b(i)
  /// @code
  template <class Op>
  static void op(Op&& op) {
    op();
  }
};

template <int n, int D>
struct forall_impl<int_constant<n>, int_constant<n + 2>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n+1>(index).set(j);
        op(index);
      }
    }
  }
};

template <int n, int D>
struct forall_impl<int_constant<n>, int_constant<n + 3>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          op(index);
        }
      }
    }
  }
};

template <int n, int D>
struct forall_impl<int_constant<n>, int_constant<n + 4>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            op(index);
          }
        }
      }
    }
  }
};

template <int n, int D>
struct forall_impl<int_constant<n>, int_constant<n + 5>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            for (int m = 0; m < D; ++m) {
              std::get<n + 4>(index).set(m);
              op(index);
            }
          }
        }
      }
    }
  }
};

template <int n, int D>
struct forall_impl<int_constant<n>, int_constant<n + 6>, D>
{
  template <class Index, class Op>
  static void op(Index index, Op&& op) {
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            for (int m = 0; m < D; ++m) {
              std::get<n + 4>(index).set(m);
              for (int o = 0; o < D; ++o) {
                std::get<n + 5>(index).set(o);
                op(index);
              }
            }
          }
        }
      }
    }
  }
};

/// The recursive contraction template.
///
/// This template is instantiated to generate a loop for each inner dimension
/// for the expression. Each loop accumulates the result of the nested loops'
/// outputs.
///
/// @tparam           n The starting index.
/// @tparam           M The final index.
/// @tparam           D The dimensionality.
/// @tparam           T The contracted type.
template <class n, class M, int D, class T>
struct contract_impl;

template <int n, int M, int D, class T>
struct contract_impl<int_constant<n>, int_constant<M>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    using next = contract_impl<int_constant<n + 1>, int_constant<M>, D, T>;
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      s += next::op(index, std::forward<Op>(op));
    }
    return s;
  }
};

/// The contraction base case evaluates the lambda on the current index.
template <int M, int D, class T>
struct contract_impl<int_constant<M>, int_constant<M>, D, T>
{
  template <class Index, class Op>
  static constexpr T op(Index index, Op&& op) noexcept {
    return op(index);
  }
};

template <int n, int D, class T>
struct contract_impl<int_constant<n>, int_constant<n + 2>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        s += op(index);
      }
    }
    return s;
  }
};

template <int n, int D, class T>
struct contract_impl<int_constant<n>, int_constant<n + 3>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          s += op(index);
        }
      }
    }
    return s;
  }
};

template <int n, int D, class T>
struct contract_impl<int_constant<n>, int_constant<n + 4>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            s += op(index);
          }
        }
      }
    }
    return s;
  }
};

template <int n, int D, class T>
struct contract_impl<int_constant<n>, int_constant<n + 5>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            for (int m = 0; m < D; ++m) {
              std::get<n + 4>(index).set(m);
              s += op(index);
            }
          }
        }
      }
    }
    return s;
  }
};

template <int n, int D, class T>
struct contract_impl<int_constant<n>, int_constant<n + 6>, D, T>
{
  template <class Index, class Op>
  static T op(Index index, Op&& op) noexcept {
    T s{};
    for (int i = 0; i < D; ++i) {
      std::get<n>(index).set(i);
      for (int j = 0; j < D; ++j) {
        std::get<n + 1>(index).set(j);
        for (int k = 0; k < D; ++k) {
          std::get<n + 2>(index).set(k);
          for (int l = 0; l < D; ++l) {
            std::get<n + 3>(index).set(l);
            for (int m = 0; m < D; ++m) {
              std::get<n + 4>(index).set(m);
              for (int o = 0; o < D; ++o) {
                std::get<n + 5>(index).set(o);
                s += op(index);
              }
            }
          }
        }
      }
    }
    return s;
  }
};

template <class E, class Index, class Op>
constexpr auto contractExtended(Index i, Op&& op) noexcept {
  using std::tuple_size;
  constexpr int n = tuple_size<outer_type<E>>::value;
  constexpr int M = tuple_size<Index>::value;
  constexpr int D = dimension<E>();
  using T = scalar_type<E>;
  using impl = contract_impl<int_constant<n>, int_constant<M>, D, T>;
  return impl::op(i, std::forward<Op>(op));
}

/// Simple local utility to take an external index, select the subset of indices
/// that appear in the Expression's outer type, and extend it with indices for
/// the Expression's inner type.
template <class E, class Index>
constexpr auto extend(Index i) {
  using std::tuple_cat;
  using Outer = outer_type<E>;
  using Inner = inner_type<E>;
  return tuple_cat(transform(Outer{}, i), Inner{});
}

/// The external entry point for contraction takes the external index set and
/// the lambda to apply in the inner loop, and instantiates the recursive
/// template to expand the inner loops.
///
/// @tparam           E The type of the expression being contracted.
/// @tparam       Index The type of the index generated externally.
/// @tparam          Op The type of the operation to evaluate in the inner loop.
///
/// @param            i The partial index generated externally.
/// @param           op The lambda expression to evaluate in the inner loop.
///
/// @returns            The fully contracted scalar value, i.e., the sum of the
///                     inner loop invocations.
template <class E, class Index, class Op>
constexpr auto contract(Index i, Op&& op) noexcept {
  return contractExtended<E>(extend<E>(i), std::forward<Op>(op));
}

/// The external entry point for evaluating an expression.
///
/// This simply evaluates the passed operation for all indices in the outer type
/// of the expression.
///
/// @tparam           E The type of the expression being contracted.
/// @tparam          Op The type of the operation to evaluate in the inner loop.
///
/// @param            i The partial index generated externally.
/// @param           op The lambda expression to evaluate in the inner loop.
template <class E, class Op>
constexpr void forall(Op&& op) noexcept {
  using std::tuple_size;
  using Index = outer_type<E>;
  constexpr int n = 0;
  constexpr int M = tuple_size<Index>::value;
  constexpr int D = dimension<E>();
  using impl = forall_impl<int_constant<n>, int_constant<M>, D>;
  impl::op(Index{}, std::forward<Op>(op));
}
} // namespace expressions
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_EXECUTION_HPP
