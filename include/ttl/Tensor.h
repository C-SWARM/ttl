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
/// @file  ttl/Tensor.h
/// @brief Provides a forward declaration of the Tensor<> template.
///
/// See ttl/TensorImpl.h for the actual Tensor implementation.
// -----------------------------------------------------------------------------
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

namespace ttl {
/// The core class template for all Tensors.
///
/// @tparam        Rank The rank of the Tensor (the number of indices needed to
///                     fully specify a scalar value in the Tensor).
/// @tparam   Dimension The dimension of the underlying manifold (the valid
///                     values for indices in the Tensor [0,Dimension)).
/// @tparam  ScalarType The underlying type of the scalar values (int, double,
///                     complex, struct, etc).
///
/// The Tensor generally defines a square multidimensional, row-major array
/// structure of statically known sizes. From a storage perspective a basic
/// Tensor and a statically-sized multidimensional array are effectively the
/// same thing, as seen in the following example.
///
/// @code
///   Tensor<4,3,double> A;
///   A[1][2][0][1] = 1;
///   double a[3][3][3][3];
///   a[1][2][0][1] = 1;
/// @code
///
/// Tensors can be initialized elementwise, using initializer lists, and through
/// the use of the expected copy/move operators.
///
/// @code
///   Tensor<2,3,int> A = { 1, 2, 3,
///                         4, 5, 6,
///                         7, 8, 9 };
///
///   Tensor<2,3,double> Uninitialized;
///   for (int i = 0; i < 3; ++i) {
///     for (int j = 0; j < 3; ++j) {
///       Uninitialized[i][j] = 1.0;
///     }
///   }
///
///   Tensor<4,4,double> Zero = {};
/// @code
///
/// The major difference between a basic Tensor and a statically sized, square
/// array, is that a Tensor can be "bound" using its family of operator()(...)
/// implementations. A bound Tensor has access to the entire suite of Tensor
/// expressions defined in TTL. A Tensor can be bound either using integers in
/// the range [0,Dimension) or using Index<> types (@see ttl/Index.h) or
/// both. Binding a Tensor using one or more integers expresses a projection
/// operation, which can be assigned to or from a lower Rank tensor of the
/// appropriate shape. Using a projection in a compound Tensor exprssion does
/// not automatically create a copy of the underlying data, it merely restrict
/// enumeration of the index space when evaluating the expression. Some examples
/// of binding operations can be seen below.
///
/// @code
///   Tensor<4,2,double> A{};
///
///   // Fully projected
///   A(1,2,0,1) = 1.0;        // assigns single element
///   auto d = A(1,2,0,1);     // reads single element
///
///   // Partial projection
///   constexpr const Index<'i'> i;                /// @see ttl/Index.h
///   constexpr const Index<'j'> j;
///   Tensor <2,2,double> P = A(i,2,1,j);          // copies data
///   Tensor <2,2,double> Q = P(i,j) + A(1,i,j,0); // restricts enumeration
/// @code
///
/// Raw Tensors are not valid types for use in most expressions (there are some
/// Library operations that can operate on certain tensor types directly, like
/// transposes). Binding a Tensor provides access to the entire suite of
/// Expressions.
///
/// @code
///   constexpr const Index<'i'> i;        // @see ttk/Index.h
///   constexpr const Index<'j'> j;
///   constexpr const Index<'k'> k;
///   constexpr const Index<'l'> l;
///
///   Tensor<2,2,double> A{}, B{};
///   Tensor<4,2,double> C;
///
///   C(i,j,k,l) = 1.9 * A(i,j) * B(k,l);  // outer product with scalar multiply
/// @code
///
/// Tensor expressions with a tensor on the LHS force evaluation of the RHS. It
/// can be convenient to express complicated expressions as multipart
/// terms. C++11 type inference allows programmers to capture the intermediate
/// expressions without evaluating them, and they can then be combined (or
/// reused) in other Expressions.
///
/// @code
///   constexpr const Index<'i'> i;        // @see ttk/Index.h
///   constexpr const Index<'j'> j;
///   constexpr const Index<'k'> k;
///
///   Tensor<2,3,double> A{}, B{};
///   Tensor<2,3,double> C;
///
///   // The following two terms capture the matrix-matrix product expressions
///   // without evaluating anything. An optimizing compiler will produce no
///   // actual code for these two lines
///   auto AxB = A(i,j) * B(j,k);
///   auto BxA = B(i,j) * A(j,k);
///
///   // Use the terms in a compound expression. This works because the
///   // type identities of the 'i' and 'j' indices maintain their nature as
///   // part of the captured expressions.
///   C(i,k) = 0.5 * (AxB + BxA);
///
///   // For the CPU-based implementation, an optimizing compiler will produce a
///   // single triply-nested loop to evaluate this expression.
///   for (int i = 0; i < 3; ++i) {
///     for (int k = 0; k < 3; k++) {
///       C(i,k) = 0.0;
///       for (int j = 0; i < 3; ++j) {
///         C(i,k) += 0.5 * (A(i,j) * B(j,k) + B(i,j) * A(j,k));
///       }
///     }
///   }
/// @code
///
/// If the scalar type of a Tensor is a pointer type, then the Tensor must be
/// initialized with a reference to an external buffer of the underlying
/// type. In this case, Tensor operations will effect that external
/// storage. This form of externally allocated Tensor storage can be useful when
/// interacting with legacy code, or when custom allocation is required (e.g.,
/// CUDA-allocated memory, or pinned network memory).
///
/// @code
///   double external[3][3][3][3];
///   Tensor<4,3,double*> A{external};
///   A(1,2,0,1) = 1.0;
///   assert(external[1][2][0][1] == 1.0);
/// @code
///
/// Tensors with external storage can be used in any context in which a Tensor
/// with internal storage can be used.
template <int Rank, int Dimension, class ScalarType>
class Tensor;
} // namespace ttl

#endif // #ifndef TTL_TENSOR_H
