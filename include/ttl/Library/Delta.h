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
#ifndef TTL_LIBRARY_DELTA_H
#define TTL_LIBRARY_DELTA_H

#pragma message("ttl::Delta<>() is deprecated, please move to ttl::delta(i,j)")

#include <ttl/Tensor.h>
#include <ttl/TensorImpl.h>
#include <ttl/util/pow.h>

namespace ttl {
namespace detail {
/// Compute the linearized index of the Nth diagonal index in a tensor with
/// dimension D.
///
/// @code
///   (2, 2, 2) = 2*D^2, 2*D^1, 2*D^0
/// @code
///
/// @tparam           D The dimensionality of the tensor.
/// @tparam           R The number of terms we're adding.
///
/// @param            n The Nth diagonal we're trying to compute.
/// @param            i The current term we're adding.
///
/// @returns            The linear index of the Nth diagonal element.
template <int D, int R>
constexpr int diagonal(const int n, const int i = 0) {
  return (i < R) ? n * ttl::util::pow(D, i) + diagonal<D, R>(n, i + 1) : 0;
}

/// Convenience constructor for recursively initializing a Delta tensor.
///
/// @tparam           R The rank of the tensor.
/// @tparam           D The dimensionality of the tensor.
/// @tparam           T The scalar type of the tensor.
/// @tparam           U The type of the scalar.
///
/// @param       tensor The tensor we're initializing.
/// @param            t The initial diagonal value.
/// @param            n The dimension we're currently initializing.
///
/// @returns            The t*Delta() for the R, D Tensor shape.
template <int R, int D, class T, class U>
Tensor<R, D, T> make_delta(Tensor<R, D, T>&& tensor, U scalar, int n = 0) {
  if (n == D) {
    return tensor;                              /// recursive base case
  }
  tensor.get(diagonal<D, R>(n)) = scalar;
  return make_delta(std::forward<Tensor<R, D, T>>(tensor), scalar, n + 1);
}
} // namespace detail


/// Create a Delta tensor for a specific dimension and rank.
///
/// The Delta tensor has non-zero indices in its "diagonal". By default these
/// will be 1 (technically Scalar(1)) but the user can select whatever they want
/// dynamically.
template <int R, int D, class T, class U = T>
constexpr Tensor<R, D, T> Delta(U scalar = U{1}) {
  return detail::make_delta(Tensor<R, D, T>{}, scalar);
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_DELTA_H
