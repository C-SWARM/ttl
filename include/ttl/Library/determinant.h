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
#ifndef TTL_LIBRARY_DETERMINANT_H
#define TTL_LIBRARY_DETERMINANT_H

#include <ttl/config.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Library/binder.h>

namespace ttl {
namespace lib {
template <class E,
          int R = expressions::rank<E>::value,
          int D = expressions::dimension<E>::value>
struct det_impl;

/// Analytical determinant for 2x2
template <class E>
struct det_impl<E, 2, 2>
{
  CUDA_BOTH static constexpr auto op(E f) {
    return f(0,0)*f(1,1) - f(0,1)*f(1,0);
  }
};

template <class E>
struct det_impl<E, 2, 3>
{
  CUDA_BOTH static auto op(E f) {
    auto t0 = f(0,0)*f(1,1)*f(2,2);
    auto t1 = f(1,0)*f(2,1)*f(0,2);
    auto t2 = f(2,0)*f(0,1)*f(1,2);
    auto s0 = f(0,0)*f(1,2)*f(2,1);
    auto s1 = f(1,1)*f(2,0)*f(0,2);
    auto s2 = f(2,2)*f(0,1)*f(1,0);
    return (t0 + t1 + t2) - (s0 + s1 + s2);
  }
};
} // namespace lib

template <class E>
CUDA_BOTH constexpr expressions::scalar_type<E> det(E e) {
  return lib::det_impl<E>::op(e);
}

template <int D, class S>
CUDA_BOTH constexpr auto det(const Tensor<2,D,S>& matrix) {
  return det(lib::bind(matrix));
}
} // namespace ttl

#endif // #ifndef TTL_LIBRARY_DETERMINANT_H
