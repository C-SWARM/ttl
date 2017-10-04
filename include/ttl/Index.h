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
/// @file  ttl/Index.h
/// @brief Contains the definition of the Index<ID> template.
// -----------------------------------------------------------------------------
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

namespace ttl {

/// This is an index template that can be used to index a tensor.
///
/// The index class template simply creates unique types for each char ID that
/// it is parameterized with. Index values that have the same class are assumed
/// to be the same index. The resulting type can be manipulated by the TTL
/// internal infrastructure at compile time. This allows TTL to perform various
/// set operations on indices in order to perform index matching and code
/// generation for expressions.
///
/// Source-level indices only occur in constant, compile-time contexts and thus
/// it is common to see them declared as constexpr, const, or both.
///
/// @code
///   Tensor<2,2,int> A, B = {...}, C = {...};
///
///   constexpr const Index<'i'> i;
///   constexpr const Index<'j'> j;
///   constexpr const Index<'k'> k;
///
///   // matrix-matrix multiply in TTL (contracts `j`)
///   C(i,k) = A(i,j) * B(j,k);
///
///   // a weirder operation in TTL (contracts `j` again, but different "slot"
///   C(i,k) = A(j,i) * B(k,j);
/// @code
template <char ID>
struct Index {
  static constexpr char id = ID;

  constexpr Index() : value_(0) {
  }

  constexpr explicit Index(int value) : value_(value) {
  }

  Index(const Index&) = default;
  Index(Index&&) = default;
  Index& operator=(const Index&) = default;
  Index& operator=(Index&&) = default;

  Index& operator=(int i) {
    value_ = i;
    return *this;
  }

  Index& set(int i) {
    value_ = i;
    return *this;
  }

  constexpr operator int() const {
    return value_;
  }

  constexpr bool operator<(int e) const {
    return value_ < e;
  }

  Index& operator++() {
    ++value_;
    return *this;
  }

 private:
  int value_;
};
} // namespace ttl

#endif // TTL_INDEX_H
