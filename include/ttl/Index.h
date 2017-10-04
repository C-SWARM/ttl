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
#ifndef TTL_INDEX_H
#define TTL_INDEX_H

namespace ttl {

/// This is an index template that can be used to index a tensor.
///
/// Each class template is parameterized by a character. Index values that have
/// the same class are assumed to be the same index.
///
/// @code
///   Index<'a'> i;
///   Index<'a'> j;
///   Tensor<2,2,int> M,N;
///   M(i,j) == M(i,i);
/// @code
///
/// Internally, ttl uses indexes as loop control variables to iterate through
/// contractions and assignment operations.
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
