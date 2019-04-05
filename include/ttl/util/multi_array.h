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
#ifndef TTL_UTIL_MULTI_ARRAY_H
#define TTL_UTIL_MULTI_ARRAY_H

namespace ttl {
namespace util {
template <int R, int D, class Storage, class T>
struct multi_array {
  constexpr multi_array(size_t n, Storage data) noexcept
      : n_(n), data_(data)
  {
  }

  constexpr const multi_array<R - 1, D, Storage, T>
  operator[](size_t i) const noexcept
  {
    return { n_ + i * util::pow(D, R - 1), data_ };
  }

  constexpr multi_array<R - 1, D, Storage, T>
  operator[](size_t i) noexcept
  {
    return { n_ + i * util::pow(D, R - 1), data_ };
  }

 private:
  size_t n_;
  Storage data_;
};

template <int D, class Storage, class T>
struct multi_array<0, D, Storage, T> {
  constexpr multi_array(size_t n, Storage data) noexcept
      : data_(data.get(n))
  {
  }

  constexpr auto operator=(std::remove_reference_t<T> rhs) noexcept {
    return (data_ = rhs);
  }

  constexpr operator T() const noexcept {
    return data_;
  }

  constexpr const auto operator&() const noexcept {
    return &data_;
  }

  constexpr auto operator&() noexcept {
    return &data_;
  }

 private:
  T data_;
};

template <int R, int D, class Storage>
constexpr auto make_multi_array(Storage&& data) {
  using T = decltype(data.get(0));
  return multi_array<R, D, Storage, T>{ 0, std::forward<Storage>(data) };
}
} // namespace util
} // namespace ttl

#endif // #define TTL_UTIL_MULTI_ARRAY_H

