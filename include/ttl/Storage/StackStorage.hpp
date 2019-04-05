#pragma once

#include <array>

namespace ttl {
template <class Scalar, size_t N>
class StackStorage {
 public:
  using T = Scalar;

  constexpr const T& operator()(size_t i) const noexcept {
    return data_[i];
  }

  constexpr T& operator()(size_t i) noexcept {
    return data_[i];
  }

  constexpr auto begin() const noexcept {
    return data_.begin();
  }

  constexpr auto begin() noexcept {
    return data_.begin();
  }

  constexpr auto end() const noexcept {
    return data_.end();
  }

  constexpr auto end() noexcept {
    return data_.end();
  }

 private:
  std::array<T, N> data_;
};
} // namespace ttl
