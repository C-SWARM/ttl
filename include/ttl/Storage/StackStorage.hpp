#pragma once

#include <array>

namespace ttl {
template <class Scalar, size_t N>
class StackStorage {
 public:
  using T = Scalar;

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

  /// Direct linear indexing into the storage.
  ///
  /// @code
  ///   Tensor<R,D,int> A;
  ///   int i = A.get(0);
  /// @code
  ///
  /// @param          i The index to access.
  /// @returns          The scalar value at @p i.
  constexpr const T& get(size_t i) const noexcept {
    return data_[i];
  }

  /// Direct linear indexing into the storage.
  ///
  /// @code
  ///   Tensor<R,D,int> A;
  ///   int i = A.get(0);
  /// @code
  ///
  /// @param          i The index to access.
  /// @returns          The scalar value at @p i.
  constexpr T& get(size_t i) noexcept {
    return data_[i];
  }

 private:
  std::array<T, N> data_;
};
} // namespace ttl
