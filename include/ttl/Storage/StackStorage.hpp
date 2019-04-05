#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>

namespace ttl {
template <class T, size_t N>
class StackStorage {
 public:
  /// Stack storage constructors... all of the default stuff is fine.
  constexpr StackStorage() noexcept = default;
  constexpr StackStorage(const StackStorage&) noexcept = default;
  constexpr StackStorage(StackStorage&&) noexcept = default;
  constexpr StackStorage& operator=(const StackStorage&) noexcept = default;
  constexpr StackStorage& operator=(StackStorage&&) noexcept = default;

  /// Stack storage can be initialized or assigned from initializer lists.
  template <class S>
  constexpr StackStorage(std::initializer_list<S> list) noexcept
      : data_(copy(list))
  {
  }

  template <class S>
  constexpr StackStorage& operator=(std::initializer_list<S> list) noexcept {
    data_ = copy(list);
    return *this;
  }

  /// Get a linear iterator to the storage array.
  constexpr auto begin() const noexcept {
    return data_.begin();
  }

  /// Get a linear iterator to the storage array.
  constexpr auto begin() noexcept {
    return data_.begin();
  }

  /// Get a linear iterator to the storage array.
  constexpr auto end() const noexcept {
    return data_.end();
  }

  /// Get a linear iterator to the storage array.
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
  template <class S>
  static constexpr std::array<T, N> copy(std::initializer_list<S> list) noexcept {
    std::array<T, N> i;
    auto min = std::min(N, list.size());
    auto p = std::copy_n(list.begin(), min, i.begin());
    std::fill_n(p, N - min, 0);
    return i;
  }

  std::array<T, N> data_;
};
} // namespace ttl
