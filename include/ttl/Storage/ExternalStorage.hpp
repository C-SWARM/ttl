#pragma once

#include <iterator>

namespace ttl {
template <class Scalar, size_t N>
class ExternalStorage {
 public:
  using T = Scalar;

  constexpr ExternalStorage(T (*data)[N]) noexcept : data_(*data) {
  }

  // constexpr ExternalStorage(const ExternalStorage&) = default;
  // constexpr ExternalStorage(ExternalStorage&&) = default;

  // constexpr ExternalStorage& operator=(const ExternalStorage&) = default;
  // constexpr ExternalStorage& operator=(ExternalStorage&&) = default;

  constexpr const T& operator()(size_t i) const noexcept {
    return data_[i];
  }

  constexpr T& operator()(size_t i) noexcept {
    return data_[i];
  }

  constexpr auto begin() const noexcept {
    return std::begin(data_);
  }

  constexpr auto begin() noexcept {
    return std::begin(data_);
  }

  constexpr auto end() const noexcept {
    return std::end(data_);
  }

  constexpr auto end() noexcept {
    return std::end(data_);
  }

 private:
  T (&data_)[N];
};
} // namespace ttl
