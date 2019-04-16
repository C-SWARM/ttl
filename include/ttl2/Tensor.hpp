#pragma once

#include "ttl2/Index.hpp"
#include "ttl2/Bind.hpp"
#include "ttl2/mp.hpp"
#include <algorithm>
#include <array>
#include <cassert>

namespace ttl {
template <int Rank, int Dimension, class ScalarType>
class Tensor {
  static constexpr size_t pow(size_t r) {
    return (r) ? D * pow(r - 1) : 1;
  }

  static constexpr size_t size() {
    return pow(Rank);
  }

 public:
  static constexpr int R = Rank;
  static constexpr int D = Dimension;
  using T = ScalarType;

  static_assert(std::is_arithmetic<ScalarType>::value,
                "Tensors require fundamental scalar type");

  constexpr Tensor() noexcept = default;
  constexpr Tensor(const Tensor&) noexcept = default;
  constexpr Tensor(Tensor&&) noexcept = default;

  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Tensor(Rhs rhs) noexcept {
    bind<typename Rhs::Index>(*this) = std::move(rhs);
  }

  template <class... U,
            class = std::enable_if_t<all_convertible<T, U...>::value and
                                     size() == sizeof...(U)>>
  constexpr Tensor(U... args) noexcept : data{static_cast<T>(args)...} {
  }

  /// Assign to a tensor using an initializer list.
  constexpr Tensor& operator=(std::initializer_list<T> rhs) {
    std::copy_n(rhs.begin(), std::min(size(), rhs.size()), data.begin());
    return *this;
  }

  /// Copy or move a tensor.
  constexpr Tensor& operator=(Tensor rhs) & noexcept {
    std::swap(data, rhs.data);
    return *this;
  }

  /// Assignment from an expression on the right hand side.
  template <class Rhs,
            typename Rhs::is_expression** = nullptr>
  constexpr Tensor& operator=(Rhs rhs) & noexcept {
    bind<typename Rhs::Index>(*this) = std::move(rhs);
    return *this;
  }

  template <class... U>
  constexpr const T& operator()(std::tuple<U...> index) const& noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(apply(index, inRange<U...>));
    return data[apply(index, rowMajor<U...>)];
  }

  template <class... U>
  constexpr T& operator()(std::tuple<U...> index) && noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(apply(index, inRange<U...>));
    return data[apply(index, rowMajor<U...>)];
  }

  template <class... U>
  constexpr T& operator()(std::tuple<U...> index) & noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    assert(apply(index, inRange<U...>));
    return data[apply(index, rowMajor<U...>)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr const T& operator()(U... index) const& noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    assert(inRange(index...));
    return data[rowMajor(index...)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(U... index) && noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    assert(inRange(index...));
    return data[rowMajor(index...)];
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr T& operator()(U... index) & noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    assert(inRange(index...));
    return data[rowMajor(index...)];
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) const& noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return bind(std::make_tuple(index...), *this);
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) && noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return bind(std::make_tuple(index...), std::move(*this));
  }

  template <class... U,
            class = std::enable_if_t<!all_integer<U...>::value>>
  constexpr auto operator()(U... index) & noexcept {
    static_assert(sizeof...(U) == R, "Incorrect number of indices provided to Tensor");
    static_assert(all_valid<U...>::value, "Invalid type(s) in index");
    return bind(std::make_tuple(index...), *this);
  }

  constexpr auto operator[](size_t i) const & noexcept {
    return data[i];
  }

  constexpr auto& operator[](size_t i) && noexcept {
    return data[i];
  }

  constexpr auto& operator[](size_t i) & noexcept {
    return data[i];
  }

  void fill(T t) {
    std::fill_n(data.begin(), size(), t);
  }

 private:
  static constexpr bool inRange() {
    return true;
  }

  template <class T, class... U>
  static constexpr bool inRange(T car, U... cdr) {
    return 0 <= car and car < D and inRange(cdr...);
  }

  static constexpr size_t rowMajor() {
    return 0;
  }

  template <class T, class... U>
  static constexpr size_t rowMajor(T car, U... cdr) {
    return car * pow(sizeof...(U)) + rowMajor(cdr...);
  }

  std::array<T, size()> data = {};
};
}
