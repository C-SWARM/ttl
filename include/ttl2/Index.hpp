#pragma once

#include <type_traits>

namespace ttl {
template <char Id>
struct Index {
  constexpr Index& operator=(Index rhs) && noexcept {
    std::swap(i, rhs.i);
    return *this;
  };

  constexpr Index& operator=(Index rhs) & noexcept {
    std::swap(i, rhs.i);
    return *this;
  };

  constexpr Index& operator=(int rhs) && noexcept {
    i = rhs;
    return *this;
  };

  constexpr Index& operator=(int rhs) & noexcept {
    i = rhs;
    return *this;
  };

  constexpr operator int() const noexcept {
    return i;
  }

  int i = {};
};

template <class T>
struct is_index {
  using type = std::true_type;
  enum : bool { value = false };
};

template <char Id>
struct is_index<Index<Id>> {
  using type = std::true_type;
  enum : bool { value = true };
};
}
