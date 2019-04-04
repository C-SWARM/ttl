#pragma once

#include <tuple>

namespace ttl {
namespace mp {
template <class T, class U>
struct index_of {
  static constexpr size_t value = 0;           // matches when U == std::tuple<>
};

template <class T, class... U>
struct index_of<T, std::tuple<T, U...>> {
  static constexpr size_t value = 0;
};

template <class T, class U0, class... U>
struct index_of<T, std::tuple<U0, U...>> {
  using next = index_of<T, std::tuple<U...>>;
  static constexpr size_t value = 1 + next::value;
};
} // namespace mp
} // namespace ttl
