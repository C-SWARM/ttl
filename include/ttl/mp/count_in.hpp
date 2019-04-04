#pragma once

#include <ttl/mp/cat.hpp>
#include <tuple>

namespace ttl {
namespace mp {
template <class T, class U>
struct count_in;

template <class T>
struct count_in<T, std::tuple<>> {
  using type = std::tuple<>;
  static constexpr int value = 0;
};

template <class T, class... U>
struct count_in<T, std::tuple<T, U...>> {
  using tail = count_in<T, std::tuple<U...>>;
  using type = typename tail::type;
  static constexpr int value = 1 + tail::value;
};

template <class T, class U0, class... U>
struct count_in<T, std::tuple<U0, U...>> {
  using tail = count_in<T, std::tuple<U...>>;
  using type = cat_t<U0, typename tail::type>;
  static constexpr int value = tail::value;
};
} // namespace mp
} // namespace ttl
