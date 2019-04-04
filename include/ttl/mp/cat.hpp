#pragma once

#include <tuple>
#include <utility>

namespace ttl {
namespace mp {
template <class T, class U>
struct cat {
  using type = std::tuple<>;
};

template <class... T, class... U>
struct cat<std::tuple<T...>, std::tuple<U...>> {
  using type = std::tuple<T..., U...>;
};

template <class T, class... U>
struct cat<T, std::tuple<U...>> {
  using type = std::tuple<T, U...>;
};

template <class... T, class U>
struct cat<std::tuple<T...>, U> {
  using type = std::tuple<T..., U>;
};

template <>
struct cat<std::index_sequence<>, std::index_sequence<>> {
  using type = std::index_sequence<>;
};

template <size_t T, size_t... U>
struct cat<std::index_sequence<T>, std::index_sequence<U...>> {
  using type = std::index_sequence<T, U...>;
};

template <class T, class U>
using cat_t = typename cat<T, U>::type;

template <size_t T, class U>
using sequence_cat_t = typename cat<std::index_sequence<T>, U>::type;
} // namespace mp
} // namespace ttl
