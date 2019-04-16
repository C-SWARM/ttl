#pragma once

#include <ttl/mp/cat.hpp>
#include <ttl/mp/index_of.hpp>
#include <tuple>
#include <utility>

namespace ttl {
namespace mp {
template <class T, class U>
struct subset_sequence {
  using type = std::index_sequence<>;
};

template <class T0, class... T, class U>
struct subset_sequence<std::tuple<T0, T...>, U> {
  using tail = typename subset_sequence<std::tuple<T...>, U>::type;
  using type = sequence_cat_t<index_of<T0, U>::value, tail>;
};

template <class T, class U>
using subset_sequence_t = typename subset_sequence<T, U>::type;
} // namespace mp
} // namespace ttl
