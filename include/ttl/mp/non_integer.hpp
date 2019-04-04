#pragma once

#include <ttl/mp/cat.hpp>
#include <ttl/mp/iif.hpp>
#include <limits>
#include <tuple>

namespace ttl {
namespace mp {
template <class T>
struct non_integer {
  using type = std::tuple<>;
};

template <class T0, class... T>
struct non_integer<std::tuple<T0, T...>> {
  using next = typename non_integer<std::tuple<T...>>::type;
  using type = iif_t<std::numeric_limits<T0>::is_integer, next, cat_t<T0, next>>;
};

template <class T>
using non_integer_t = typename non_integer<T>::type;
} // namespace mp
} // namespace ttl
