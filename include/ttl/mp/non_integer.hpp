#pragma once

#include <ttl/mp/cat.hpp>
#include <ttl/mp/iif.hpp>
#include <limits>
#include <tuple>

namespace ttl {
namespace mp {
template <class T>
struct is_integer {
  enum : bool { value = std::numeric_limits<std::decay_t<T>>::is_integer };
};

template <class T>
struct non_integer {
  using type = std::tuple<>;
};

template <class T0, class... T>
struct non_integer<std::tuple<T0, T...>> {
  using next = typename non_integer<std::tuple<T...>>::type;
  using type = iif_t<is_integer<T0>::value, next, cat_t<T0, next>>;
};

template <class T>
using non_integer_t = typename non_integer<T>::type;

template <class... T>
struct is_all_integer;

template <>
struct is_all_integer<> {
  enum : bool { value = true };
};

template <class T0, class... T>
struct is_all_integer<T0, T...> {
  enum : bool {
    value = is_integer<T0>::value and is_all_integer<T...>::value
  };
};

template <class... T>
struct is_all_integer<std::tuple<T...>> : public is_all_integer<T...> {
};
} // namespace mp
} // namespace ttl
