#pragma once

#include <ttl/mp/count_in.hpp>
#include <ttl/mp/iif.hpp>
#include <tuple>
#include <type_traits>

namespace ttl {
namespace mp {
template <class T, class U>
struct subset {
  using type = std::true_type;
  static constexpr bool value = true;
  static constexpr bool equiv = true;
};

template <class T0, class... T>
struct subset<std::tuple<T0, T...>, std::tuple<>> {
  using type = std::false_type;
  static constexpr bool value = false;
  static constexpr bool equiv = false;
};

template <class U0, class... U>
struct subset<std::tuple<>, std::tuple<U0, U...>> {
  using type = std::true_type;
  static constexpr bool value = true;
  static constexpr bool equiv = false;
};

template <class T0, class... T, class U>
struct subset<std::tuple<T0, T...>, U> {
  using count = count_in<T0, U>;
  using tail = subset<std::tuple<T...>, typename count::type>;
  using type = iif_t<count::value, typename tail::type, std::false_type>;
  static constexpr bool value = (count::value) ? tail::value : false;
  static constexpr bool equiv = (count::value) ? tail::equiv : false;
};

template <class T, class U>
using subset_t = typename subset<T, U>::type;

template <class L, class R>
using equivalent_t = std::integral_constant<bool, subset<L, R>::equiv>;
} // namespace mp
} // namespace ttl
