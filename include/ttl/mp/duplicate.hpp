#pragma once

#include <ttl/mp/cat.hpp>
#include <ttl/mp/count_in.hpp>
#include <ttl/mp/iif.hpp>
#include <tuple>

namespace ttl {
namespace mp {
/// Generate a tuple of the duplicate types in T by scanning through each type
/// and counting the number of instances of the type in the pack.
template <class T>
struct duplicate {
  using type = std::tuple<>;
};

template <class T0, class... T>
struct duplicate<std::tuple<T0, T...>> {
  using count = count_in<T0, std::tuple<T...>>;
  using  next = typename duplicate<typename count::type>::type;
  using  type = iif_t<count::value == 0, next, cat_t<T0, next>>;
};

template <class T>
using duplicate_t = typename duplicate<T>::type;

template <class L, class R>
using and_t = duplicate_t<cat_t<L, R>>;
} // namespace mp
} // namespace ttl
