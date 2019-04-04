#pragma once

#include <ttl/mp/cat.hpp>
#include <ttl/mp/count_in.hpp>
#include <ttl/mp/iif.hpp>
#include <tuple>

namespace ttl {
namespace mp {
template <class T, class U>
struct unique {
  using type = std::tuple<>;
};

template <class T0, class... T, class U>
struct unique<std::tuple<T0, T...>, U> {
  using count = count_in<T0, U>;
  using next = typename unique<std::tuple<T...>, U>::type;
  using type = iif_t<count::value != 1, next, cat_t<T0, next>>;
};

template <class T>
using unique_t = typename unique<T, T>::type;

template <class L, class R>
using xor_t = unique_t<cat_t<L, R>>;
} // namespace mp
} // namespace ttl
