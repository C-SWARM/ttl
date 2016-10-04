// -*- C++ -*-
#ifndef TTL_DETAIL_SHUFFLE_H
#define TTL_DETAIL_SHUFFLE_H

#include <ttl/Pack.h>

namespace ttl {
namespace detail {

template <int Rank, class T, class U>
struct shuffle_impl;

template <int Rank, class... T>
struct shuffle_impl<Rank, Pack<T...>, Pack<T...>> {
  static std::array<int, Rank> op(std::array<int, Rank> i) {
    return i;
  }
};

template <int Rank, class T0, class... T, class... U>
struct shuffle_impl<Rank, Pack<T0, T...>, Pack<U...>> {
  static std::array<int, Rank> op(std::array<int, Rank> i) {
    auto out = shuffle_impl<Rank, Pack<T...>, Pack<U...>>::op(i);
    out[indexof<T0, U...>::value] = i[Rank - sizeof...(T) - 1];
    return out;
  }
};

template <int Rank, class... U>
struct shuffle_impl<Rank, Pack<>, Pack<U...>> {
  static std::array<int, Rank> op(std::array<int, Rank> i) {
    return i;
  }
};

template <int Rank, class T, class U>
inline std::array<int, Rank> shuffle(std::array<int, Rank> i) {
  return shuffle_impl<Rank, T, U>::op(i);
}
}
}

#endif // #ifndef TTL_DETAIL_SHUFFLE_H
