// -*- C++ -*-
#ifndef TTL_DETAIL_SHUFFLE_H
#define TTL_DETAIL_SHUFFLE_H

#include <ttl/Pack.h>
#include <ttl/Index.h>

namespace ttl {
namespace detail {

template <int Rank, class T, class U>
struct shuffle_impl;

template <int Rank, class T0, class... T, class... U>
struct shuffle_impl<Rank, Pack<T0, T...>, Pack<U...>> {
  static IndexSet<Rank> op(IndexSet<Rank> i) {
    auto out = shuffle_impl<Rank, Pack<T...>, Pack<U...>>::op(i);
    out[indexof<T0, U...>::value] = i[Rank - sizeof...(T) - 1];
    return out;
  }
};

template <int Rank, class... U>
struct shuffle_impl<Rank, Pack<>, Pack<U...>> {
  static constexpr IndexSet<Rank> op(IndexSet<Rank> i) {
    return i;
  }
};

template <int Rank, class T, class U>
inline constexpr IndexSet<Rank> shuffle(IndexSet<Rank> i) {
  return (is_equal<T, U>::value) ? i : shuffle_impl<Rank, T, U>::op(i);
}
} // namespace detail
} // namespace ttl

#endif // #ifndef TTL_DETAIL_SHUFFLE_H
