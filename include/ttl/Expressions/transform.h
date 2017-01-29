// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TRANSFORM_H
#define TTL_EXPRESSIONS_TRANSFORM_H

#include <ttl/Expressions/pack.h>
#include <tuple>

namespace ttl {
namespace expressions {
namespace detail {

/// Select, at compile time, which value to use for a tuple element.
///
/// In the transform template we would really like to be able to write something
/// like:
///
/// @code
///    if (is_integral<T>) : std::get<n>(to) : std::get<index_of(T, From>>(from);
/// @code
///
/// Unfortunately, when T is an integral type then it won't be found in the
/// index_of test in the from index. This ends up causing a compilation error
/// because, e.g., std::get<1>(std::tuple<>{}) doesn't compile. In order to
/// avoid this problem we use this helper template such that we can partially
/// specialize to avoid trying to expand that illegal get<> when the type is
/// integral.
///
/// @returns          (is_integral<T>::value) ? to : find(from)
template <class T, bool Integral = std::is_integral<T>::value>
struct select
{
  /// This specialization finds the index of the type T in the From pack, and
  /// returns its value. We check to make sure that the index is actually
  /// found.
  template <class From,
            int n = index_of<T, From>::value,     // intel can't handle these in
            int N = std::tuple_size<From>::value> // constexpr function
  static constexpr T op(const T, const From from) {
    static_assert(n < N, "Index space is incompatible");
    return std::get<n>(from);
  }
};

template <class T>
struct select<T, true>
{
  /// The integral type version is really easy, we can just return the value
  /// we're passed.
  template <class From>
  static constexpr T op(const T to, From) {
    return to;
  }
};

/// The transform implementation creates a tuple in the target space.
///
/// We implement this template by iterating over each tuple, and concatenating
/// the values together.
template <class Index, int n = 0, int N = std::tuple_size<Index>::value>
struct transform_impl
{
  template <class From>
  static constexpr auto op(const Index to, const From from) noexcept {
    return std::tuple_cat(head(std::get<n>(to), from), tail(to, from));
  }

 private:
  template <class From, class T>
  static constexpr auto head(const T to, const From from) noexcept {
    return std::make_tuple(select<T>::op(to, from));
  }

  template <class From>
  static constexpr auto tail(const Index to, const From from) noexcept {
    return transform_impl<Index, n+1>::op(to, from);
  }
};

// Base case is empty pack.
template <template <class...> class Pack, class... I, int N>
struct transform_impl<Pack<I...>, N, N>
{
  template <class From>
  static constexpr auto op(const Pack<I...>, const From) noexcept {
    return Pack<>{};
  }
};

} // namespace detail

template <class To, class From>
constexpr To transform(const To to, const From from) noexcept {
  return detail::transform_impl<To>::op(to, from);
}

template <class To, class From>
constexpr To transform(const From from) noexcept {
  return detail::transform_impl<To>::op(To{}, from);
}
} // namespace expressions
} // namespace ttl

#endif // #ifndef TTL_EXPRESSIONS_TRANSFORM_H
