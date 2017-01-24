// -*- C++ -*-------------------------------------------------------------------
/// This header includes some utility metafunctions to deal with type packs.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_PACK_H
#define TTL_EXPRESSIONS_PACK_H

#include <type_traits>

namespace ttl {
namespace expressions {
// -----------------------------------------------------------------------------
// Metaprogramming template declaration for dealing with packed parameters.
// -----------------------------------------------------------------------------
namespace detail {
template <class B, class T, class U> struct iif_impl;
template <class Pack> struct car_impl;
template <class Pack> struct cdr_impl;
template <class L, class R> struct concat_impl;
template <class L, class R> struct subset_impl;
template <class T, class Pack> struct remove_impl;
template <class Pacl> struct non_integral_impl;
template <class Pack> struct unique_impl;
template <class Pack> struct duplicate_impl;
template <int N, class T, class Pack> struct index_of_impl;
template <class Pack, int n, int N> struct print_impl;
} // namespace detail

// -----------------------------------------------------------------------------
// Convenience bindings for the metaprogramming templates.
// -----------------------------------------------------------------------------
template <class B, class T, class U>
using iif = typename detail::iif_impl<B, T, U>::type;

template <class Pack>
using car = typename detail::car_impl<Pack>::type;

template <class Pack>
using cdr = typename detail::cdr_impl<Pack>::type;

template <class L, class R>
using concat = typename detail::concat_impl<L, R>::type;

template <class L, class R>
using subset = typename detail::subset_impl<L, R>::type;

template <class T, class Pack>
using remove = typename detail::remove_impl<T, Pack>::type;

template <class Pack>
using non_integral = typename detail::non_integral_impl<Pack>::type;

template <class Pack>
using unique = typename detail::unique_impl<Pack>::type;

template <class Pack>
using duplicate = typename detail::duplicate_impl<Pack>::type;

/// Get the index of a type in a parameter pack.
///
/// @tparam           T The type that we're searching for.
/// @tparam...        U The pack that we're searching.
template <class T, class Pack>
using index_of = typename detail::index_of_impl<0, T, Pack>::type;

/// Two packs are equivalent when they are subsets of each other.
template <class L, class R>
using equivalent = iif<subset<L, R>, subset<R, L>, std::false_type>;

template <class L, class R>
using set_xor = unique<concat<L, R>>;

template <class L, class R>
using set_and = duplicate<concat<L, R>>;

// -----------------------------------------------------------------------------
// Implementations of the metaprogramming functions.
// -----------------------------------------------------------------------------
namespace detail {
template <class B, class T, class U>
struct iif_impl {
  using type = T;
};

template <class T, class U>
struct iif_impl<std::false_type, T, U> {
  using type = U;
};

template <template <class...> class Pack>
struct car_impl<Pack<>>
{
  using type = Pack<>;                          // base case is empty list
};

template <template <class...> class Pack, class T0, class... T>
struct car_impl<Pack<T0, T...>>
{
  using type = Pack<T0>;                        // return the head of the list
};

template <template <class...> class Pack>
struct cdr_impl<Pack<>>
{
  using type = Pack<>;                          // base case is empty list
};

template <template <class...> class Pack, class T0, class... T>
struct cdr_impl<Pack<T0, T...>>
{
  using type = Pack<T...>;                      // return the tail of the list
};

template <template <class...> class Pack, class... T, class... U>
struct concat_impl<Pack<T...>, Pack<U...>>
{
  using type = Pack<T..., U...>;                // join the two packs
};

template <template <class...> class Pack>
struct subset_impl<Pack<>, Pack<>>
{
  using type = std::true_type;                  // empty set is always in U
};

template <template <class...> class Pack, class U>
struct subset_impl<Pack<>, U>
{
  using type = std::true_type;                  // empty set is always in U
};

template <class T, template <class...> class Pack>
struct subset_impl<T, Pack<>>
{
  using type = std::false_type;                 // T is never in empty set
};

template <template <class...> class Pack, class T, class... U>
struct subset_impl<Pack<T>, Pack<T, U...>>
{
  using type = std::true_type;                  // T is matched
};

template <template <class...> class Pack, class T, class U0, class... U>
struct subset_impl<Pack<T>, Pack<U0, U...>>
{
  using type = subset<Pack<T>, Pack<U...>>;     // test the tail
};

template <class T, class U>
struct subset_impl
{
  // if the head of the pack, T, is a subset of U, continue processing the tail
  // of T, otherwise subset fails
  using head = car<T>;
  using tail = cdr<T>;
  using next = subset<tail, U>;
  using type = iif<subset<head, U>, next, std::false_type>;
};

template <template <class...> class Pack, class T>
struct remove_impl<Pack<T>, Pack<>>
{
  using type = Pack<>;                          // base case is empty list
};

template <class T, class Pack>
struct remove_impl
{
  // process the tail of the pack, and then append the head to the returned list
  // if it doesn't match T
  using head = car<Pack>;
  using tail = cdr<Pack>;
  using next = remove<T, tail>;
  using type = iif<typename std::is_same<T, head>::type, next, concat<head, next>>;
};

template <template <class...> class Pack>
struct non_integral_impl<Pack<>>
{
  using type = Pack<>;
};

template <class Pack>
struct non_integral_impl
{
  using head = car<Pack>;
  using tail = cdr<Pack>;
  using next = non_integral<tail>;
  using type = iif<std::is_integral<head>, concat<head, next>, next>;
};

template <template <class...> class Pack>
struct unique_impl<Pack<>>
{
  using type = Pack<>;                          // base case is empty list
};

template <class Pack>
struct unique_impl
{
  // Filter out any instances of the head from the tail of the pack and find the
  // unique types in that restricted list. Then, for this version of the list,
  // if the head is not in the tail then append it to the returned list,
  // otherwise just return the list.
  using head = car<Pack>;
  using tail = cdr<Pack>;
  using next = unique<remove<head, tail>>;
  using type = iif<subset<head, tail>, next, concat<head, next>>;
};

template <template <class...> class Pack>
struct duplicate_impl<Pack<>>
{
  using type = Pack<>;                          // base case is empty list
};

template <class Pack>
struct duplicate_impl
{
  // Filter out any instances of the head from the tail of the pack and find the
  // duplicate types in that restricted list. Then, for this version of the list,
  // if the head is in the tail then append it to the returned list, otherwise
  // just return the list.
  using head = car<Pack>;
  using tail = cdr<Pack>;
  using next = duplicate<remove<head, tail>>;
  using type = iif<subset<head, tail>, concat<head, next>, next>;
};

template <int N, template <class...> class Pack, class T>
struct index_of_impl<N, Pack<T>, Pack<>>
{
  using type = std::integral_constant<int, N>;
};

template <int N, class T, class Pack>
struct index_of_impl
{
  using head = car<Pack>;
  using tail = cdr<Pack>;
  using match = typename std::is_same<T, head>::type;
  using next = typename index_of_impl<N + 1, T, tail>::type;
  using type = iif<match, std::integral_constant<int, N>, next>;
};

template <class Pack, int N>
struct print_impl<Pack,N,N>
{
  static std::ostream& op(std::ostream& os, const Pack&) {
    return os;
  }
};

template <class Pack,
          int n = 0,
          int N = std::tuple_size<Pack>::value>
struct print_impl
{
  static std::ostream& op(std::ostream& os, const Pack& pack) {
    os << std::get<n>(pack) << ", ";
    return print_impl<Pack, n+1>::op(os, pack);
  }
};

} // namespace detail
template <class Pack>
std::ostream& print_pack(std::ostream& os, const Pack& pack) {
  return detail::print_impl<Pack>::op(os, pack);
}
} // namespace expressions
} // namespace ttl

#endif // #define TTL_EXPRESSIONS_PACK_H
