// -*- C++ -*-
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

#include <type_traits>

namespace ttl {
/// The core class template for all tensors, which are really just
/// multidimensional dense arrays in TTL.
template <int Rank, int Dimension, class ScalarType>
class Tensor;

/// We need to do some metaprogramming with tensors where external functions
/// would like to know their template types. This tensor_traits structure, and
/// the associated traits type functions, provide us the ability to query the
/// traits for a tensor.
template <class T>
struct tensor_traits;

template <int R, int D, class T>
struct tensor_traits <Tensor<R, D, T>>
{
  using rank = std::integral_constant<int, R>;
  using dimension = std::integral_constant<int, D>;
  using scalar_type = typename std::remove_pointer<T>::type;
};
} // namespace ttl

#endif // #ifndef TTL_TENSOR_H
