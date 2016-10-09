// -*- C++ -*-
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

namespace ttl {
/// The core class template for all tensors, which are really just
/// multidimensional dense arrays in TTL.
template <int Rank, typename ScalarType, int Dimension>
class Tensor;

/// We need to do some metaprogramming with tensors where external functions
/// would like to know their template types. This tensor_traits structure, and
/// the associated traits type functions, provide us the ability to query the
/// traits for a tensor.
template <class T>
struct tensor_traits;

template <int Rank, typename ScalarType, int Dimension>
struct tensor_traits <Tensor<Rank, ScalarType, Dimension>>
{
  static constexpr int rank = Rank;
  static constexpr int dimension = Dimension;
  using scalar_type = ScalarType;
};
} // namespace ttl

#endif // #ifndef TTL_TENSOR_H
