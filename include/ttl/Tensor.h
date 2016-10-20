// -*- C++ -*-
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

namespace ttl {
/// The core class template for all tensors, which are really just
/// multidimensional dense arrays (ScalarType data[Dimension^Rank]) in TTL.
template <int Rank, int Dimension, class ScalarType>
class Tensor;
} // namespace ttl

#endif // #ifndef TTL_TENSOR_H
