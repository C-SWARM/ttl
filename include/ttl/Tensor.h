// -*- C++ -*-
#ifndef TTL_TENSOR_H
#define TTL_TENSOR_H

#include "ttl/Expression.h"
#include "ttl/Index.h"
#include <type_traits>

namespace ttl {
template <int Rank, typename Scalar, int Dimension>
class Tensor : public Expression<Tensor<Rank, Scalar, Dimension>> {
  static constexpr auto R = Rank;
  using S = Scalar;
  static constexpr auto D = Dimension;

 public:
  Tensor() : components_() {
  }

  template <typename ... Indices,
            typename = typename std::enable_if<sizeof...(Indices) == R>::type>
  auto operator()(Indices ... indices) {
    return expressions::Bind<Tensor<R, S, D>, IndexSet<Indices...>>(*this);
  }

 private:
  Tensor<Rank - 1, Scalar, Dimension> components_[Dimension];
};

template <typename Scalar, int Dimension>
class Tensor<0, Scalar, Dimension> : public Expression<Tensor<0, Scalar, Dimension>> {
 public:
  Tensor() : value_() {
  }

 private:
  Scalar value_;
};
}

#endif // TTL_TENSOR_H
