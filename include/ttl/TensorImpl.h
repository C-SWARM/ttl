// -*- C++ -*-
#ifndef TTL_TENSOR_IMPL_H
#define TTL_TENSOR_IMPL_H

#include <ttl/Tensor.h>
#include <ttl/Expressions/TensorBind.h>
#include <ttl/util/linearize.h>
#include <ttl/util/pow.h>
#include <cassert>
#include <tuple>
#include <algorithm>

namespace ttl {
template <int R, int D, class S>
class Tensor
{
  /// The number of elements in the tensor.
  static constexpr int Size = util::pow(D, R);

  template <class... Indices>
  using bind = expressions::TensorBind<Tensor, std::tuple<Indices...>>;

  template <class... Indices>
  using const_bind = expressions::TensorBind<const Tensor, std::tuple<Indices...>>;

 public:
  /// The scalar array backing the tensor.
  S data[Size];

  /// Assignment from a compatible tensor.
  template <class T>
  Tensor& operator=(const Tensor<R, D, T>& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }

  /// Fill all of the tensor elements with a scalar.
  template <class T>
  Tensor& fill(T t) {
    std::fill_n(data, Size, t);
    return *this;
  }

  /// Simple linear addressing.
  /// @{
  constexpr S operator[](int i) const {
    return data[i];
  }

  constexpr S& operator[](int i) {
    return data[i];
  }
  /// @}

  /// Multidimensional indexing for expressions.
  /// @{
  template <class Index>
  constexpr S eval(Index index) const {
    return data[linearize(index)];
  }

  template <class Index>
  constexpr S& eval(Index index) {
    return data[linearize(index)];
  }
  /// @}

  /// Create a tensor indexing expression for this tensor.
  ///
  /// This operation will bind the tensor to an Pack, which will allow us
  /// to make type inferences about the types of operations that should be
  /// available, as well as generate code to actually index the tensor in loops
  /// and to evaluate its elements.
  ///
  /// @code
  ///   static constexpr Index<'i'> i;
  ///   static constexpr Index<'j'> j;
  ///   static constexpr Index<'k'> k;
  ///   Tensor<3, double, 3> T;
  ///   auto expr = T(i, j, k);
  ///   T(i, j, k) = U(k, i, j);
  /// @code
  ///
  /// @tparam     Index The set of indices to bind to the tensor dimensions.
  ///
  /// @param     (anon) The actual set of indices to bind (e.g., (i,j,k)).
  ///
  /// @returns          A tensor indexing expression.
  template <class... Indices>
  bind<Indices...> operator()(Indices...) {
    static_assert(R == sizeof...(Indices), "Tensor indexing mismatch.");
    return bind<Indices...>(*this);
  }

  template <class... Indices>
  constexpr const const_bind<Indices...> operator()(Indices...) const {
    static_assert(R == sizeof...(Indices), "Tensor indexing mismatch.");
    return const_bind<Indices...>(*this);
  }

 private:
  template <class Index>
  static constexpr int linearize(Index index) {
    static_assert(R == std::tuple_size<Index>::value, "Invalid indexing width");
    return util::linearize<D>(index);
  }
};

/// A partial specialization of the tensor template for external storage.
///
/// This specialization is provided for compatibility with external storage
/// allocation. The user must specify a pointer to the external storage during
/// construction, which will then be used to store row-major tensor data.
template <int R, int D, class S>
class Tensor<R, D, S*>
{
  static constexpr int Size = util::pow(D, R);

  template <class... Indices>
  using bind = expressions::TensorBind<Tensor, std::tuple<Indices...>>;

  template <class... Indices>
  using const_bind = expressions::TensorBind<const Tensor, std::tuple<Indices...>>;

 public:
  /// Store a reference to the external data buffer.
  S (&data)[Size];

  /// The external storage tensor does not support default construction
  /// @{
  Tensor() = delete;
  /// @}

  Tensor(Tensor&& rhs) = delete;
  Tensor(const Tensor& rhs) : data(rhs.data) {
  }

  /// The only way to construct an external storage tensor is with the pointer
  /// to the external data buffer. The constructor simply captures a reference
  /// to this location.
  Tensor(S (*data)[Size]) : data(*data) {
  }

  Tensor(S* data) : Tensor(reinterpret_cast<S(*)[Size]>(data)) {
  }

  /// Assignment from a tensor is interpreted as a copy of the underlying data.
  /// @{
  Tensor& operator=(const Tensor& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }

  template <class T>
  Tensor& operator=(const Tensor<R, D, T>& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }
  /// @}

  /// Assignment from an initializer_list copies data to the external buffer.
  template <class T>
  Tensor& operator=(std::initializer_list<T>&& rhs) {
    /// @todo Static assert is okay with c++14.
    /// static_assert(data.size() == Size, "Initializer list has invalid length.");
    assert(rhs.size() == Size);
    std::copy_n(rhs.begin(), Size, data);
    return *this;
  }

  /// Fill the tensor with a scalar.
  template <class T>
  Tensor& fill(T scalar) {
    std::fill_n(data, Size, scalar);
    return *this;
  }

  constexpr S operator[](int i) const {
    return data[i];
  }

  S& operator[](int i) {
    return data[i];
  }

  /// Multidimensional indexing for Index tuples.
  /// @{
  template <class Index>
  constexpr S eval(Index index) const {
    return data[linearize(index)];
  }

  template <class Index>
  constexpr S& eval(Index index) {
    return data[linearize(index)];
  }
  /// @}

  template <class... Indices>
  constexpr const const_bind<Indices...> operator()(Indices...) const {
    static_assert(R == sizeof...(Indices), "Tensor indexing mismatch.");
    return const_bind<Indices...>(*this);
  }

  template <class... Indices>
  bind<Indices...> operator()(Indices...) {
    static_assert(R == sizeof...(Indices), "Tensor indexing mismatch.");
    return bind<Indices...>(*this);
  }

 private:
  template <class Index>
  static constexpr int linearize(Index index) {
    static_assert(R == std::tuple_size<Index>::value, "Invalid indexing width");
    return util::linearize<D>(index);
  }
};
} // namespace ttl

#endif // #ifndef TTL_TENSOR_IMPL_H
