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
template <int R, int D, class T>
class Tensor
{
  /// The number of elements in the tensor.
  static constexpr int Size = util::pow(D, R);

  template <class... Index>
  using bind = expressions::TensorBind<Tensor, std::tuple<Index...>>;

  template <class... Index>
  using const_bind = expressions::TensorBind<const Tensor, std::tuple<Index...>>;

 public:
  /// The scalar array backing the tensor.
  T data[Size];

  /// Assignment from a compatible tensor.
  template <class U>
  Tensor& operator=(const Tensor<R, D, U>& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }

  /// Fill all of the tensor elements with a scalar.
  template <class U>
  Tensor& fill(U scalar) {
    std::fill_n(data, Size, scalar);
    return *this;
  }

  /// Simple linear addressing.
  /// @{
  constexpr const T& operator[](int i) const {
    return data[i];
  }

  T& operator[](int i) {
    return data[i];
  }
  /// @}

  /// Multidimensional indexing for expressions.
  /// @{
  template <class I>
  constexpr T get(I index) const {
    static_assert(R == std::tuple_size<I>::value, "Invalid indexing width");
    return data[util::linearize<D>(index)];
  }

  template <class I, class U>
  void set(I index, U scalar) {
    static_assert(R == std::tuple_size<I>::value, "Invalid indexing width");
    data[util::linearize<D>(index)] = scalar;
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
  template <class... Index>
  bind<Index...> operator()(Index...) {
    static_assert(R == sizeof...(Index), "Tensor indexing mismatch.");
    return bind<Index...>(*this);
  }

  template <class... Index>
  constexpr const const_bind<Index...> operator()(Index...) const {
    static_assert(R == sizeof...(Index), "Tensor indexing mismatch.");
    return const_bind<Index...>(*this);
  }
};

/// A partial specialization of the tensor template for external storage.
///
/// This specialization is provided for compatibility with external storage
/// allocation. The user must specify a pointer to the external storage during
/// construction, which will then be used to store row-major tensor data.
template <int R, int D, class T>
class Tensor<R, D, T*>
{
  static constexpr int Size = util::pow(D, R);

  template <class... Index>
  using bind = expressions::TensorBind<Tensor, std::tuple<Index...>>;

  template <class... Index>
  using const_bind = expressions::TensorBind<const Tensor, std::tuple<Index...>>;

 public:
  /// Store a reference to the external data buffer.
  T (&data)[Size];

  /// The external storage tensor does not support default or copy construction.
  /// @{
  Tensor() = delete;
  Tensor(Tensor&& rhs) = delete;
  Tensor(const Tensor&) = delete;
  /// @}

  /// The only way to construct an external storage tensor is with the pointer
  /// to the external data buffer. The constructor simply captures a reference
  /// to this location.
  Tensor(T (*data)[Size]) : data(*data) {
  }

  Tensor(T* data) : Tensor(reinterpret_cast<T(*)[Size]>(data)) {
  }

  /// Assignment from a tensor is interpreted as a copy of the underlying data.
  /// @{
  Tensor& operator=(const Tensor& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }

  template <class U>
  Tensor& operator=(const Tensor<R, D, U>& rhs) {
    std::copy_n(rhs.data, Size, data);
    return *this;
  }
  /// @}

  /// Assignment from an initializer_list copies data to the external buffer.
  template <class U>
  Tensor& operator=(std::initializer_list<U>&& rhs) {
    /// @todo Static assert is okay with c++14.
    /// static_assert(data.size() == Size, "Initializer list has invalid length.");
    assert(rhs.size() == Size);
    std::copy_n(rhs.begin(), Size, data);
    return *this;
  }

  /// Fill the tensor with a scalar.
  template <class U>
  Tensor& fill(U scalar) {
    std::fill_n(data, Size, scalar);
    return *this;
  }

  constexpr T operator[](int i) const {
    return data[i];
  }

  T& operator[](int i) {
    return data[i];
  }

  /// Multidimensional indexing for Index tuples.
  /// @{
  template <class I>
  constexpr T get(I index) const {
    static_assert(R == std::tuple_size<I>::value, "Invalid indexing width");
    return data[util::linearize<D>(index)];
  }

  template <class I, class U>
  void set(I index, U scalar) {
    static_assert(R == std::tuple_size<I>::value, "Invalid indexing width");
    data[util::linearize<D>(index)] = scalar;
  }
  /// @}

  template <class... Index>
  constexpr const const_bind<Index...> operator()(Index...) const {
    static_assert(R == sizeof...(Index), "Tensor indexing mismatch.");
    return const_bind<Index...>(*this);
  }

  template <class... Index>
  bind<Index...> operator()(Index...) {
    static_assert(R == sizeof...(Index), "Tensor indexing mismatch.");
    return bind<Index...>(*this);
  }
};
} // namespace ttl

#endif // #ifndef TTL_TENSOR_IMPL_H
