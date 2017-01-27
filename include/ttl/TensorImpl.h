// -*- C++ -*-
#ifndef TTL_TENSOR_IMPL_H
#define TTL_TENSOR_IMPL_H

#include <ttl/Tensor.h>
#include <ttl/Expressions.h>
#include <ttl/util/linearize.h>
#include <ttl/util/multi_array.h>
#include <ttl/util/pow.h>
#include <cassert>
#include <tuple>
#include <algorithm>

namespace ttl {

/// Common functionality for the tensor specializations.
///
/// The TensorBase implements a quasi-CRTP pattern in order to statically
/// dispatch based on the subclass implementation of the data storage.
///
/// @tparam           R The tensor's rank.
/// @tparam           D The tensor's dimension.
/// @tparam           S The tensor's scalar storage (only used explicitly when
///                     casting to the derived tensor type).
template <int R, int D, class S>
class TensorBase
{
 private:
  static constexpr std::size_t Size = util::pow(D,R); ///!< Number of elements

  constexpr const auto& derived() const noexcept {
    return *static_cast<const Tensor<R,D,S>* const>(this);
  }

  constexpr auto& derived() noexcept {
    return *static_cast<Tensor<R,D,S>* const>(this);
  }

 protected:
  template <class T>
  auto& copy(std::initializer_list<T> list) noexcept {
    // http://stackoverflow.com/questions/8452952/c-linker-error-with-class-static-constexpr
    auto size = Size;
    std::size_t min = std::min(size, list.size());
    std::copy_n(list.begin(), min, derived().data);     // copy prefix
    std::fill_n(&derived().data[min], Size - min, 0);   // 0-fill suffix
    return derived();
  }

  template <class T>
  auto& copy(const Tensor<R,D,T>& rhs) noexcept {
    std::copy_n(rhs.data, Size, derived().data);
    return derived();
  }

  template <class T>
  auto& copy(Tensor<R,D,T>&& rhs) noexcept {
    std::copy_n(std::move(rhs.data), Size, derived().data);
    return derived();
  }

  template <class Index>                        // required for icc 16
  constexpr const auto bind(const Index index) const noexcept {
    return expressions::make_bind(derived(), index);
  }

  template <class Index>                        // required for icc 16
  constexpr auto bind(const Index index) noexcept {
    return expressions::make_bind(derived(), index);
  }

  template <class E>
  auto& apply(E&& rhs) noexcept {
    bind(expressions::outer_type<E>{}) = std::forward<E>(rhs);
    return derived();
  }

 public:
  /// Fill a tensor with a scalar value.
  ///
  /// @code
  ///   auto T = Tensor<R,D,int>().fill(1);
  /// @code
  ///
  /// @tparam         T The type of the scalar, must be compatible with S.
  /// @param     scalar The actual scalar.
  /// @returns          A reference to the tensor so that fill() can be
  ///                   chained.
  template <class T>
  auto& fill(T scalar) noexcept {
    std::fill_n(derived().data, Size, scalar);
    return derived();
  }

  /// Basic linear indexing into the tensor.
  ///
  /// @code
  ///   Tensor<R,D,int> T;
  ///   int i = T[0];
  /// @code
  ///
  /// @param          i The index to access.
  /// @returns          The scalar value at @p i.
  constexpr const auto get(int i) const noexcept {
    return derived().data[i];
  }

  /// Basic linear indexing into the tensor.
  ///
  /// @code
  ///   Tensor<R,D,int> T;
  ///   T[0] = 42;
  /// @code
  ///
  /// @param          i The index to access.
  /// @returns          A reference to the scalar value at @p i.
  constexpr auto& get(int i) noexcept {
    return derived().data[i];
  }

  /// Multidimensional indexing into the tensor, used during evaluation.
  ///
  /// @code
  ///   Tensor<R,D,int> T
  ///   Index I = {0,1,...}
  ///   int i = T.eval(I)
  /// @code
  ///
  /// @tparam     Index The tuple type for the index.
  /// @param      index The multidimension index to access.
  /// @returns          The scalar value at the linearized @p index.
  template <class Index>
  constexpr const auto eval(Index index) const noexcept {
    using NIndex = std::tuple_size<Index>;
    static_assert(R == NIndex::value, "Index size does not match tensor rank");
    return derived().data[util::linearize<D>(index)];
  }

  /// Multidimensional indexing into the tensor, used during evaluation.
  ///
  /// @code
  ///   Tensor<R,D,int> T
  ///   Index I = {0,1,...}
  ///   T.eval(I) = 42
  /// @code
  ///
  /// @tparam     Index The tuple type for the index.
  /// @param      index The multidimension index to access.
  /// @returns          The a reference to the scalar value at the linearized @p
  ///                   index.
  template <class Index, int N = std::tuple_size<Index>::value>
  constexpr auto& eval(Index index) noexcept {
    static_assert(R == N, "Index size does not match tensor rank");
    return derived().data[util::linearize<D>(index)];
  }

  /// Bind a Bind expression to a tensor.
  ///
  /// This is the core operation that provides the tensor syntax that we want to
  /// provide. The user binds a tensor to a set of indices, and the TTL
  /// expression template structure can
  ///
  /// 1) Type check expressions.
  /// 2) Generate loops over indices during evaluation and contraction.
  ///
  /// @code
  ///  Index<'i'> i;
  ///  Index<'j'> j;
  ///  Index<'k'> k;
  ///  Tensor<R,D,S> A, B, C;
  ///  C(i,j) = A(i,k)*B(k,j)
  /// @code
  ///
  /// @tparam...      I The index types to bind to the tensor.
  /// @param... indices The index values.
  /// @returns          A Bind expression that can serves as the leaf
  ///                   expression in TTL expressions.
  template <class... I>
  constexpr const auto operator()(I... indices) const noexcept {
    static_assert(R == sizeof...(I), "Index size does not match tensor rank");
    return bind(std::make_tuple(indices...));
  }

  /// Bind a Bind expression to a tensor.
  ///
  /// This is the core operation that provides the tensor syntax that we want to
  /// provide. The user binds a tensor to a set of indices, and the TTL
  /// expression template structure can
  ///
  /// 1) Type check expressions.
  /// 2) Generate loops over indices during evaluation and contraction.
  ///
  /// @code
  ///  Index<'i'> i;
  ///  Index<'j'> j;
  ///  Index<'k'> k;
  ///  Tensor<R,D,S> A, B, C;
  ///  C(i,j) = A(i,k)*B(k,j)
  /// @code
  ///
  /// @tparam...      I The index types to bind to the tensor.
  /// @param... indices The index values.
  /// @returns          A Bind expression that can serves as the leaf
  ///                   expression in TTL expressions.
  template <class... I>
  constexpr auto operator()(I... indices) noexcept {
    static_assert(R == sizeof...(I), "Index size does not match tensor rank");
    return bind(std::make_tuple(indices...));
  }
};

/// The normal tensor type.
///
/// The Tensor stores an array of scalar values, and provides the necessary
/// constructors and assignment operators so that the tensor can be used as a
/// value-type aggregate.
template <int R, int D, class S>
class Tensor : public TensorBase<R,D,S>
{
 public:
  /// Standard default constructor.
  ///
  /// The default constructor leaves the tensor data uninitialized. To
  /// initialize a tensor to 0 use Tensor<R,D,S> T = {};
  constexpr Tensor() noexcept = default;
  constexpr Tensor(const Tensor&) noexcept = default;
  constexpr Tensor(Tensor&&) noexcept = default;

  /// Allow list initialization of tensors.
  ///
  /// @code
  ///  Tensor<R,D,S> T = {0,1,...};
  /// @code
  ///
  /// @param       list The initializer list for the tensor.
  Tensor(std::initializer_list<S> list) noexcept {
    this->copy(list);
  }

  /// Allow initialization from tensors of compatible type.
  ///
  /// @code
  ///  Tensor<R,D,double> A;
  ///  Tensor<R,D,const double> B;
  ///  Tensor<R,D,int> C;
  ///  const int d[]={...};
  ///  Tensor<R,D,const int*> D(d);
  ///  A = B; // copy from const tensor
  ///  A = C; // copy and promote type
  ///  A = D; // copy from const external tensor
  /// @code
  ///
  /// The copy is performed using the promotion and compatibility rules of the
  /// std::copy_n algorithm.
  ///
  /// @tparam         T The scalar type for the right hand side.
  /// @param        rhs The tensor to copy from.
  template <class T>
  Tensor(const Tensor<R,D,T>& rhs) noexcept {
    this->copy(rhs);
  }

  /// Allow initialization from tensors of compatible type.
  ///
  /// @code
  ///  Tensor<R,D,double> A;
  ///  const int d[]={...};
  ///  A = std::move(Tensor<R,D,const double>{}); // move from const tensor
  ///  A = std::move(Tensor<R,D,int>{});          // move and promote type
  ///  A = std::move(Tensor<R,D,const int*>(d));  // move const external tensor
  /// @code
  ///
  /// The copy is performed using the promotion and compatibility rules of the
  /// std::copy_n algorithm.
  ///
  /// @tparam         T The scalar type for the right hand side.
  /// @param        rhs The tensor rvalue to copy from.
  template <class T>
  Tensor(Tensor<R,D,T>&& rhs) noexcept {
    this->copy(std::move(rhs));
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   Tensor<R,D,S> A = B(i,j)
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(const expressions::Expression<E>&& rhs) noexcept {
    this->apply(std::move(rhs));
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   auto b = B(i,j)
  ///   Tensor<R,D,S> A = b
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(const expressions::Expression<E>& rhs) noexcept {
    this->apply(rhs);
  }

  /// Normal assignment and move operators.
  constexpr Tensor& operator=(const Tensor&) noexcept = default;
  constexpr Tensor& operator=(Tensor&&) noexcept = default;

  /// Allow assignment from expressions of compatible type without explicit bind
  ///
  /// @code
  ///   Tensor<R,D,S> A;
  ///   A = B(i,j)
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  constexpr Tensor& operator=(const expressions::Expression<E>&& rhs) noexcept {
    return this->apply(std::move(rhs));
  }

  /// Allow assignment from expressions of compatible type without explicit bind
  ///
  /// @code
  ///   auto b = B(i,j)
  ///   Tensor<R,D,S> A;
  ///   A = b
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  constexpr Tensor& operator=(const expressions::Expression<E>& rhs) noexcept {
    return this->apply(rhs);
  }

  constexpr auto operator[](int i) const noexcept {
    using Multi = util::multi_array<R,D,const S>;
    return (*reinterpret_cast<const Multi*>(&data))[i];
  }

  /// Direct multidimensional array access to the data.
  constexpr auto operator[](int i) noexcept
    -> util::multi_array<R-1,D,S>& // icc https://software.intel.com/en-us/forums/intel-c-compiler/topic/709454
  {
    using Multi = util::multi_array<R,D,S>;
    return (*reinterpret_cast<Multi*>(&data))[i];
  }

  // We remove the constness from the type for tensors so that we can use the
  // default constructor to leave the data uninitialized. If we don't do this
  // then expressions like Tensor<R,D,const S> T = {}; don't work.
  std::remove_const_t<S> data[util::pow(D,R)];  ///!< scalar storage
};

/// The tensor specialization for external storage.
///
/// During construction this specialization captures a reference to external
/// storage, and then tries to provide the same value semantics as the basic
/// tensor class.
template <int R, int D, class S>
class Tensor<R,D,S*> : public TensorBase<R,D,S*>
{
  static constexpr int Size = util::pow(D,R);

 public:
  /// No default construction (we _must_ have external space)
  constexpr Tensor() noexcept = delete;

  /// Copy and move constructors just capture the same external buffer.
  constexpr Tensor(const Tensor&) noexcept = default;
  constexpr Tensor(Tensor&&) noexcept = default;

  /// Initialize the tensor with an external buffer of data.
  ///
  /// This constructor captures a reference to a buffer of the right type and
  /// size.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,int*> A(a);
  ///   Tensor<1,4,int*> A(a);
  ///   Tensor<2,2,const int*> A(a);
  /// @code
  ///
  /// @param       data The external buffer.
  Tensor(S (*data)[Size]) : data(*data) {
  }

  /// Allow a simple pointer to be used as the external buffer.
  ///
  /// This constructor interprets the pointer as the right sized buffer.
  ///
  /// @code
  ///   int a[8];
  ///   Tensor<2,2,int*> A(&a[4]);
  /// @code
  ///
  /// @param       data The external buffer.
  Tensor(S* data) : Tensor(reinterpret_cast<S(*)[Size]>(data)) {
  }

  /// Copy construction makes the new tensor use the same buffer as the rhs.
  ///
  /// This uses built in type promotion rules to ensure that a pointer to the @p
  /// T type is compatible with a pointer to the @p S type. We expect this to
  /// only work for cv types.
  ///
  /// This will only match external tensor types. It is not possible to
  /// initialize an external tensor with an internal tensor.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,int*> A(a);
  ///   Tensor<2,2,const int*> B = A;
  /// @code
  ///
  /// @tparam       T The pointed to scalar type for the right-hand-side.
  /// @param      rhs The right hand side tensor.
  template <class T>
  Tensor(const Tensor<R,D,T*>& rhs) noexcept : data(rhs.data) {
  }

  /// Move construction captures the buffer from the rhs.
  ///
  ///
  /// This uses built in type promotion rules to ensure that a pointer to the @p
  /// T type is compatible with a pointer to the @p S type. We expect this to
  /// only work for cv types.
  ///
  /// This will only match external tensor types. It is not possible to
  /// initialize an external tensor with an internal tensor.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,const int*> B = std::move(Tensor<2,2,int*>(a));
  /// @code
  ///
  /// @tparam       T The pointed to scalar type for the right-hand-side.
  /// @param      rhs The right hand side tensor.
  template <class T>
  Tensor(Tensor<R,D,T*>&& rhs) noexcept : data(std::move(rhs.data)) {
  }

  /// Allow list initialization of tensors.
  ///
  /// @code
  ///  int a[]
  ///  Tensor<R,D,S*> T = {a , {...}};
  /// @code
  ///
  /// @param       list The initializer list for the tensor.
  Tensor(S* data, std::initializer_list<S> list) noexcept : Tensor(data) {
    this->copy(list);
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   int a[]
  ///   const Tensor<R,D,S*> A = {a, B(i,j)};
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param       data The external data buffer.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(S* data, const expressions::Expression<E>&& rhs) noexcept
      : Tensor(data)
  {
    this->apply(std::move(rhs));
  }

  /// Allow initialization from expressions of compatible type.
  ///
  /// @code
  ///   auto b = B(i,j);
  ///   int a[]
  ///   const Tensor<R,D,S*> A = {a, b};
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param       data The external data buffer.
  /// @param        rhs The right hand side expression.
  template <class E>
  Tensor(S* data, const expressions::Expression<E>& rhs) noexcept
      : Tensor(data)
  {
    this->apply(rhs);
  }

  /// Copy the data from the rhs.
  ///
  /// As opposed to construction, the copy operator actually copies the data
  /// from the rhs tensor. This implements the value semantics expected from
  /// tensors.
  ///
  /// @param        rhs The right hand side of the assignment.
  /// @returns          A reference to *this;
  constexpr Tensor& operator=(const Tensor& rhs) noexcept {
    return this->copy(rhs);
  }

  /// Move the data from the rhs.
  ///
  /// As opposed to construction, the move operator actually copies the data
  /// from the rhs tensor. This implements the value semantics expected from
  /// tensors.
  ///
  /// @param        rhs The right hand side of the assignment.
  /// @returns          A reference to *this;
  constexpr Tensor& operator=(Tensor&& rhs) noexcept {
    return this->copy(std::move(rhs));
  }

  /// Assign an initializer list to the external tensor.
  ///
  /// The normal "in-place" storage tensor gets this functionality from its
  /// initializer list constructor, but we need it explicitly since we have no
  /// initializer list constructor.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,int*> A(a);
  ///   A = {0,1,2,3};
  /// @code
  ///
  /// @param        rhs The right hand side of the assignment.
  /// @returns          A reference to *this.
  constexpr Tensor& operator=(std::initializer_list<S> list) noexcept {
    return this->copy(list);
  }

  /// Copy the data from any type of right hand side tensor.
  ///
  /// This supports both in place and external tensor assignments, and
  /// implements the expected value-type copy. T->S compatibility is governed by
  /// the std::copy_n semantics.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,int*> A(a);
  ///   Tensor<2,2,int> B = {0,1,2,3};
  ///   A = B;
  /// @code
  ///
  /// @tparam         T The scalar type for the right hand side tensor.
  /// @param        rhs The right hand side tensor.
  /// @returns          A reference to *this.
  template <class T>
  constexpr Tensor& operator=(const Tensor<R,D,T>& rhs) noexcept {
    return this->copy(rhs);
  }

  /// Copy the data from any type of right hand side tensor.
  ///
  /// This supports both in place and external tensor assignments, and
  /// implements the expected value-type copy. T->S compatibility is governed by
  /// the std::copy_n semantics.
  ///
  /// @code
  ///   int a[4];
  ///   Tensor<2,2,int*> A(a);
  ///   Tensor<2,2,int> B = {0,1,2,3};
  ///   A = B;
  /// @code
  ///
  /// @tparam         T The scalar type for the right hand side tensor.
  /// @param        rhs The right hand side tensor.
  /// @returns          A reference to *this.
  template <class T>
  constexpr Tensor& operator=(Tensor<R,D,T>&& rhs) noexcept {
    return this->copy(std::move(rhs));
  }

  /// Allow assignment from expressions of compatible type without explicit bind
  ///
  /// @code
  ///   int a[]
  ///   Tensor<R,D,S*> A(a);
  ///   A = B(i,j)
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  constexpr Tensor& operator=(const expressions::Expression<E>&& rhs) noexcept {
    return this->apply(std::move(rhs));
  }

  /// Allow assignment from expressions of compatible type without explicit bind
  ///
  /// @code
  ///   auto b = B(i,j);
  ///   int a[]
  ///   Tensor<R,D,S*> A(a);
  ///   A = b
  /// @code
  ///
  /// @tparam         E The type of the right-hand-side expression.
  /// @param        rhs The right hand side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  constexpr Tensor& operator=(const expressions::Expression<E>& rhs) noexcept {
    return this->apply(rhs);
  }

  /// Direct multidimensional array access to the data.
  constexpr const auto operator[](int i) const noexcept {
    using Multi = util::multi_array<R,D,const S>;
    return (*reinterpret_cast<const Multi*>(&data))[i];
  }

  constexpr auto operator[](int i) noexcept
    -> util::multi_array<R-1,D,S>& // icc https://software.intel.com/en-us/forums/intel-c-compiler/topic/709454
  {
    using Multi = util::multi_array<R,D,S>;
    return (*reinterpret_cast<Multi*>(&data))[i];
  }

  S (&data)[Size];                              ///!< The external storage
};

/// Special-case Rank 0 tensors.
///
/// They are basically just scalars and should be treated as such wherever
/// possible.
template <int D, class S>
class Tensor<0,D,S>
{
 public:
  auto& operator()() {
    return data;
  }

  auto& operator[](int i) {
    assert(i==0);
    return data;
  }

  S data;
};

} // namespace ttl

#endif // #ifndef TTL_TENSOR_IMPL_H
