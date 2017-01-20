// -*- C++ -*-------------------------------------------------------------------
/// This header includes the functionality for binding and evaluating tensor
/// expressions.
///
/// Its primary template is the TensorBind expression which is generated using
/// a Tensor's operator(), e.g., A(i,j). It also defines the Expression traits
/// for the TensorBind expression, as well as some ::detail metafunctions to
/// process type sets.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_TENSOR_BIND_H
#define TTL_EXPRESSIONS_TENSOR_BIND_H


#include <ttl/Index.h>
#include <ttl/Tensor.h>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/iif.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>
#include <utility>

namespace ttl {
namespace expressions {
namespace detail {
// -----------------------------------------------------------------------------
/// A metafunction to split an index set into free and contracted types.
///
/// When the user binds a tensor to a set of indices they can specify self
/// contraction by binding the same index to multiple slots. In this case, TTL
/// needs to generate outer loops over the free indices and inner loops over the
/// contracted indices.
///
/// @code
///   A(j) = B(i,i,j);
///
///   for j:(0,D-1)
///     for i:(0,D-1)
///       A[j] += B[i][i][j]
/// @code
///
/// @tparam           T The initial set of indices.
/// @tparam           U The free indices in @p T.
/// @tparam           V The contracted indices in @p T.
// -----------------------------------------------------------------------------
template <class T, class U, class V>
struct split;
}

/// The expression that represents binding a Tensor to an Index map.
///
/// This expression is the leaf expression for all tensor operations, and
/// results from expressions that look like `x(i)`, `A(j, k, l)`, etc. It allows
/// multidimensional indexing into tensors, tensor assignment, and natively
/// supports shuffle operations.
///
/// @tparam      Tensor The tensor type.
/// @tparam       Index The index map for this expression.
template <class Tensor, class Index>
class TensorBind;

/// The expression traits for TensorBind expressions.
///
/// @tparam      Tensor The class for the underlying tensor.
/// @tparam       Index The indices bound to this expression.
template <class Tensor, class Index>
struct traits<TensorBind<Tensor, Index>> : public traits<rinse<Tensor>>
{
  // using split = detail::split<Index, std::tuple<>, std::tuple<>>;
  // static_assert(std::is_same<Index, typename split::free_type>::value, "not same");
 // public:
 //  using free_type = typename split::free_type;
  using free_type = Index;
};

template <class Tensor, class Index>
class TensorBind : public Expression<TensorBind<Tensor, Index>>
{
 public:
  /// A TensorBind expression just keeps a reference to the Tensor it wraps.
  ///
  /// @tparam         t The underlying tensor.
  constexpr TensorBind(Tensor& t) noexcept : t_(t) {
  }

  /// Default assignment, move, and copy should work fine when the
  /// right-hand-side is a tensor expression of the same type, meaning it has
  /// the same underlying tensor shape (Rank, Dimension, and Scalar type) and is
  /// being indexed through the same indices.
  ///
  /// @nb gcc is happy defaulting these but icc 16 won't
  /// @{
  constexpr TensorBind(const TensorBind& rhs) noexcept : t_(rhs.t_) {
  }

  constexpr TensorBind(TensorBind&& rhs) noexcept : t_(rhs.t_) {
  }

  TensorBind& operator=(TensorBind&& rhs) {
    t_ = rhs.t_;
    return *this;
  }

  TensorBind& operator=(const TensorBind& rhs) {
    t_ = rhs.t_;
    return *this;
  }
  /// @}

  /// The index operator maps the index array using the normal interpretation of
  /// multidimensional indexing, using index 0 as the most-significant-index.
  ///
  /// @code
  ///        Rank = n = sizeof(index)
  ///   Dimension = k
  ///       index = {a, b, c, ..., z}
  ///           i = a*k^(n-1) + b*k^(n-2) + c*k^(n-3) + ... + z*k^0
  /// @code
  ///
  /// @param      index The index array, which must be the same length as the
  ///                   Rank of this TensorBind.
  ///
  /// @returns          The scalar value at the linearized offset.
  template <class I>
  constexpr scalar_type<Tensor> eval(I i) const {
    return t_.eval(transform<Index>(i));
  }

  /// This eval operation is used during evaluation to set a left-hand-side
  /// element.
  constexpr scalar_type<Tensor>& eval(Index index) {
    return t_.eval(index);
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam         E Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  TensorBind& operator=(E&& rhs) {
    static_assert(dimension<E>::value == dimension<Tensor>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Index, free_type<E>>::value,
                  "Attempted assignment of incompatible Tensors");
    apply<>::op([&](Index i) { eval(i) = rhs.eval(i); });
    return *this;
  }

  /// Accumulate from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam         E Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  TensorBind& operator+=(E&& rhs) {
    static_assert(dimension<E>::value == dimension<Tensor>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Index, free_type<E>>::value,
                  "Attempted assignment of incompatible Tensors");
    E e = std::move(rhs);
    apply<>::op([&,this](Index i) { eval(i) += rhs.eval(i); });
    return *this;
  }

 private:
  /// The recursive template class that evaluates tensor expressions.
  ///
  /// The fundamental goal of ttl is to generate loops over tensor dimension,
  /// evaluating the right-hand-side of the expression for each of the
  /// combinations of inputs on the left hand side.
  ///
  /// @code
  ///   for i : 0..D-1
  ///    for j : 0..D-1
  ///      ...
  ///        for n : 0..D-1
  ///           lhs(i,j,...,n) = rhs(i,j,...,n)
  /// @code
  ///
  /// We use recursive template expansion to generate these "loops" statically, by
  /// dynamically enumerating the index dimensions. There is presumably a static
  /// enumeration technique, as our bounds are all known statically.
  ///
  /// @note I tried to use a normal recursive function but there were typing
  ///       errors for the case where M=0 (i.e., the degenerate scalar tensor
  ///       type).
  ///
  /// @tparam           n The current dimension that we need to traverse.
  /// @tparam           M The total number of free indices to enumerate.
  template <int n = 0, int M = free_size<TensorBind>::value>
  struct apply
  {
    /// The evaluation routine just iterates through the values of the nth
    /// dimension of the tensor, recursively calling the template.
    ///
    /// @tparam      Op The lambda to evaluate for each index.
    /// @param    index The partially constructed index.
    template <class Op>
    static void op(Op&& f, Index index = {}) {
      for (int i = 0; i < dimension<Tensor>::value; ++i) {
        std::get<n>(index) = i;
        apply<n + 1>::op(std::forward<Op>(f), index);
      }
    }
  };

  /// The base case for tensor evaluation.
  ///
  /// @tparam         M The total number of dimensions to enumerate.
  template <int M>
  struct apply<M, M>
  {
    template <class Op>
    static void op(Op&& f, const Index& index) {
      f(index);
    }

    /// Specialize for the case where the index space is empty (i.e., the
    /// left-hand-side is a scalar as in an inner product).
    ///
    /// @code
    ///   c() = a(i) * b(i)
    /// @code
    template <class Op>
    static void op(Op&& f) {
      f(Index{});
    }
  };

  Tensor& t_;                                   ///<! The underlying tensor.
};


namespace detail {
/// Base case for split is when we've processed all of the parameters.
template <template <class...> class Pack, class U, class V>
struct split<Pack<>, U, V> {
  using free_type = U;
  using contracted_type = V;
};

/// The implementation of the split metafunction.
///
/// This metafunction recursively splits the set @p T into two sets, @p U and @p
/// V, where @p U is the set of free indices and @p V is the set of contracted
/// indices. It does this using a template specialization that picks off the
/// head type, T0, in @p T and checks to see if it appears again in the tail of
/// @p T. If it is not repeated then it appends T0 to @p U and recurses on the
/// tail of @p T. Otherwise, it appends T0 to @p V and recurses on the tail of
/// @p T, after removing T0 from the tail.
///
/// @tparam         pack The class carrying the index pack (e.g., std::tuple).
/// @tparam           T0 The next type to process.
/// @tparam         T... The tail of the index set.
/// @tparam         U... The current set of free indices.
/// @tparam         V... The current set of contracted indices.
// template <template <class...> class Pack, class T0, class... T, class U, class V>
// struct split<Pack<T0, T...>, U, V> {
//  private:
//   using duplicate = util::contains<T0, T...>;
//   using tail = difference<Pack<T...>, Pack<T0>>;
//   using free = split<tail, util::append<T0, U>, V>;
//   using contracted = split<tail, U, util::append<T0, V>>;
//   using next = util::iif<duplicate, contracted, free>;

//  public:
//   using free_type = typename next::free_type;
// };
}
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
