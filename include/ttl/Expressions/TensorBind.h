// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_BIND_H
#define TTL_EXPRESSIONS_TENSOR_BIND_H

#include <ttl/Index.h>
#include <ttl/Tensor.h>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/transform.h>
#include <ttl/util/is_equivalent.h>
#include <ttl/Expressions/traits.h>
#include <utility>

namespace ttl {
namespace expressions {

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
struct traits<TensorBind<Tensor, Index>> : traits<Tensor>
{
  using free_type = Index;
};

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
/// @tparam           n The current dimension that we need to traverse.
/// @tparam           L The type of the left-hand-side expression.
/// @tparam           R The type of the right-hand-side expression.
/// @tparam           M The total number of free indices to enumerate.
template <int n, class L, class R, int M = free_size<L>::value>
struct evaluate {
  static void op(L& lhs, const R& rhs, free_type<L> index) {
    for (int i = 0; i < dimension<L>::value; ++i) {
      std::get<n>(index) = i;
      evaluate<n + 1, L, R>::op(lhs, rhs, index);
    }
  }
};

/// The recursive base case for evaluation.
///
/// In this base case we have built the entire index and simply need to assign
/// the evaluation of the right hand side to the evaluation of the left hand
/// side.
///
/// @tparam           L The type of the left-hand-side expression.
/// @tparam           R The type of the right-hand-side expression.
/// @tparam           M The number of free indices (bounds recursion).
template <class L, class R, int M>
struct evaluate<M, L, R, M> {
  static void op(L& lhs, const R& rhs, free_type<L> index) {
    lhs.set(index, rhs.get(index));
  }
};

template <class Tensor, class Index>
class TensorBind : public Expression<TensorBind<Tensor, Index>>
{
 public:
  /// A TensorBind expression just keeps a reference to the Tensor it wraps.
  ///
  /// @tparam         t The underlying tensor.
  explicit TensorBind(Tensor& t) : t_(t) {
  }

  /// Default assignment, move, and copy should work fine when the
  /// right-hand-side is a tensor expression of the same type, meaning it has
  /// the same underlying tensor shape (Rank, Dimension, and Scalar type) and is
  /// being indexed through the same indices.
  ///
  /// @{
  TensorBind(TensorBind&&) = default;
  TensorBind& operator=(TensorBind&& rhs) {
    t_ = rhs.t_;
    return *this;
  };

  TensorBind(const TensorBind&) = default;
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
  constexpr const scalar_type<TensorBind> get(I i) const {
    return t_.get(transform<free_type<TensorBind>>(i));
  }

  /// This set operation is used during evaluation to set a left-hand-side
  /// element.
  template <class I, class U>
  void set(I index, U scalar) const {
    t_.set(transform<free_type<TensorBind>>(index), scalar);
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// This is the core functionality provided by the TensorBind expression. Its
  /// job is to loop through the index space defined by its IndexType and
  /// evaluate the right hand side for the remapped index pack.
  ///
  /// @code
  ///   A(i, j) = B(j, i)
  ///
  ///   for (i in 0..Dimension-1)
  ///     for (j in 0..Dimension-1)
  ///       a = linearize(i, j)
  ///       b = linearize(j, i)
  ///       A[a] = B[b]
  ///
  /// @code
  ///
  /// The only way that I can figure out to do this is through recursion. We use
  /// the recursive apply template to force that expansion to happen at runtime
  /// and insure that there is the opportunity for inlining and optimization.
  ///
  /// @tparam         R Right-hand-side expression.
  /// @tparam    (anon) Restrict this operation to expressions that match.
  template <class R>
  TensorBind& operator=(const R& rhs) {
    static_assert(dimension<R>::value == dimension<TensorBind>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(util::is_equivalent<free_type<TensorBind>, free_type<R>>::value,
                  "Attempted assignment of incompatible Tensors");
    evaluate<0, TensorBind, R>::op(*this, rhs, {});
    return *this;
  }

 private:
  Tensor& t_;
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
