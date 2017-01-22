// -*- C++ -*-------------------------------------------------------------------
/// This header includes the functionality for binding and evaluating tensor
/// expressions.
///
/// Its primary template is the Bind expression which is generated using
/// a Tensor's operator(), e.g., A(i,j). It also defines the Expression traits
/// for the Bind expression, as well as some ::detail metafunctions to
/// process type sets.
// -----------------------------------------------------------------------------
#ifndef TTL_EXPRESSIONS_TENSOR_BIND_H
#define TTL_EXPRESSIONS_TENSOR_BIND_H


#include <ttl/Index.h>
#include <ttl/Tensor.h>
#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/contract.h>
#include <ttl/Expressions/pack.h>
#include <ttl/Expressions/traits.h>
#include <ttl/Expressions/transform.h>
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
class Bind;

/// The expression traits for Bind expressions.
///
/// @tparam      Tensor The class for the underlying tensor.
/// @tparam       Index The indices bound to this expression.
template <class Tensor, class Index>
struct traits<Bind<Tensor, Index>> : public traits<rinse<Tensor>>
{
  using outer_type = unique<Index>;
  using inner_type = duplicate<Index>;
  using rank = typename std::tuple_size<outer_type>::type;
};

template <class Tensor, class Index>
class Bind : public Expression<Bind<Tensor, Index>>
{
  using Outer = unique<Index>;

 public:
  /// A Bind expression just keeps a reference to the Tensor it wraps.
  ///
  /// @tparam         t The underlying tensor.
  constexpr Bind(Tensor& t) noexcept : t_(t) {
  }

  /// Default assignment, move, and copy should work fine when the
  /// right-hand-side is a tensor expression of the same type, meaning it has
  /// the same underlying tensor shape (Rank, Dimension, and Scalar type) and is
  /// being indexed through the same indices.
  ///
  /// @nb gcc is happy defaulting these but icc 16 won't
  /// @{
  constexpr Bind(const Bind& rhs) noexcept : t_(rhs.t_) {
  }

  constexpr Bind(Bind&& rhs) noexcept : t_(rhs.t_) {
  }

  Bind& operator=(Bind&& rhs) {
    t_ = rhs.t_;
    return *this;
  }

  Bind& operator=(const Bind& rhs) {
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
  ///                   Rank of this Bind.
  ///
  /// @returns          The scalar value at the linearized offset.
  template <class I>
  constexpr auto eval(I i) const {
    return contract<Bind>(i, [&](auto index) {
        // intel 16.0 can't handle the "transform" symbol here without the
        // namespace
        return t_.eval(ttl::expressions::transform<Index>(index));
      });
  }

  /// This eval operation is used during evaluation to set a left-hand-side
  /// element.
  constexpr auto& eval(Outer index) {
    static_assert(std::is_same<Outer, Index>::value,
                  "LHS evaluation must not contain a contraction");
    return t_.eval(index);
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam         E Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  Bind& operator=(E&& rhs) {
    static_assert(dimension<E>::value == dimension<Tensor>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<E>>::value,
                  "Attempted assignment of incompatible Tensors");
    apply<>::op(Outer{}, [&](Outer i) { eval(i) = rhs.eval(i); });
    return *this;
  }

  /// Accumulate from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam         E Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class E>
  Bind& operator+=(E&& rhs) {
    static_assert(dimension<E>::value == dimension<Tensor>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<E>>::value,
                  "Attempted assignment of incompatible Tensors");
    E e = std::move(rhs);
    apply<>::op(Outer{}, [&,this](Outer i) { eval(i) += rhs.eval(i); });
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
  template <int n = 0, int M = std::tuple_size<Outer>::value>
  struct apply
  {
    /// The evaluation routine just iterates through the values of the nth
    /// dimension of the tensor, recursively calling the template.
    ///
    /// @tparam      Op The lambda to evaluate for each index.
    /// @param    index The partially constructed index.
    template <class Op>
    static void op(Outer index, Op&& f) {
      for (int i = 0; i < dimension<Tensor>::value; ++i) {
        std::get<n>(index) = i;
        apply<n + 1>::op(index, std::forward<Op>(f));
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
    static void op(Outer index, Op&& f) {
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
      f(Outer{});
    }
  };

  Tensor& t_;                                   ///<! The underlying tensor.
};
} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
