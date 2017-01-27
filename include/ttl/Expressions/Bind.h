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
/// The expression that represents binding an index space to a subtree.
///
/// @tparam           E The subtree type.
/// @tparam       Index The index map for this expression.
template <class E, class Index>
class Bind;

/// The expression traits for Bind expressions.
///
/// Bind currently strips the cvref keywords from the bound type.
///
/// @tparam           E The subtree expression type.
/// @tparam       Index The indices bound to this expression.
template <class E, class Index>
struct traits<Bind<E, Index>> : public traits<rinse<E>>
{
  using outer_type = unique<non_integral<Index>>;
  using inner_type = duplicate<non_integral<Index>>;
  using rank = typename std::tuple_size<outer_type>::type;
};

template <class E, class Index>
class Bind : public Expression<Bind<E, Index>>
{
  using Outer = outer_type<Bind>;

 public:
  /// A Bind expression keeps a reference to the E it wraps, and a
  ///
  /// @tparam         t The underlying tensor.
  constexpr Bind(E& t, const Index i) noexcept : t_(t), i_(i) {
  }

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
        return t_.eval(ttl::expressions::transform(i_, index));
      });
  }

  /// Assignment from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam       RHS Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class RHS>
  Bind& operator=(RHS&& rhs) {
    static_assert(dimension<E>::value == dimension<RHS>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    apply<>::op(Outer{}, [&](Outer i) { t_.eval(i) = rhs.eval(i); });
    return *this;
  }

  /// Accumulate from any right hand side expression that has an equivalent
  /// index pack.
  ///
  /// @tparam       RHS Type of the Right-hand-side expression.
  /// @param        rhs The right-hand-side expression.
  /// @returns          A reference to *this for chaining.
  template <class RHS>
  Bind& operator+=(RHS&& rhs) {
    static_assert(dimension<E>::value == dimension<RHS>::value,
                  "Cannot operate on expressions of differing dimension");
    static_assert(equivalent<Outer, outer_type<RHS>>::value,
                  "Attempted assignment of incompatible Expressions");
    apply<>::op(Outer{}, [&](Outer i) { t_.eval(i) += rhs.eval(i); });
    return *this;
  }

  /// Basic print-to-stream functionality.
  ///
  /// This iterates through the index space and prints the index and results to
  /// the stream.
  std::ostream& print(std::ostream& os) const {
    apply<>::op(Outer{}, [&,this](Outer i) {
        print_pack(os, i) << ": " << eval(i) << "\n";
      });
    return os;
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
      for (int i = 0; i < dimension<E>::value; ++i) {
        std::get<n>(index).set(i);
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

  E& t_;                                        ///<! The underlying tree.
  const Index i_;                               ///<! The bound index.
};

template <class T, class Index>
Bind<T, Index> make_bind(T& t, const Index i) {
  return Bind<T,Index>(t, i);
}

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_BIND_H
