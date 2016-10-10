// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Expressions/Expression.h>
#include <ttl/Expressions/index_type.h>
#include <ttl/Expressions/promote.h>

namespace ttl {
namespace expressions {

/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
class TensorProduct;

/// The expression Traits for TensorProduct.
///
/// @tparam           L The type of the left hand expression.
/// @tparam           R The type of the right hand expression.
template <class L, class R>
struct expression_traits<TensorProduct<L, R>>
{
  using free_type = outer_type<typename expression_traits<L>::free_type,
                               typename expression_traits<R>::free_type>;
  using scalar_type = promote<L, R>;
  using dimension = typename expression_traits<L>::dimension;
};

namespace detail {
/// Contract a set of dimensions in an expression.
///
/// @tparam           E The expression type of the tensor product.
/// @tparam           I The index type of the product.
/// @tparam           n The next index being contracted.
/// @tparam           N The total number of indices to iterate.
template <class E, class I, int n,
          int N = std::tuple_size<I>::value>
struct contract_impl
{
  static scalar_type<E> op(const E& e, I index) {
    scalar_type<E> s(0);
    for (int i = 0; i < dimension<E>::value; ++i) {
      std::get<n>(index).set(i);
      s += contract_impl<E, I, n + 1>::op(e, index);
    }
    return s;
  }
};

template <class E, class I, int N>
struct contract_impl<E, I, N, N>
{
  static constexpr scalar_type<E> op(const E& e, I i) {
    return e(i);
  }
};
}

template <int n, class E, class I>
inline constexpr scalar_type<E> contract(const E& e, I index) {
  return detail::contract_impl<E, I, n>::op(e, index);
}

/// The TensorProduct expression implementation.
template <class L, class R>
class TensorProduct : Expression<TensorProduct<L, R>>
{
  static_assert(is_expression<L>::value, "Operand is not Expression");
  static_assert(is_expression<R>::value, "Operand is not Expression");

  /// The type of the inner dimensions, needed during contraction.
  ///
  /// @todo C++14 scope this inside of operator[]
  using hidden_type = intersection<typename expression_traits<L>::free_type,
                                   typename expression_traits<R>::free_type>;
 public:
  TensorProduct(L lhs, R rhs) : lhs_(lhs), rhs_(rhs) {
  }

  template <class I>
  constexpr scalar_type<TensorProduct> operator[](I i) const {
    return contract<
      std::tuple_size<I>::value>(*this, std::tuple_cat(i, hidden_type()));
  }

  /// Used as a leaf call during contraction.
  template <class I>
  constexpr scalar_type<TensorProduct> operator()(I i) const {
    return lhs_[i] * rhs_[i];
  }

 private:
  L lhs_;
  R rhs_;
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
