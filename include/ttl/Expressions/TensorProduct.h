// -*- C++ -*-
#ifndef TTL_EXPRESSIONS_TENSOR_PRODUCT_H
#define TTL_EXPRESSIONS_TENSOR_PRODUCT_H

#include <ttl/Pack.h>
#include <ttl/Index.h>
#include <ttl/Expressions/Expression.h>

namespace ttl {
namespace expressions {

// template <class E>
// using full_type = typename expression_traits<E>::full_type;

// template <class E>
// struct full_rank {
//   static constexpr int value = sizeof...(full_type<E>);
// };

// template <class E>
// using full_index = IndexSet<free_rank<E>::value>;

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
  using free_type = symdif<typename expression_traits<L>::free_type,
                           typename expression_traits<R>::free_type>;
  using scalar_type = promote<L, R>;
  static constexpr int dimension = expression_traits<L>::dimension;
};

template <class E>
struct union_size;

template <class L, class R>
struct union_size<TensorProduct<L, R>> {
  using type = unite<typename expression_traits<L>::free_type,
                     typename expression_traits<R>::free_type>;
  static constexpr int value = size<type>::value;
};

template <class E>
using union_index = IndexSet<union_size<E>::value>;

/// The TensorProduct expression implementation.
template <class L, class R>
class TensorProduct : Expression<TensorProduct<L, R>>
{
 public:
  TensorProduct(L lhs, R rhs) : lhs_(lhs), rhs_(rhs) {
  }

  scalar_type<TensorProduct> operator()(free_index<TensorProduct> i) const {
    static constexpr int size = free_size<TensorProduct>::value;

    // Extend the index set with enough space for the inner, contracted size,
    // copy the outer dimensions into it, and then perform the contraction of
    // the inner indices.
    union_index<TensorProduct> j;
    std::copy(&i[0], &i[size], &j[0]);
    return contract(j);
  }

 private:
  template <int n = free_size<TensorProduct>::value,
            int N = union_size<TensorProduct>::value>
  scalar_type<TensorProduct> contract(union_index<TensorProduct> i) const {
    if (n < N) {
      scalar_type<TensorProduct> s(0);
      for (i[n] = 0; i[n] < dimension<TensorProduct>::value; ++i[n]) {
        s += contract<n + 1>(i);
      }
      return s;
    }
    else {
      return lhs_(i) * rhs_(i);
    }
  }

  L lhs_;
  R rhs_;
};

} // namespace expressions
} // namespace ttl

#endif // TTL_EXPRESSIONS_TENSOR_PRODUCT_H
