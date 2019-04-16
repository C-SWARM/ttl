#pragma once

#include "ttl/mp/duplicate.hpp"
#include "ttl/mp/non_integer.hpp"
#include "ttl/mp/unique.hpp"
#include "ttl2/mp.hpp"

namespace ttl {
template <int Dimension, class IndexType, class Op>
class Expr {
 public:
  using    is_expression = void;
  using            Index = IndexType;
  using                T = decltype(std::declval<Op>()(std::declval<Index>()));
  static constexpr int R = std::tuple_size<Index>::value;
  static constexpr int D = Dimension;

  Expr(Op op) : op_(std::move(op)) {
  }

  constexpr auto operator()(Index index) const & noexcept {
    return op_(index);
  }

  constexpr auto operator()(Index index) && noexcept {
    return op_(index);
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr auto operator()(U... index) const & noexcept {
    return this->operator()(std::make_tuple(index...));
  }

  template <class... U,
            class = std::enable_if_t<all_integer<U...>::value>>
  constexpr auto operator()(U... index) && noexcept {
    return this->operator()(std::make_tuple(index...));
  }

  constexpr operator iif_t<R == 0, T, void>() const & noexcept {
    return op_(Index{});
  }

  constexpr operator iif_t<R == 0, T, void>() && noexcept {
    return op_(Index{});
  }

 protected:
  Op op_;
};

template <int Dimension, class Index, class Op>
constexpr Expr<Dimension, Index, Op> make_expression(Op op) noexcept {
  return { std::move(op) };
}

template <class Op,
          typename Op::is_expression** = nullptr>
constexpr auto operator-(Op op) noexcept {
  using Index = typename Op::Index;
  return make_expression<Op::D, Index>([op=std::move(op)](Index index) {
    return -op(index);
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator/(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l(index) / r;
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator%(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l(index) % r;
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator*(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l(index) * r;
  });
}

template <class Lhs, class Rhs,
          std::enable_if_t<is_scalar<Lhs>::value, void**> = nullptr,
          typename Rhs::is_expression** = nullptr>
constexpr auto operator*(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Rhs::Index;
  return make_expression<Rhs::D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l * r(index);
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          typename Rhs::is_expression** = nullptr>
constexpr auto operator+(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
  constexpr auto D = dimension<Lhs, Rhs>::value;
  static_assert(compatible, "Expressions must export the same index type");
  return make_expression<D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l(index) + r(index);
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          typename Rhs::is_expression** = nullptr>
constexpr auto operator-(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  constexpr auto compatible = std::is_same<Index, typename Rhs::Index>();
  constexpr auto D = dimension<Lhs, Rhs>::value;
  static_assert(compatible, "Expressions must export the same index type");
  return make_expression<D, Index>([l=std::move(lhs), r=std::move(rhs)](Index index) {
    return l(index) - r(index);
  });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          typename Rhs::is_expression** = nullptr>
constexpr auto operator*(Lhs lhs, Rhs rhs) noexcept {
  using lIndex = typename Lhs::Index;
  using rIndex = typename Rhs::Index;
  using Outer = mp::xor_t<lIndex, rIndex>;
  using Inner = mp::and_t<lIndex, rIndex>;
  constexpr auto D = dimension<Lhs, Rhs>::value;
  return make_expression<D, Outer>([l=std::move(lhs), r=std::move(rhs)](Outer outer) {
    return contract<D, Inner>(outer, [l=std::move(l), r=std::move(r)](auto index) {
      return l(subset<lIndex>(index)) * r(subset<rIndex>(index));
    });
  });
}

template <int D = -1, class... Index>
constexpr auto zero(Index... index) noexcept {
  return make_expression<D, std::tuple<Index...>>([](auto) { return 0; });
}

template <class Index, class Op>
constexpr auto permutation(Op op) noexcept {
  constexpr auto D = Op::D;
  using ChildIndex = typename Op::Index;
  return make_expression<D, Index>([op=std::move(op)](Index index) {
    return op(select<ChildIndex>(index));
  });
}

template <class ChildIndex, class Op>
constexpr auto slice(ChildIndex index, Op op) noexcept {
  constexpr auto D = Op::D;
  using Index = mp::non_integer_t<ChildIndex>;
  return make_expression<D, Index>([i=index,op=std::move(op)](Index index) {
    return op(select(i, index));
  });
}
}
