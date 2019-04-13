#pragma once

#include "ttl/mp/duplicate.hpp"
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

  constexpr operator iif_t<R == 0, T, void>() && noexcept {
    return op_(Index{});
  }

  constexpr operator iif_t<R == 0, T, void>() & noexcept {
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
  return make_expression<Op::D, Index>(
      [op=std::move(op)](Index index) {
        return -op(index);
      });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator/(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
        return l(index) / r;
      });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator%(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
        return l(index) % r;
      });
}

template <class Lhs, class Rhs,
          typename Lhs::is_expression** = nullptr,
          std::enable_if_t<is_scalar<Rhs>::value, void**> = nullptr>
constexpr auto operator*(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Lhs::Index;
  return make_expression<Lhs::D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
        return l(index) * r;
      });
}

template <class Lhs, class Rhs,
          std::enable_if_t<is_scalar<Lhs>::value, void**> = nullptr,
          typename Rhs::is_expression** = nullptr>
constexpr auto operator*(Lhs lhs, Rhs rhs) noexcept {
  using Index = typename Rhs::Index;
  return make_expression<Rhs::D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
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
  return make_expression<D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
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
  return make_expression<D, Index>(
      [l=std::move(lhs), r=std::move(rhs)](Index index) {
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
  return make_expression<D, Outer>(
      [l=std::move(lhs), r=std::move(rhs)](Outer index)
      {
        auto i = std::tuple_cat(index, Inner{});
        using Index = decltype(i);
        static constexpr int n = std::tuple_size<Outer>::value;
        static constexpr int N = std::tuple_size<Index>::value;
        return contract<D, n, N>::op(i,
            [l=std::move(l), r=std::move(r)](Index index) {
              return l(select<lIndex>(index)) * r(select<rIndex>(index));
            });
      });
}

template <int D = -1, class... Index>
constexpr auto zero(Index... index) noexcept {
  return make_expression<D, std::tuple<Index...>>([](auto) { return 0; });
}
}
